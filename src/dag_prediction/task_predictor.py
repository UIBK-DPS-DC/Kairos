import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
from collections import Counter

import src.database.db_interface as db
import src.dag_prediction.fallback_model as fallback_model
import src.dag_prediction.global_models as global_models
import src.dag_prediction.setar_tree as setar
import src.dag_prediction.space_models as space

SplitTimeModel = Literal["ridge", "setar", "linear", "lightgbm"]
History = List[tuple[datetime, str]]  # chronological (timestamp, provider) pairs


@dataclass
class TaskPrediction:
    """Holds prediction results for the next task execution."""

    def __init__(
        self, predicted_next_access: datetime, predicted_provider_id: str = None
    ):
        self.predicted_next_access = predicted_next_access
        self.predicted_provider_id = predicted_provider_id

    def __repr__(self):
        return (
            f"TaskPrediction("
            f"  predicted_next_access = {self.predicted_next_access.isoformat()}"
            f"  predicted_provider_id = {self.predicted_provider_id}"
            f")"
        )


def train_split_models(
    histories: Dict[str, History], time_model: SplitTimeModel = "ridge"
) -> dict:
    """Train the per-task time model and the Markov provider model.

    The time model is None when there is too little history to train it (cold
    start); ``"linear"`` needs no trained model.
    """
    times = {key: [ts for ts, _ in h] for key, h in histories.items()}
    try:
        if time_model == "setar":
            trained = setar.train_setar_tree(times)
        elif time_model in ("ridge", "lightgbm"):
            trained = global_models.train_global_model(times, time_model)
        else:
            trained = None
    except ValueError:
        trained = None
    return {
        "time_model": time_model,
        "time": trained,
        "space": space.train_markov({key: [p for _, p in h] for key, h in histories.items()}),
    }


def _unix(ts) -> float:
    # Naive timestamps are converted the same way in both directions
    # (pd.Timestamp(..., unit="s")), so only differences matter.
    return pd.Timestamp(ts).timestamp()


MAX_ROLLOUT_STEPS = 100


def _steps_until(last, one_step, target) -> int:
    """Provider steps from the last observed execution to the one at ``target``.

    Executions are spaced by the predicted interval ``one_step - last``.
    Rounding (not flooring) tolerates the jitter of individual start times;
    executions that ran but are not ingested yet are counted too.
    """
    if one_step is None or target is None:
        return 1
    step = _unix(one_step) - _unix(last)
    if step <= 0:
        return 1
    return int(min(MAX_ROLLOUT_STEPS, max(1, round((_unix(target) - _unix(last)) / step))))


class SplitPredictor:
    """Predicts time and provider separately; database-independent.

    With a pretrained ``fallback`` (see fallback_model.py):
    - time: the task's own model (from :func:`train_split_models`) once the task
      has ``min_history`` executions, the fallback before that;
    - provider: the task's own Markov model as soon as it has seen exactly the
      current provider context, the fallback otherwise.
    The fallback also fills in whenever the own models give no prediction.
    Without a fallback the own models are always used.
    """

    def __init__(
        self,
        models: dict,
        fallback: Optional[dict] = None,
        min_history: int = 10,
        number_of_inference_timestamps: int = 5,
    ):
        self.models = models
        self.fallback = fallback
        self.min_history = min_history
        self.number_of_inference_timestamps = number_of_inference_timestamps

    def predict(
        self, key: str, history: History, now=None, at=None
    ) -> Optional[TaskPrediction]:
        """Predict the next execution of series ``key`` after ``now`` (default: wall clock).

        ``at``: an execution is known to start at that time (e.g. the child of a
        task that just finished); it is predicted instead of the next one after
        ``now``. Either way the provider sequence is rolled forward over the
        executions between the last observed one and the predicted one, which
        covers runs that already happened but are not in ``history`` yet.
        """
        if not history:
            return None
        times = [ts for ts, _ in history]
        providers = [p for _, p in history]

        next_access = at if at is not None else self._time(key, times, now)
        if next_access is None:
            return None

        rolled = list(providers)
        provider = None
        steps = _steps_until(times[-1], self._time(key, times, times[-1]), next_access)
        for _ in range(steps):
            nxt = self._next_provider(key, rolled)
            if nxt is None:
                break
            provider = nxt
            rolled.append(nxt)
        if provider is None:
            provider = Counter(providers).most_common(1)[0][0]
        return TaskPrediction(predicted_next_access=next_access, predicted_provider_id=provider)

    def _time(self, key: str, times: list, now) -> Optional[datetime]:
        if self.fallback is None:
            return self._own_time(key, times, now)
        return (
            self._own_time(key, times, now) if len(times) >= self.min_history else None
        ) or fallback_model.predict_next_time(self.fallback, times, now)

    def _next_provider(self, key: str, providers: List[str]) -> Optional[str]:
        if self.fallback is None:
            return space.predict_markov(self.models["space"], key, providers)
        return space.predict_markov(
            self.models["space"], key, providers, backoff=False
        ) or fallback_model.predict_next_provider(self.fallback, providers)

    def _own_time(self, key: str, times: list, now) -> Optional[datetime]:
        recent = times[-self.number_of_inference_timestamps:]
        kind, model = self.models["time_model"], self.models["time"]
        if kind == "linear":
            if len(recent) < 2:
                return None
            now_unix = _unix(pd.Timestamp.now() if now is None else now)
            return pd.Timestamp(linear_next_unix([_unix(t) for t in recent], now_unix), unit="s")
        if model is None:
            return None
        if kind == "setar":
            if len(recent) < model["lag"] + 1:
                return None
            return setar.predict_next_for_pipeline_with_timestamps(model, key, recent, now=now)
        return global_models.predict_next_for_pipelines(model, {key: recent}, now).get(key)


class TaskPredictor:
    """
    Predicts when and where a task will be executed next.

    Args:
        db_manager: Database interface for reading task execution history.
        model: Pre-trained SETARTree model dict.  If ``None`` a new model is trained
            from all historical data in the database.
        number_of_inference_timestamps: How many recent timestamps to pass to the
            model at inference time.
        strategy: ``"setar"`` (default) uses the SETAR tree with forward-roll fix;
            ``"linear"`` delegates to :class:`LinearRegressionPredictor` instead.
            Both forecast per (task, provider) and pick the earliest provider.
            ``"split"`` predicts the time from the task's full series and the
            provider with a Markov model over the task's provider sequence.
        split_time_model: Time model of the ``"split"`` strategy. ``"ridge"``
            (default) had the lowest time error in evaluation/prediction.
        use_fallback: ``"split"`` only: use the pretrained fallback model
            (fallback_model.json) for tasks with little history of their own.
        min_history: ``"split"`` only: executions a task needs before its own
            models replace the fallback.
    """

    def __init__(
        self,
        db_manager: db.DBInterface,
        model: setar.SETARTree = None,
        number_of_inference_timestamps: int = 5,
        strategy: Literal["setar", "linear", "split"] = "setar",
        split_time_model: SplitTimeModel = "ridge",
        use_fallback: bool = True,
        min_history: int = 10,
    ):
        self.db_manager = db_manager
        self.strategy = strategy
        self.number_of_inference_timestamps = number_of_inference_timestamps
        if strategy == "setar":
            self.model = model if model is not None else self.newly_trained_model()
        elif strategy == "split":
            self.model = None
            histories = {}
            for t in self.db_manager.read_all_tasks():
                history = self._read_task_history(t.id, t.dag_id)
                if history:
                    histories[f"{t.dag_id}.{t.id}"] = history
            self._split_predictor = SplitPredictor(
                train_split_models(histories, split_time_model),
                fallback=fallback_model.load_fallback() if use_fallback else None,
                min_history=min_history,
                number_of_inference_timestamps=number_of_inference_timestamps,
            )
        else:
            self.model = None  # LinearRegressionPredictor does not use a pre-trained model
        self._linear_predictor = (
            LinearRegressionPredictor(
                db_manager=db_manager,
                number_of_inference_timestamps=number_of_inference_timestamps,
            )
            if strategy == "linear"
            else None
        )

    def newly_trained_model(self) -> setar.SETARTree:
        """Train the SETAR model using historical task execution data."""

        task_histories = {}
        for t in self.db_manager.read_all_tasks():
            used_providers = self.db_manager.read_providers_used_by_task(t.id, t.dag_id)
            for p in used_providers:
                task_executions = (
                    self.db_manager.read_task_executions_by_provider_and_task(
                        p, t.id, t.dag_id
                    )
                )
                timestamps = [
                    datetime.fromtimestamp(float(te.id)) for te in task_executions
                ]
                task_histories[f"{t.id}_{p}"] = timestamps

        return setar.train_setar_tree(task_histories)

    def _read_task_history(self, task_id: str, dag_id: str) -> List[tuple[datetime, str]]:
        """All executions of a task as chronological (timestamp, provider) pairs.

        The timestamp is the execution's first data access, which is the deadline
        for a replication, not the task start encoded in the execution id (the
        task may wait before it reads). An execution that accessed several
        providers is attributed to the one it accessed most, so every execution
        appears exactly once.
        """
        best: Dict[str, tuple[int, str]] = {}
        for te_id, provider, count in self.db_manager.read_access_counts_by_task(task_id, dag_id):
            best[te_id] = max(best.get(te_id, (0, "")), (count, provider))
        first_access = self.db_manager.read_first_access_times_by_task(task_id, dag_id)
        return sorted(
            (
                datetime.fromisoformat(first_access[te_id])
                if first_access.get(te_id)
                else datetime.fromtimestamp(float(te_id)),
                provider,
            )
            for te_id, (_, provider) in best.items()
        )

    def _predict_split(self, task_id: str, dag_id: str, at=None, now=None) -> Optional[TaskPrediction]:
        return self._split_predictor.predict(
            f"{dag_id}.{task_id}", self._read_task_history(task_id, dag_id), now=now, at=at
        )

    def predict_next_task_execution_on_provider(
        self, task_id: str, dag_id: str, provider_id: str
    ) -> Optional[TaskPrediction]:
        """Predict next execution time for a specific task and provider."""
        timestamps = [
            datetime.fromtimestamp(float(te.id))
            for te in self.db_manager.read_last_x_task_executions_by_provider_and_task(
                provider_id, task_id, dag_id, self.number_of_inference_timestamps
            )
        ]

        if not timestamps:
            return None
        required_timestamps = self.model["lag"] + 1
        if len(timestamps) < required_timestamps:
            return None

        next_access = setar.predict_next_for_pipeline_with_timestamps(
            model=self.model,
            prediction_object_name=f"{task_id}_{provider_id}",
            new_times=timestamps,
        )

        return TaskPrediction(
            predicted_next_access=next_access, predicted_provider_id=provider_id
        )

    def predict_next_task_execution(
        self,
        task_id: str,
        dag_id: str,
        at: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> Optional[TaskPrediction]:
        """Predict next access time and provider.

        ``at`` (``"split"`` only): the task is known to start at that time; see
        :meth:`SplitPredictor.predict`.
        ``now`` (``"split"`` only): predict the next execution after this time
        instead of after the wall clock.
        Delegates to :class:`LinearRegressionPredictor` when ``strategy='linear'``.
        """
        if self.strategy == "split":
            return self._predict_split(task_id, dag_id, at, now)
        if self.strategy == "linear" and self._linear_predictor is not None:
            return self._linear_predictor.predict_next_task_execution(task_id, dag_id)

        providers = self.db_manager.read_providers_used_by_task(task_id, dag_id)

        predictions = []

        for p in providers:
            prediction = self.predict_next_task_execution_on_provider(
                task_id, dag_id, p
            )
            if prediction:
                predictions.append(prediction)

        if not predictions:
            return None

        # Return the prediction with the earliest predicted next access time
        next_execution: TaskPrediction = min(
            predictions, key=lambda x: x.predicted_next_access
        )

        return TaskPrediction(
            predicted_next_access=next_execution.predicted_next_access,
            predicted_provider_id=next_execution.predicted_provider_id,
        )

    def predict_all_tasks(self) -> Dict[str, TaskPrediction]:
        """Helper to predict next execution for all known tasks."""
        results = {}

        for t in self.db_manager.read_all_tasks():
            prediction = self.predict_next_task_execution(task_id=t.id, dag_id=t.dag_id)
            if prediction:
                results[f"{t.id}_{t.dag_id}"] = prediction
        return results

    def predict_all_tasks_for_dag(self, dag_id: str) -> Dict[str, TaskPrediction]:
        """Helper to predict next execution for all known tasks in a specific DAG."""
        results = {}

        for t in self.db_manager.read_all_tasks():
            if t.dag_id == dag_id:
                prediction = self.predict_next_task_execution(task_id=t.id, dag_id=t.dag_id)
                if prediction:
                    results[f"{t.id}_{t.dag_id}"] = prediction
        return results


def linear_next_unix(unix_times: List[float], now_unix: float) -> float:
    """Extrapolate the next timestamp after ``now_unix`` from sorted unix times.

    Fits ``unix_time ~ index`` with a degree-1 polynomial and walks forward from
    the last observed index until the extrapolated timestamp lies in the future.
    """
    unix_times = np.asarray(unix_times, dtype=float)
    indices = np.arange(len(unix_times), dtype=float)

    # Fit degree-1 polynomial: unix_time ≈ slope * index + intercept
    slope, intercept = np.polyfit(indices, unix_times, deg=1)

    if slope <= 0:
        # Non-positive slope means timestamps are not increasing — fall back to mean
        # interval and extrapolate from the last observed timestamp.
        mean_interval = float(np.mean(np.diff(unix_times)))
        slope = mean_interval if mean_interval > 0 else 60.0
        intercept = unix_times[-1] - slope * indices[-1]

    # Walk forward from the last observed index until we land in the future.
    next_index = indices[-1] + 1.0
    next_unix = slope * next_index + intercept
    while next_unix <= now_unix:
        next_index += 1.0
        next_unix = slope * next_index + intercept
    return float(next_unix)


class LinearRegressionPredictor:
    """Stateless task-execution predictor based on linear extrapolation of timestamps.

    For each (task, provider) pair the predictor fetches the most recent N timestamps
    from the database, fits a degree-1 polynomial ``timestamp_unix ~ index`` using
    :func:`numpy.polyfit`, and walks forward from the last observed index until the
    extrapolated timestamp exceeds ``now``.

    This approach always produces a future prediction regardless of when it is called,
    making it robust to the case where Kairos runs long after the last DAG execution
    recorded in the database.

    Args:
        db_manager: Database interface for reading task execution history.
        number_of_inference_timestamps: How many recent timestamps to use when fitting
            the regression.  More points → smoother trend; too few → unstable.
    """

    def __init__(
        self,
        db_manager: db.DBInterface,
        number_of_inference_timestamps: int = 10,
    ):
        self.db_manager = db_manager
        self.number_of_inference_timestamps = number_of_inference_timestamps

    def _predict_for_provider(
        self, task_id: str, dag_id: str, provider_id: str
    ) -> Optional[TaskPrediction]:
        """Predict the next execution time for one (task, provider) pair."""
        executions = self.db_manager.read_last_x_task_executions_by_provider_and_task(
            provider_id, task_id, dag_id, self.number_of_inference_timestamps
        )
        if len(executions) < 2:
            # Need at least 2 points to fit a line.
            return None

        unix_times = sorted(float(te.id) for te in executions)
        next_unix = linear_next_unix(
            unix_times, datetime.now(tz=timezone.utc).timestamp()
        )
        return TaskPrediction(
            predicted_next_access=pd.Timestamp(next_unix, unit="s"),
            predicted_provider_id=provider_id,
        )

    def predict_next_task_execution(
        self, task_id: str, dag_id: str
    ) -> Optional[TaskPrediction]:
        """Predict next access time and provider across all providers used by the task."""
        providers = self.db_manager.read_providers_used_by_task(task_id, dag_id)
        predictions = [
            p
            for provider_id in providers
            if (p := self._predict_for_provider(task_id, dag_id, provider_id)) is not None
        ]
        if not predictions:
            return None
        # Return the earliest future prediction across providers.
        return min(predictions, key=lambda x: x.predicted_next_access)

    def predict_all_tasks_for_dag(self, dag_id: str) -> Dict[str, TaskPrediction]:
        """Predict next execution for every task in a specific DAG."""
        results = {}
        for t in self.db_manager.read_all_tasks():
            if t.dag_id == dag_id:
                prediction = self.predict_next_task_execution(
                    task_id=t.id, dag_id=t.dag_id
                )
                if prediction:
                    results[f"{t.id}_{t.dag_id}"] = prediction
        return results