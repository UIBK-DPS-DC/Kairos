import time
from datetime import datetime, timedelta

import pytest

from src.dag_prediction.fallback_model import load_fallback
from src.dag_prediction.task_predictor import SplitPredictor, TaskPredictor, train_split_models
from src.database.db_interface import Task, Task_Execution

PROVIDERS = ["A", "B", "C"]


class FakeDB:
    """Task_1 of dag_1 runs every 30 s, round-robin over A, B, C; the last run was 10 s ago."""

    def __init__(self, n_runs: int = 30):
        last = time.time() - 10
        self.executions = {p: [] for p in PROVIDERS}
        for i in range(n_runs):
            ts = last - (n_runs - 1 - i) * 30
            self.executions[PROVIDERS[i % len(PROVIDERS)]].append(
                Task_Execution(id=str(ts), task_id="Task_1", task_dag_id="dag_1")
            )

    def read_all_tasks(self):
        return [Task(id="Task_1", dag_id="dag_1")]

    def read_providers_used_by_task(self, task_id, dag_id):
        return list(self.executions)

    def read_task_executions_by_provider_and_task(self, provider_id, task_id, dag_id):
        return self.executions[provider_id]

    def read_access_counts_by_task(self, task_id, dag_id):
        return [
            (te.id, p, 1)
            for p in self.read_providers_used_by_task(task_id, dag_id)
            for te in self.read_task_executions_by_provider_and_task(p, task_id, dag_id)
        ]

    def read_first_access_times_by_task(self, task_id, dag_id):
        return {}


@pytest.mark.unit
@pytest.mark.parametrize("time_model", ["ridge", "setar", "linear", "lightgbm"])
def test_split_strategy_predicts_next_provider_and_future_time(time_model):
    db = FakeDB()
    # 30 runs: the last one (index 29) ran on PROVIDERS[29 % 3] = "C", so next is "A".
    predictor = TaskPredictor(db_manager=db, strategy="split", split_time_model=time_model)

    prediction = predictor.predict_next_task_execution("Task_1", "dag_1")

    assert prediction.predicted_provider_id == "A"
    now = datetime.now()
    assert now < prediction.predicted_next_access < now + timedelta(seconds=60)


@pytest.mark.unit
def test_split_strategy_cold_start_uses_fallback():
    # Only 2 executions: too little to train own models.
    db = FakeDB(n_runs=2)

    with_fallback = TaskPredictor(db_manager=db, strategy="split")
    prediction = with_fallback.predict_next_task_execution("Task_1", "dag_1")
    assert prediction is not None
    assert prediction.predicted_next_access > datetime.now()

    without_fallback = TaskPredictor(db_manager=db, strategy="split", use_fallback=False)
    assert without_fallback.predict_next_task_execution("Task_1", "dag_1") is None


@pytest.mark.unit
def test_split_strategy_empty_database_does_not_raise():
    class EmptyDB(FakeDB):
        def read_all_tasks(self):
            return []

    predictor = TaskPredictor(db_manager=EmptyDB(), strategy="split")
    assert predictor.predict_all_tasks() == {}


@pytest.mark.unit
def test_split_strategy_unknown_task_returns_none():
    predictor = TaskPredictor(db_manager=FakeDB(), strategy="split")

    class EmptyDB(FakeDB):
        def read_providers_used_by_task(self, task_id, dag_id):
            return []

    predictor.db_manager = EmptyDB()
    assert predictor.predict_next_task_execution("Task_1", "dag_1") is None


@pytest.mark.unit
def test_split_history_has_one_entry_per_execution_with_majority_provider():
    class MultiProviderDB(FakeDB):
        def read_access_counts_by_task(self, task_id, dag_id):
            # Execution "100" read from A (3 accesses) and B (5); "200" only from C.
            return [("100", "A", 3), ("100", "B", 5), ("200", "C", 2)]

    predictor = TaskPredictor(db_manager=MultiProviderDB(n_runs=2), strategy="split")

    assert predictor._read_task_history("Task_1", "dag_1") == [
        (datetime.fromtimestamp(100), "B"),
        (datetime.fromtimestamp(200), "C"),
    ]


@pytest.mark.unit
def test_split_history_uses_the_first_access_instead_of_the_task_start():
    class DelayedReadDB(FakeDB):
        def read_access_counts_by_task(self, task_id, dag_id):
            return [("100", "A", 1), ("200", "A", 1)]

        def read_first_access_times_by_task(self, task_id, dag_id):
            # Execution "100" first read 30 s after it started; "200" has no timepoint.
            return {"100": datetime.fromtimestamp(130).isoformat()}

    predictor = TaskPredictor(db_manager=DelayedReadDB(n_runs=2), strategy="split")

    assert predictor._read_task_history("Task_1", "dag_1") == [
        (datetime.fromtimestamp(130), "A"),
        (datetime.fromtimestamp(200), "A"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("use_fallback", [True, False])
def test_split_predictor_rolls_provider_forward_over_unseen_runs(use_fallback):
    # Round-robin A, B, C every 30 s; the last observed run was on "C".
    last = datetime(2026, 1, 1, 12, 0, 0)
    history = [(last - timedelta(seconds=30 * i), PROVIDERS[(29 - i) % 3]) for i in range(30)][::-1]
    models = train_split_models({"dag_1.Task_1": history})
    predictor = SplitPredictor(models, fallback=load_fallback() if use_fallback else None)

    # Next run after +65 s is at +90 s: +30 s ("A") and +60 s ("B") ran but are
    # not ingested yet, so it is on "C" again.
    after = predictor.predict("dag_1.Task_1", history, now=last + timedelta(seconds=65))
    assert after.predicted_provider_id == "C"

    # A run known to start at +62 s (start times jitter) is the one at +60 s: "B".
    at = predictor.predict("dag_1.Task_1", history, at=last + timedelta(seconds=62))
    assert at.predicted_provider_id == "B"
    assert at.predicted_next_access == last + timedelta(seconds=62)

    # Right after the last run, the next one is on "A".
    fresh = predictor.predict("dag_1.Task_1", history, now=last + timedelta(seconds=5))
    assert fresh.predicted_provider_id == "A"
