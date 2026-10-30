"""Global interval regressors (LightGBM, Ridge) pooled over all series.

Like the SETAR tree, one model is trained on the lagged inter-arrival intervals
of every pipeline (e.g. "{task}_{provider}" or one series per task), but the
pipeline key is an additional categorical feature, so the model can learn
per-series behaviour.
"""

from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dag_prediction.setar_tree import build_lagged_features

CATEGORICAL = ["series"]


def _sorted_times_and_intervals(times: List[Any]) -> tuple[pd.Series, np.ndarray]:
    dt = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
    return dt, dt.diff().dt.total_seconds().dropna().to_numpy()


def _features(rows: List[tuple], lag: int, categories: Dict[str, List[str]]) -> pd.DataFrame:
    X = pd.DataFrame(rows, columns=[f"lag_{i}" for i in range(lag)] + CATEGORICAL)
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def train_global_model(
    pipeline_history: Dict[str, List[Any]],
    kind: Literal["lightgbm", "ridge"],
    lag: int = 2,
) -> Dict[str, Any]:
    """Train one regressor on the pooled lagged intervals of all pipelines.

    Args:
        pipeline_history: dict of series key -> list of datetime-like objects
        kind: "lightgbm" (gradient-boosted trees) or "ridge" (RidgeCV, one-hot categories)
    """
    rows, targets = [], []
    for key, times in pipeline_history.items():
        _, intervals = _sorted_times_and_intervals(times)
        X, y = build_lagged_features(intervals, lag)
        rows += [(*x, key) for x in X]
        targets += list(y)
    if not rows:
        raise ValueError("No pipeline has enough intervals to train (need > lag intervals).")

    categories = {col: sorted({r[lag + i] for r in rows}) for i, col in enumerate(CATEGORICAL)}
    X = _features(rows, lag, categories)

    if kind == "lightgbm":
        # Imported lazily: LightGBM needs libgomp, which the runtime image lacks.
        from lightgbm import LGBMRegressor

        # Small datasets: allow small leaves, keep the trees shallow.
        estimator = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5, verbose=-1
        )
    else:
        estimator = make_pipeline(
            ColumnTransformer(
                [("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                remainder=StandardScaler(),
            ),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
    # RidgeCV's leave-one-out divides by zero on the tiny histories of a cold start.
    with np.errstate(divide="ignore", invalid="ignore"):
        estimator.fit(X, np.asarray(targets))
    return {"kind": kind, "estimator": estimator, "lag": lag, "categories": categories}


def roll_past(
    last: pd.Timestamp, pred_seconds: float, intervals: np.ndarray, now: pd.Timestamp
) -> pd.Timestamp:
    """``last + n * pred_seconds`` for the smallest n >= 1 that lies after ``now``.

    A non-positive prediction is replaced by the mean observed interval.
    """
    if pred_seconds <= 0:
        mean_interval = float(np.mean(intervals))
        pred_seconds = mean_interval if mean_interval > 0 else 60.0
    step = pd.Timedelta(seconds=float(pred_seconds))
    n_steps = max(1, int((now - last) // step) + 1)
    return last + n_steps * step


def predict_next_for_pipelines(
    model: Dict[str, Any], histories: Dict[str, List[Any]], now: Any = None
) -> Dict[str, pd.Timestamp]:
    """Predict the next start of every pipeline in ``histories`` in one batch.

    Pipelines with fewer than ``lag + 1`` timestamps are skipped. Like the SETAR
    predictor, the forecast is rolled forward until it lies after ``now``
    (default: wall clock).
    """
    lag = model["lag"]
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)

    keys, rows, series = [], [], []
    for key, times in histories.items():
        dt, intervals = _sorted_times_and_intervals(times)
        if len(intervals) < lag:
            continue
        keys.append(key)
        rows.append((*intervals[-lag:], key))
        series.append((dt.iloc[-1], intervals))
    if not keys:
        return {}

    predicted = model["estimator"].predict(_features(rows, lag, model["categories"]))

    results = {}
    for key, pred_seconds, (last, intervals) in zip(keys, predicted, series):
        results[key] = roll_past(last, pred_seconds, intervals, now)
    return results
