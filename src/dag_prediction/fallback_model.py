"""Pretrained, task-agnostic fallback predictor for cold starts.

Used by the split TaskPredictor while a task has too little history of its own
(see TaskPredictor's ``min_history``). Both parts transfer to unseen tasks:

- time: Ridge on the last ``lag`` intervals normalised by their mean, so it
  learns the shape of a task's rhythm, not its period.
- space: a pattern Markov model over providers relabelled by first appearance
  (see space_models.train_pattern_markov).

The model is plain JSON (coefficients and counts), so loading it does not
depend on the scikit-learn version. Rebuild it from a mixture of the synthetic
workloads with:
    python -m evaluation.prediction.build_fallback
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.dag_prediction.global_models import roll_past
from src.dag_prediction.setar_tree import build_lagged_features
from src.dag_prediction.space_models import predict_pattern_markov, train_pattern_markov

logger = logging.getLogger(__name__)

FALLBACK_MODEL_PATH = Path(__file__).with_name("fallback_model.json")


def _intervals(times: List[Any]) -> tuple[pd.Series, np.ndarray]:
    dt = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
    return dt, dt.diff().dt.total_seconds().dropna().to_numpy()


def train_fallback(
    time_series: List[List[Any]],
    provider_sequences: List[List[str]],
    lag: int = 2,
    order: int = 4,
    alpha: float = 1.0,
) -> Dict[str, Any]:
    """Train the fallback model on many task histories (timestamps and providers)."""
    X_all, y_all = [], []
    for times in time_series:
        X, y = build_lagged_features(_intervals(times)[1], lag)
        if len(X) == 0:
            continue
        scale = X.mean(axis=1)
        ok = scale > 0
        X_all.append(X[ok] / scale[ok, None])
        y_all.append(y[ok] / scale[ok])
    if not X_all:
        raise ValueError("No series has enough intervals to train the fallback time model.")
    ridge = Ridge(alpha=alpha).fit(np.vstack(X_all), np.concatenate(y_all))

    return {
        "time": {"lag": lag, "coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)},
        "space": train_pattern_markov(provider_sequences, order),
    }


def save_fallback(model: Dict[str, Any], path: Path = FALLBACK_MODEL_PATH) -> None:
    path.write_text(json.dumps(model, indent=1, sort_keys=True), encoding="utf-8")


def load_fallback(path: Path = FALLBACK_MODEL_PATH) -> Optional[Dict[str, Any]]:
    """Load the fallback model; None (with a warning) if the file is missing."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Fallback prediction model not found at %s", path)
        return None


def predict_next_time(
    model: Dict[str, Any], times: List[Any], now: Any = None
) -> Optional[pd.Timestamp]:
    """Next start after ``now`` (default: wall clock); None with fewer than 2 timestamps.

    With fewer than ``lag`` intervals the last interval is repeated.
    """
    dt, intervals = _intervals(times)
    if len(intervals) == 0:
        return None
    lag = model["time"]["lag"]
    if len(intervals) < lag:
        pred_seconds = float(intervals[-1])
    else:
        lags = intervals[-lag:]
        scale = float(np.mean(lags))
        ratio = float(np.dot(model["time"]["coef"], lags / scale)) + model["time"]["intercept"] if scale > 0 else 0.0
        pred_seconds = scale * ratio
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    return roll_past(dt.iloc[-1], pred_seconds, intervals, now)


def predict_next_provider(model: Dict[str, Any], providers: List[str]) -> Optional[str]:
    """Next provider from the provider sequence; None for an empty sequence.

    If the pattern predicts a provider outside the recent context, the most
    used provider outside that context is taken (else the most used overall).
    """
    if not providers:
        return None
    predicted = predict_pattern_markov(model["space"], providers)
    if predicted is not None:
        return predicted
    counts = Counter(providers)
    context = set(providers[-model["space"]["order"]:])
    outside = Counter({p: c for p, c in counts.items() if p not in context})
    return (outside or counts).most_common(1)[0][0]
