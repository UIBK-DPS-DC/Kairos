import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Tuple, List


class SETARTree:
    """Simple recursive SETAR tree that fits linear models at leaves.

    Splits are chosen by searching candidate lags (features) and thresholds
    that minimize pooled RSS. Designed to be trained on pooled (cross-trained)
    lagged examples coming from multiple pipelines.
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        min_error_reduction: float = 0.0,
        n_thresholds: int = 15,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_error_reduction = min_error_reduction
        self.n_thresholds = n_thresholds
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        n_samples, n_features = X.shape

        # Stopping conditions: max depth or too few samples
        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
            model = LinearRegression()
            if n_samples > 0:
                model.fit(X, y)
            return {"is_leaf": True, "model": model, "n_samples": n_samples}

        best_cost = np.inf
        best_split = None

        parent_rss = np.sum((y - np.mean(y)) ** 2)

        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            if len(values) > self.n_thresholds:
                # sample candidate thresholds using percentiles
                qs = np.linspace(0, 100, self.n_thresholds + 2)[1:-1]
                thresholds = np.unique(np.percentile(X[:, feature_idx], qs))
            else:
                thresholds = values

            for th in thresholds:
                left_mask = X[:, feature_idx] <= th
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                cost = np.sum((y_left - np.mean(y_left)) ** 2) + np.sum(
                    (y_right - np.mean(y_right)) ** 2
                )

                if cost < best_cost:
                    best_cost = cost
                    best_split = {
                        "feature_idx": feature_idx,
                        "threshold": th,
                        "left_idx": np.where(left_mask)[0],
                        "right_idx": np.where(right_mask)[0],
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                        "cost": cost,
                    }

        # If no valid split found or no sufficient improvement -> leaf
        if best_split is None:
            model = LinearRegression()
            if n_samples > 0:
                model.fit(X, y)
            return {"is_leaf": True, "model": model, "n_samples": n_samples}

        # Check improvement
        improvement = (parent_rss - best_cost) / parent_rss if parent_rss > 0 else 0.0
        if improvement < self.min_error_reduction:
            model = LinearRegression()
            model.fit(X, y)
            return {"is_leaf": True, "model": model, "n_samples": n_samples}

        # Otherwise recurse
        left_X, left_y = X[best_split["left_idx"]], y[best_split["left_idx"]]
        right_X, right_y = X[best_split["right_idx"]], y[best_split["right_idx"]]

        return {
            "is_leaf": False,
            "feature_idx": best_split["feature_idx"],
            "threshold": best_split["threshold"],
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1),
        }

    def predict_row(self, node: dict, row: np.ndarray) -> float:
        if node["is_leaf"]:
            model = node["model"]
            # If the model was not fitted (edge case), return mean 0
            return (
                float(model.predict(row.reshape(1, -1))[0])
                if hasattr(model, "coef_")
                else 0.0
            )

        if row[node["feature_idx"]] <= node["threshold"]:
            return self.predict_row(node["left"], row)
        else:
            return self.predict_row(node["right"], row)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_row(self.tree, row) for row in X])


def build_lagged_features(
    series: np.ndarray, lag: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 1D series into lagged matrix X and target y."""
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i : i + lag])
        y.append(series[i + lag])
    return np.array(X), np.array(y)


def train_setar_tree(
    pipeline_history: Dict[str, List[Any]],
    lag: int = 2,
    max_depth: int = 5,
    min_samples_leaf: int = 3,
    min_error_reduction: float = 0.0,
    n_thresholds: int = 15,
) -> Dict[str, Any]:
    """Train a single SETAR tree on pooled lagged intervals across pipelines.

    Args:
        pipeline_history: dict of pipeline_name -> list/array of datetime-like objects (strings or pd.Timestamp)
    Returns:
        model dict containing the trained tree and metadata required for prediction.
    """
    pooled_X = []
    pooled_y = []
    meta = {}

    for name, times in pipeline_history.items():
        dt = pd.to_datetime(pd.Series(times)).sort_values().reset_index(drop=True)
        intervals = dt.diff().dt.total_seconds().dropna().values
        meta[name] = {
            "last_start": dt.iloc[-1],
            "last_intervals": intervals[-lag:] if len(intervals) >= lag else None,
            "n_intervals": len(intervals),
        }

        if len(intervals) <= lag:
            continue

        X, y = build_lagged_features(intervals, lag)
        pooled_X.append(X)
        pooled_y.append(y)

    if len(pooled_X) == 0:
        raise ValueError(
            "No pipeline has enough intervals to train (need > lag intervals)."
        )

    X_all = np.vstack(pooled_X)
    y_all = np.concatenate(pooled_y)

    tree = SETARTree(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_error_reduction=min_error_reduction,
        n_thresholds=n_thresholds,
    )
    tree.tree = tree.fit(X_all, y_all)

    return {"tree": tree, "lag": lag, "meta": meta}


def predict_next_from_model(
    model: Dict[str, Any], pipelines: List[str] = None
) -> Dict[str, pd.Timestamp]:
    """Predict the next start timestamp for each pipeline using the trained model.

    If a pipeline did not have enough history at training time, it will be skipped.
    """
    tree: SETARTree = model["tree"]
    lag: int = model["lag"]
    meta = model["meta"]

    results = {}
    for name, info in meta.items():
        if pipelines and name not in pipelines:
            continue

        last_intervals = info.get("last_intervals")
        if last_intervals is None or len(last_intervals) < lag:
            # cannot predict for this pipeline
            continue

        x = np.array(last_intervals).reshape(1, -1)
        pred_seconds = tree.predict(x)[0]
        next_time = info["last_start"] + pd.Timedelta(seconds=float(pred_seconds))
        results[name] = next_time

    return results


def predict_next_for_pipeline_with_timestamps(
    model: Dict[str, Any],
    prediction_object_name: str,
    new_times: List[Any],
    now: Any = None,
) -> pd.Timestamp:
    """Predict next start for a single pipeline using new timestamps without retraining.

    Args:
        model: the dict returned by `train_setar_tree`
        prediction_object_name: arbitrary name for the pipeline (used only in result)
        new_times: list/array of datetime-like objects (strings or pd.Timestamp)
        now: reference time the prediction is rolled past (default: wall clock)

    Returns:
        predicted next pd.Timestamp
    """
    lag: int = model["lag"]
    tree: SETARTree = model["tree"]

    dt = pd.to_datetime(pd.Series(new_times)).sort_values().reset_index(drop=True)
    intervals = dt.diff().dt.total_seconds().dropna().values

    if len(intervals) < lag:
        raise ValueError(
            f"Not enough intervals in provided timestamps: need at least {lag+1} timestamps."
        )

    last_lags = intervals[-lag:]
    x = np.array(last_lags).reshape(1, -1)
    pred_seconds = tree.predict(x)[0]

    # Guard: if the model predicts a non-positive interval
    if pred_seconds <= 0:
        mean_interval = float(np.mean(intervals))
        pred_seconds = mean_interval if mean_interval > 0 else 60.0

    next_time = dt.iloc[-1] + pd.Timedelta(seconds=float(pred_seconds))

    # Forward-roll to future
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    while next_time <= now:
        next_time += pd.Timedelta(seconds=float(pred_seconds))

    return next_time


if __name__ == "__main__":
    sample = {
        "A": [
            "2023-10-01 02:00:00",
            "2023-10-02 02:05:00",
            "2023-10-03 01:50:00",
            "2023-10-04 02:10:00",
            "2023-10-05 02:00:00",
            "2023-10-06 01:55:00",
            "2023-10-07 02:05:00",
            "2023-10-08 02:02:00",
            "2023-10-09 01:58:00",
        ],
        "B": [
            "2023-10-01 08:00:00",
            "2023-10-01 16:00:00",
            "2023-10-02 08:15:00",
            "2023-10-02 16:10:00",
            "2023-10-03 07:55:00",
            "2023-10-03 15:50:00",
            "2023-10-04 08:05:00",
            "2023-10-04 16:05:00",
            "2023-10-05 08:00:00",
        ],
    }

    m = train_setar_tree(sample, lag=5, max_depth=4, min_samples_leaf=3)

    print("Training complete.")

    preds = predict_next_from_model(m)
    for k, v in preds.items():
        print(f"{k} -> {v}")

    # Demonstrate predicting for a single pipeline with new timestamps (no retrain)
    new_times_for_A = sample["A"] + ["2023-10-10 02:03:00"]
    next_for_A = predict_next_for_pipeline_with_timestamps(m, "A", new_times_for_A)
    print(f"A (from new timestamps) -> {next_for_A}")
