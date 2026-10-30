from datetime import datetime, timedelta

import pytest

from src.dag_prediction.fallback_model import (
    load_fallback,
    predict_next_provider,
    predict_next_time,
    save_fallback,
    train_fallback,
)

START = datetime(2026, 1, 1, 12, 0, 0)


def _times(n: int, period_s: float) -> list:
    return [START + timedelta(seconds=i * period_s) for i in range(n)]


@pytest.fixture
def model():
    return train_fallback(
        time_series=[_times(30, 30), _times(30, 60)],
        provider_sequences=[list("ABCD" * 5), list("AAAAAAAB" "AAAAAAAC" "AAAAAAA")],
    )


@pytest.mark.unit
def test_time_model_is_scale_free(model):
    # Trained on 30 s / 60 s periods, applied to a 300 s period.
    times = _times(5, 300)
    predicted = predict_next_time(model, times, now=times[-1])

    assert abs((predicted - times[-1]).total_seconds() - 300) < 1


@pytest.mark.unit
def test_time_prediction_with_little_history(model):
    assert predict_next_time(model, _times(1, 30), now=START) is None
    two = _times(2, 45)
    assert predict_next_time(model, two, now=two[-1]) == two[-1] + timedelta(seconds=45)


@pytest.mark.unit
def test_time_prediction_is_rolled_past_now(model):
    times = _times(5, 30)
    now = times[-1] + timedelta(seconds=100)

    assert predict_next_time(model, times, now=now) > now


@pytest.mark.unit
def test_provider_prediction(model):
    assert predict_next_provider(model, list("WXYZ")) == "W"
    assert predict_next_provider(model, list("HHHX")) == "H"
    assert predict_next_provider(model, []) is None


@pytest.mark.unit
def test_json_round_trip(model, tmp_path):
    path = tmp_path / "fallback.json"
    save_fallback(model, path)
    loaded = load_fallback(path)

    assert loaded["time"] == model["time"]
    assert predict_next_provider(loaded, list("WXYZ")) == "W"
    assert load_fallback(tmp_path / "missing.json") is None


@pytest.mark.unit
def test_shipped_fallback_model_loads_and_predicts():
    shipped = load_fallback()
    assert shipped is not None, "run python -m evaluation.prediction.build_fallback"

    times = _times(5, 30)
    assert predict_next_time(shipped, times, now=times[-1]) > times[-1]
    assert predict_next_provider(shipped, list("ABCDABC")) == "D"
