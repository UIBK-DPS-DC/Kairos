import pytest

from src.dag_prediction.space_models import (
    predict_markov,
    predict_most_frequent,
    predict_pattern_markov,
    train_markov,
    train_most_frequent,
    train_pattern_markov,
)


@pytest.mark.unit
def test_markov_predicts_round_robin_successor():
    model = train_markov({"t": list("ABCD" * 5)})

    assert predict_markov(model, "t", list("ABC")) == "D"
    assert predict_markov(model, "t", list("ABCD")) == "A"


@pytest.mark.unit
def test_markov_returns_home_after_failure():
    sequence = list("AAAAB" "AAAAC" "AAAAD" "AAAA")
    model = train_markov({"t": sequence})

    assert predict_markov(model, "t", list("AAA")) == "A"
    assert predict_markov(model, "t", list("AAC")) == "A"


@pytest.mark.unit
def test_markov_backs_off_to_shorter_context_and_frequency():
    model = train_markov({"t": list("ABAB")}, order=2)

    # Context ("C", "A") was never seen -> back off to ("A",) -> "B".
    assert predict_markov(model, "t", ["C", "A"]) == "B"
    # No context at all -> most frequent provider (tie broken alphabetically).
    assert predict_markov(model, "t", ["Z"]) == "A"


@pytest.mark.unit
def test_markov_without_backoff_needs_exact_context():
    model = train_markov({"t": list("ABAB")}, order=2)

    assert predict_markov(model, "t", ["B", "A"], backoff=False) == "B"
    assert predict_markov(model, "t", ["C", "A"], backoff=False) is None


@pytest.mark.unit
def test_pattern_markov_transfers_to_other_provider_names():
    model = train_pattern_markov([list("ABCD" * 5), list("AAAAAAAB" "AAAAAAAC" "AAAAAAA")])

    # Round-robin over 4 with unseen names: continue with the oldest one.
    assert predict_pattern_markov(model, list("WXYZ")) == "W"
    # Static home with one failure: return home.
    assert predict_pattern_markov(model, list("HHHX")) == "H"
    assert predict_pattern_markov(model, list("HHHH")) == "H"


@pytest.mark.unit
def test_pattern_markov_returns_none_for_new_provider_or_unknown_pattern():
    model = train_pattern_markov([list("ABC")], order=2)

    # Pattern "0,1" was always followed by a provider outside the context.
    assert predict_pattern_markov(model, list("XY")) is None
    assert predict_pattern_markov(model, []) is None


@pytest.mark.unit
def test_markov_unknown_series_returns_none():
    assert predict_markov(train_markov({"t": list("AB")}), "other", ["A"]) is None


@pytest.mark.unit
def test_most_frequent():
    model = train_most_frequent({"t": list("ABBC")})

    assert predict_most_frequent(model, "t") == "B"
    assert predict_most_frequent(model, "other") is None
