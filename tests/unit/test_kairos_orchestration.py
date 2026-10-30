from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.dag_prediction.task_predictor import TaskPrediction
from src.data_placement.strategies.Kairos import Kairos
import src.data_placement.strategies.Kairos as kairos_module
from src.network_monitoring.network_bandwith import PROBE_SIZE_BYTES

MB = 1_000_000


def _kairos(bandwidth=None, prices=None):
    kairos = Kairos.__new__(Kairos)  # skip __init__: it measures provider latencies
    kairos.onedata_manager = MagicMock()
    kairos.latency_map = {}
    kairos.provider_latency_matrix = {"p": {"h": 50.0}, "q": {"h": 50.0}}
    kairos.provider_bandwidth_matrix = bandwidth
    kairos.prices = prices or {"p": 1.0, "q": 1.0}
    kairos.task_predictor = None
    return kairos


def _candidate(file_id, vcr, cost, start_in_s=60.0, task_order=0, size=MB, to="p"):
    return Kairos.transition_candidate(
        file_id, to, vcr, value=1.0, cost=cost,
        predicted_time=datetime.now() + timedelta(seconds=start_in_s),
        holders=["h"], size=size, task_order=task_order,
    )


def _replicated(kairos):
    return [c.args[0] for c in kairos.onedata_manager.replicate_to_provider.call_args_list]


@pytest.mark.unit
def test_transitions_are_executed_by_vcr_then_topological_order_within_budget():
    kairos = _kairos()
    candidates = [
        _candidate("low", vcr=1.0, cost=1.0),
        _candidate("tie_late", vcr=5.0, cost=1.0, task_order=2),
        _candidate("tie_early", vcr=5.0, cost=1.0, task_order=1),
        _candidate("too_expensive", vcr=4.0, cost=10.0),
    ]

    remaining = kairos._execute_transitions(candidates, budget=2.5)

    assert _replicated(kairos) == ["tie_early", "tie_late"]
    assert remaining == pytest.approx(0.5)


@pytest.mark.unit
def test_execution_stops_once_the_budget_is_used_up():
    kairos = _kairos()

    kairos._execute_transitions([_candidate("a", 2.0, cost=1.0), _candidate("b", 1.0, cost=0.0)], budget=1.0)

    assert _replicated(kairos) == ["a"]


@pytest.mark.unit
def test_replication_that_finishes_after_the_predicted_start_is_skipped(monkeypatch):
    monkeypatch.setattr(kairos_module, "REPLICATION_BANDWIDTH_BYTES_PER_SECOND", 1 * MB)
    kairos = _kairos(bandwidth={"p": {"h": PROBE_SIZE_BYTES / 5.0}})  # 5 s overhead
    candidates = [
        _candidate("slow", vcr=2.0, cost=0.0, size=100 * MB, start_in_s=30),  # needs 105 s
        _candidate("fast", vcr=1.0, cost=0.0, size=10 * MB, start_in_s=30),  # needs 15 s
    ]

    kairos._execute_transitions(candidates, budget=1.0)

    assert _replicated(kairos) == ["fast"]


@pytest.mark.unit
def test_without_measured_probe_the_default_overhead_is_required(monkeypatch):
    monkeypatch.setattr(kairos_module, "DEFAULT_REPLICATION_OVERHEAD_SECONDS", 10.0)
    kairos = _kairos(bandwidth=None)

    kairos._execute_transitions(
        [_candidate("in_time", 1.0, 0.0, start_in_s=20), _candidate("too_soon", 2.0, 0.0, start_in_s=5)],
        budget=1.0,
    )

    assert _replicated(kairos) == ["in_time"]


@pytest.mark.unit
def test_replication_time_is_the_lowest_probe_overhead_plus_the_transfer(monkeypatch):
    monkeypatch.setattr(kairos_module, "REPLICATION_BANDWIDTH_BYTES_PER_SECOND", 4 * MB)
    monkeypatch.setattr(kairos_module, "DEFAULT_REPLICATION_OVERHEAD_SECONDS", 10.0)
    kairos = _kairos(bandwidth={"p": {
        "h1": PROBE_SIZE_BYTES / 8.0, "h2": PROBE_SIZE_BYTES / 4.0, "h3": float("inf"),
    }})

    assert kairos._replication_time("p", ["h1", "h2", "h3"], 8 * MB) == pytest.approx(4.0 + 2.0)
    assert kairos._replication_time("q", ["h1"], 8 * MB) == pytest.approx(10.0 + 2.0)


@pytest.mark.unit
def test_interrupt_replicates_dynamic_files_by_vcr_within_the_cycle_budget():
    kairos = _kairos(prices={"p": 1.0, "q": 4.0})
    kairos._get_file_id_with_retry = lambda path, written_after=None: f"id_{path}"
    distribution = MagicMock(fully_replicated_providers=["h"], maximum_physical_size=10**9)
    kairos._get_global_replica_distribution = MagicMock(return_value=distribution)
    soon = datetime.now() + timedelta(seconds=600)  # leaves time to replicate 1 GB
    predictions = {
        "c1_dag": TaskPrediction(predicted_next_access=soon, predicted_provider_id="p"),
        "c2_dag": TaskPrediction(predicted_next_access=soon, predicted_provider_id="q"),
    }
    file_paths = {"t": ([], {"out1": ["c1"], "out2": ["c2"]}, {})}
    budget = {"remaining": 1.5}

    # One hour horizon and 1 GB files: cost 1 $ to p (higher VCR) and 4 $ to q.
    kairos._on_interrupt({"task_id": "t", "dag_id": "dag"}, predictions, file_paths, 3600, budget)

    assert _replicated(kairos) == ["id_out1"]
    assert budget["remaining"] == pytest.approx(0.5)
