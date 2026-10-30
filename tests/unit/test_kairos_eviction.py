import json
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

import src.data_placement.strategies.Kairos as kairos_module
from src.data_placement.strategies.Kairos import Kairos
from src.network_monitoring.network_config import NetworkConfig, load_network_config
from src.rest_api.onedata.onedata_helpers import ReplicaDistribution

MB = 1_000_000


@pytest.fixture(autouse=True)
def _no_protected_replicas():
    kairos_module._protected_replicas.clear()
    yield
    kairos_module._protected_replicas.clear()


def _kairos(popularity, replicas):
    """``popularity``: provider -> {file_id: score}; ``replicas``: file_id -> holding providers."""
    kairos = Kairos.__new__(Kairos)  # skip __init__: it measures provider latencies
    kairos.onedata_manager = MagicMock()
    kairos.zone_manager = MagicMock()
    kairos.network_config = None
    kairos.onedata_manager.get_file_popularity_with_values.return_value = {
        pid: [{"key": score, "value": fid} for fid, score in files.items()]
        for pid, files in popularity.items()
    }
    kairos._get_global_replica_distribution = lambda fid: ReplicaDistribution(
        fid, {pid: {"blocksPercentage": 100, "virtualSize": MB} for pid in replicas.get(fid, [])}
    )
    return kairos


def _evicted(kairos):
    return [c.args for c in kairos.onedata_manager.evict_from_provider_id.call_args_list]


@pytest.mark.unit
def test_the_least_popular_replica_is_evicted():
    kairos = _kairos({"a": {"f": 1.0}, "b": {"f": 5.0}}, {"f": ["a", "b"]})

    kairos.budget_expansion(1, {}, 60)

    assert _evicted(kairos) == [("f", "a")]


@pytest.mark.unit
def test_a_replica_kairos_just_created_is_not_evicted():
    kairos = _kairos({"a": {"f": 5.0}, "b": {"f": 9.0}}, {"f": ["a", "b", "new"]})
    kairos_module._protect_replica("f", "new", datetime.now() + timedelta(seconds=30))

    kairos.budget_expansion(1, {}, 60)

    # Without protection "new" (popularity 0) would be evicted.
    assert _evicted(kairos) == [("f", "a")]


@pytest.mark.unit
def test_executing_a_transition_protects_its_replica():
    kairos = _kairos({"a": {"f": 5.0}}, {"f": ["a", "p"]})
    kairos.provider_bandwidth_matrix = None
    candidate = Kairos.transition_candidate(
        "f", "p", 1.0, value=1.0, cost=0.0,
        predicted_time=datetime.now() + timedelta(seconds=60), holders=["a"], size=MB,
    )

    kairos._execute_transitions([candidate], budget=1.0)
    kairos.budget_expansion(1, {}, 60)

    assert kairos.onedata_manager.replicate_to_provider.call_args.args == ("f", "p")
    assert _evicted(kairos) == [("f", "a")]


@pytest.mark.unit
def test_the_protection_ends_after_the_predicted_start(monkeypatch):
    monkeypatch.setattr(kairos_module, "REPLICA_PROTECTION_SECONDS", 60.0)
    now = time.time()
    kairos_module._protect_replica("f", "new", datetime.fromtimestamp(now + 30))

    monkeypatch.setattr(kairos_module.time, "time", lambda: now + 80)
    assert kairos_module._is_protected("f", "new")
    monkeypatch.setattr(kairos_module.time, "time", lambda: now + 100)
    assert not kairos_module._is_protected("f", "new")


@pytest.mark.unit
def test_files_the_dag_reads_are_not_evicted():
    kairos = _kairos(
        {"a": {"input": 0.0, "other": 1.0}, "b": {"input": 0.0, "other": 2.0}},
        {"input": ["a", "b"], "other": ["a", "b"]},
    )

    kairos.budget_expansion(1, {}, 60, protected_file_ids={"input"})

    assert _evicted(kairos) == [("other", "a")]


@pytest.mark.unit
def test_static_file_ids_resolve_the_read_paths_of_all_tasks():
    kairos = _kairos({}, {})
    kairos.onedata_manager.get_file_id_from_path.side_effect = {"/in1": "id1", "/in2": None}.get
    file_paths = {"t1": (["/in1"], {"/out": ["t2"]}, {}), "t2": (["/in2"], {}, {})}

    assert kairos._static_file_ids(file_paths) == {"id1"}


@pytest.mark.unit
def test_network_config_sets_the_replication_time():
    kairos = _kairos({}, {})
    kairos.provider_bandwidth_matrix = None
    kairos.network_config = NetworkConfig(
        replication_overhead_seconds=4.0, replication_bandwidth_bytes_per_second=10 * MB
    )

    assert kairos._replication_time("p", ["h"], 100 * MB) == pytest.approx(14.0)


@pytest.mark.unit
def test_with_a_network_config_the_upload_rtt_is_not_measured(monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("upload RTT must not be measured")

    monkeypatch.setattr(kairos_module, "check_all_providers_latency_with_ids", _fail)
    monkeypatch.setattr(Kairos, "read_price_config", lambda self: {})
    config = NetworkConfig(upload_rtt_ms={"p": 400.0})

    kairos = Kairos({}, MagicMock(), MagicMock(), MagicMock(), network_config=config)

    assert kairos.latency_map == {"p": 400.0}


@pytest.mark.unit
def test_load_network_config(tmp_path):
    path = tmp_path / "net.json"
    path.write_text(json.dumps({
        "_comment": "ignored",
        "pairwise_read_latency_ms": {"r": {"h": 12}},
        "upload_rtt_ms": {"r": 400},
        "replication_overhead_seconds": 10,
    }))

    config = load_network_config(str(path))

    assert config.pairwise_read_latency_ms == {"r": {"h": 12.0}}
    assert config.upload_rtt_ms == {"r": 400.0}
    assert config.replication_overhead_seconds == 10.0
    assert config.replication_bandwidth_bytes_per_second is None


@pytest.mark.unit
def test_the_shipped_network_config_covers_all_provider_pairs():
    config = load_network_config("src/network_monitoring/network_conf.json")

    providers = set(config.upload_rtt_ms)
    assert len(providers) == 4
    for reader in providers:
        assert set(config.pairwise_read_latency_ms[reader]) == providers - {reader}
