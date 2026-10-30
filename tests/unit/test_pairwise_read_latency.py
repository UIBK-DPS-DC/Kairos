from unittest.mock import MagicMock

import pytest

from src.data_placement.strategies.Kairos import Kairos
from src.network_monitoring import provider_latency
from src.network_monitoring.provider_latency import check_pairwise_read_latency

LOCAL_READ_MS = 50.0  # time between Kairos and every provider
REMOTE_FETCH_MS = {("a", "b"): 30.0, ("b", "a"): 40.0}  # (reader, holder) -> on-the-fly fetch


class Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now


class FakeProvider:
    """One provider of a shared space: reads of files stored elsewhere cost a remote fetch."""

    def __init__(self, provider_id, files, clock):
        self.provider_id = provider_id
        self.provider_name = provider_id
        self.files = files
        self.clock = clock
        self.reads = []

    def upload_file(self, filename, content):
        self.files[filename] = self.provider_id
        return filename

    def has_file(self, file_id):
        return file_id in self.files

    def read_file(self, file_id):
        holder = self.files[file_id]
        self.reads.append(file_id)
        fetch_ms = 0.0 if holder == self.provider_id else REMOTE_FETCH_MS[(self.provider_id, holder)]
        self.clock.now += (LOCAL_READ_MS + fetch_ms) / 1000
        return b""

    def delete_file(self, file_id):
        del self.files[file_id]


@pytest.fixture
def providers(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(provider_latency.time, "perf_counter", clock.perf_counter)
    files = {}
    return {name: FakeProvider(name, files, clock) for name in ("a", "b")}


def _zone():
    zone = MagicMock()
    zone.resolve_provider_id.side_effect = lambda key: key
    return zone


@pytest.mark.unit
def test_read_latency_is_the_remote_fetch_without_the_time_to_kairos(providers):
    latency = check_pairwise_read_latency(_zone(), providers, rounds=2)

    assert latency["a"]["a"] == 0.0
    assert latency["a"]["b"] == pytest.approx(30.0)
    assert latency["b"]["a"] == pytest.approx(40.0)
    assert providers["a"].files == {}  # every probe is deleted again


@pytest.mark.unit
def test_every_probe_is_read_once_so_no_provider_caches_it(providers):
    check_pairwise_read_latency(_zone(), providers, rounds=2)

    reads = providers["a"].reads + providers["b"].reads
    assert len(reads) == len(set(reads)) == 8  # 2 providers x 2 rounds x (local + remote)


@pytest.mark.unit
def test_pair_stays_unmeasured_when_the_probe_never_becomes_visible(providers, monkeypatch):
    monkeypatch.setattr(provider_latency, "_wait_until_visible", lambda *a: False)

    latency = check_pairwise_read_latency(_zone(), providers, rounds=1)

    assert latency["a"]["b"] is None
    assert latency["b"]["a"] is None


def _kairos(latency_map, matrix):
    kairos = Kairos.__new__(Kairos)  # skip __init__: it measures provider latencies
    kairos.latency_map = latency_map
    kairos.provider_latency_matrix = matrix
    return kairos


@pytest.mark.unit
def test_kairos_uses_the_closest_holder_of_the_predicted_provider():
    kairos = _kairos({}, {"p": {"h1": 80.0, "h2": 25.0, "h3": None}})

    assert kairos._remote_read_latency("p", ["h1", "h2", "h3"]) == 25.0


@pytest.mark.unit
def test_kairos_estimates_unmeasured_pairs_via_itself():
    kairos = _kairos({"p": 150.0, "h1": 120.0, "h2": 400.0}, {"p": {"h1": None}})

    assert kairos._remote_read_latency("p", ["h1", "h2"]) == 270.0
    assert _kairos({"h1": 120.0}, None)._remote_read_latency("p", ["h1"]) is None
