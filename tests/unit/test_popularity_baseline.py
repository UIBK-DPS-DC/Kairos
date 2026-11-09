from types import SimpleNamespace

import pytest

from src.data_placement.strategies.Popularity_Baseline import Popularity_Baseline


def _baseline(provider_names):
    zone_manager = SimpleNamespace(list_providers=lambda: [])
    return Popularity_Baseline(
        provider_managers={name: SimpleNamespace() for name in provider_names},
        zone_manager=zone_manager,
        onedata_manager=SimpleNamespace(),
    )


@pytest.mark.unit
def test_baseline_only_places_on_the_three_matrix_providers():
    strategy = _baseline(["cloud-pl", "Cloud", "uibk", "Cloud-SK"])

    assert strategy.provider_ids == ["Cloud", "uibk", "Cloud-SK"]
    assert [len(row) for row in strategy.L] == [3, 3, 3]
    assert all(strategy.L[i][i] == 0.0 for i in range(3))
    assert set(strategy.calculate_coordinates_providers()) == set(strategy.provider_ids)


@pytest.mark.unit
def test_baseline_requires_a_manager_for_every_matrix_provider():
    with pytest.raises(ValueError, match="Cloud-SK"):
        _baseline(["Cloud", "uibk"])


@pytest.mark.unit
def test_reconcile_file_placement_schedules_replication_and_eviction():
    class Distribution:
        def __init__(self, providers):
            self.provider_ids = providers
            self.fully_replicated_providers = providers

    class ProviderManager:
        def __init__(self, providers):
            self.providers = providers

        def get_replica_distribution(self, file_id):
            assert file_id == "file-1"
            return Distribution(self.providers)

        def replicate(self, file_id, provider_id, zone_manager):
            transfers.append(("replicate", file_id, provider_id, zone_manager))
            return "replication"

        def evict(self, file_id, provider_id):
            transfers.append(("evict", file_id, provider_id))
            return "eviction"

    transfers = []

    strategy = Popularity_Baseline.__new__(Popularity_Baseline)
    strategy.provider_ids = ["provider-a", "provider-b", "provider-c"]
    strategy.provider_managers = {
        "provider-a": ProviderManager({"provider-a", "provider-c"}),
        "provider-b": ProviderManager(set()),
    }
    strategy.onedata_manager = strategy.provider_managers["provider-a"]
    strategy.zone_manager = SimpleNamespace()

    result = strategy.reconcile_file_placement("file-1", {"provider-a", "provider-b"})

    assert result == {
        "replications": ["replication"],
        "evictions": ["eviction"],
    }
    assert transfers == [
        ("replicate", "file-1", "provider-b", strategy.zone_manager),
        ("evict", "file-1", "provider-c"),
    ]