from types import SimpleNamespace

import pytest

from src.data_placement.strategies.Popularity_Baseline import Popularity_Baseline


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