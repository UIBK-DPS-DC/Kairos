"""Smoke tests for Onedata Prometheus monitor."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.dag_monitoring.monitor_db import OnedataPrometheusMonitor


class MockDatabaseManager:
    def __init__(self):
        self.calls = []
        self.accesses = []

    def insert_dag(self, dag):
        self.calls.append(("insert_dag", dag.id))

    def insert_task(self, task):
        self.calls.append(("insert_task", task.id, task.dag_id))

    def insert_dag_execution(self, execution):
        self.calls.append(("insert_dag_execution", execution.id, execution.dag_id))

    def insert_task_execution(self, execution):
        self.calls.append(("insert_task_execution", execution.id, execution.task_id))

    def insert_file(self, file):
        self.calls.append(("insert_file", file.id))

    def insert_provider(self, provider):
        self.calls.append(("insert_provider", provider.id))

    def insert_space(self, space):
        self.calls.append(("insert_space", space.id))

    def read_access(self, access_id):
        return next((a for a in self.accesses if a.id == access_id), None)

    def insert_accesses(self, accesses):
        accesses = list(accesses)
        self.accesses.extend(accesses)
        self.calls.append(("insert_accesses", len(accesses)))
        return len(accesses)


def mock_query_response(*_args, **_kwargs):
    now = datetime.now().timestamp()
    return {
        "resultType": "vector",
        "result": [
            {
                "metric": {
                    "operation": "read",
                    "space_id": "space_1",
                    "file_id": "file_123",
                    "provider_id": "provider_1",
                    "session_id": "dag_1:dag_exec_1:task_1:task_exec_1",
                },
                "value": [now, "1"],
            }
        ],
    }


@pytest.fixture
def monitor(monkeypatch):
    """Create a monitor with a mocked Prometheus response and DB manager."""
    database_manager = MockDatabaseManager()
    data_manager = MagicMock()
    data_manager.get_file_path.return_value = "/space/file_123"
    monitor = OnedataPrometheusMonitor(
        prometheus_host="http://localhost:9090",
        data_manager=data_manager,
        database_manager=database_manager,
        query_interval=60,
        lookback_minutes=5,
        include_duration=False,
        test_mode=True,
    )

    monkeypatch.setattr(monitor, "_query_prometheus", mock_query_response)
    return monitor


@pytest.mark.smoke
def test_get_onedata_operations_smoke(monitor):
    operations = monitor.get_onedata_read_operations()

    assert len(operations) == 1
    operation = operations[0]
    assert operation["operation"] == "read"
    assert operation["space_id"] == "space_1"
    assert operation["file_id"] == "file_123"
    assert operation["provider_id"] == "provider_1"
    assert operation["dag_id"] == "dag_1"
    assert operation["task_id"] == "task_1"

    assert monitor.database_manager is not None
    assert monitor.database_manager.accesses
    assert monitor.database_manager.calls[-1] == ("insert_accesses", 1)


@pytest.mark.smoke
def test_get_latest_operations_restores_lookback(monitor, monkeypatch):
    monitor.lookback_minutes = 12
    monkeypatch.setattr(monitor, "get_onedata_read_operations", lambda: ["ok"])
    monkeypatch.setattr(monitor, "get_onedata_write_operations", lambda: [])

    results = monitor.get_latest_operations()

    assert results == ["ok"]
    assert monitor.lookback_minutes == 12