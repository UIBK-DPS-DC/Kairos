from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.dag_monitoring.monitor_db import _build_access_id, _parse_series, _persist_accesses
from src.database.db_interface import AccessType
from src.database.sqlite3_manager import SQLite3Manager

SESSION = "dag_1:1700000000.5:task_1:1700000001.25"


def _matrix(*samples):
    return {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "space_id": "space_1",
                    "file_id": "file_1",
                    "provider_id": "provider_1",
                    "session_id": SESSION,
                },
                "values": [list(s) for s in samples],
            }
        ],
    }


@pytest.mark.unit
def test_parse_series_yields_one_access_per_series():
    records = _parse_series(
        _matrix((1700000015, "0.2"), (1700000030, "0.5"), (1700000045, "0.5")), AccessType.READ
    )

    assert len(records) == 1
    record = records[0]
    assert record["duration_seconds"] == 0.5
    assert record["task_id"] == "task_1"
    assert record["task_execution_id"] == "1700000001.25"
    assert record["operation"] == "read"


@pytest.mark.unit
def test_access_id_is_stable_across_query_windows():
    earlier = _parse_series(_matrix((1700000015, "0.5"), (1700000030, "0.5")), AccessType.READ)
    later = _parse_series(_matrix((1700000022, "0.5"), (1700000037, "0.5")), AccessType.READ)

    assert _build_access_id(earlier[0]) == _build_access_id(later[0])


@pytest.mark.unit
def test_repeated_ingestion_stores_access_once(tmp_path: Path):
    db = SQLite3Manager(tmp_path / "kairos.sqlite3", initialize_db=True)
    data_manager = MagicMock()
    data_manager.get_file_path.return_value = "/space/file_1"

    _persist_accesses(_parse_series(_matrix((1700000015, "0.2")), AccessType.READ), db, data_manager)
    _persist_accesses(
        _parse_series(_matrix((1700000022, "0.5"), (1700000037, "0.5")), AccessType.READ),
        db,
        data_manager,
    )

    accesses = db.read_accesses_by_task_execution("1700000001.25")
    assert len(accesses) == 1
    assert accesses[0].duration_seconds == 0.5  # counter grew while the task was running
    assert accesses[0].timepoint == datetime.fromtimestamp(1700000015).isoformat()  # earliest sample kept
    data_manager.get_file_path.assert_called_once()  # known access: no second path lookup


@pytest.mark.unit
def test_first_access_time_is_the_earliest_access_of_the_execution(tmp_path: Path):
    db = SQLite3Manager(tmp_path / "kairos.sqlite3", initialize_db=True)
    data_manager = MagicMock()
    data_manager.get_file_path.return_value = "/space/file_1"
    later_file = _matrix((1700000045, "0.1"))
    later_file["result"][0]["metric"]["file_id"] = "file_2"

    _persist_accesses(_parse_series(later_file, AccessType.READ), db, data_manager)
    _persist_accesses(_parse_series(_matrix((1700000030, "0.2")), AccessType.READ), db, data_manager)

    assert db.read_first_access_times_by_task("task_1", "dag_1") == {
        "1700000001.25": datetime.fromtimestamp(1700000030).isoformat()
    }
