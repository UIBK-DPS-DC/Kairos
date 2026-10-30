from unittest.mock import MagicMock

import pytest

from src.data_placement.strategies import Kairos as kairos_module
from src.data_placement.strategies.Kairos import Kairos

TASK_START = 1_700_000_000.0


def _kairos(ids, mtimes):
    # Skip __init__: it measures provider latencies over the network.
    kairos = Kairos.__new__(Kairos)
    kairos.onedata_manager = MagicMock()
    kairos.onedata_manager.get_file_id_from_path.side_effect = ids
    kairos.onedata_manager.get_file_mtime.side_effect = lambda file_id: mtimes[file_id]
    return kairos


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(kairos_module.time, "sleep", lambda _: None)


@pytest.mark.unit
def test_file_id_of_previous_run_is_retried_until_the_new_file_is_visible():
    kairos = _kairos(["old", "old", "new"], {"old": TASK_START - 30, "new": TASK_START + 4})

    assert kairos._get_file_id_with_retry("/space/out.txt", written_after=TASK_START) == "new"


@pytest.mark.unit
def test_file_id_is_taken_as_is_without_task_start_or_mtime():
    assert _kairos(["old"], {"old": TASK_START - 30})._get_file_id_with_retry("/space/out.txt") == "old"
    assert _kairos(["id"], {"id": None})._get_file_id_with_retry(
        "/space/out.txt", written_after=TASK_START
    ) == "id"


@pytest.mark.unit
def test_stale_file_id_is_not_returned_after_timeout():
    kairos = _kairos(["old"] * 100, {"old": TASK_START - 30})

    assert kairos._get_file_id_with_retry("/space/out.txt", written_after=TASK_START) is None
