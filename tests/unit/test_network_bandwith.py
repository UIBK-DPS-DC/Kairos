from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest

from src.network_monitoring import network_bandwith as nb
from src.rest_api.onedata.onedata_helpers import Transfer, TransferType


def _make_transfer(
    transfer_id: str = "t-1", start_time: int = 0, finish_time: int = 200
) -> Transfer:
    return Transfer(
        transfer_id=transfer_id,
        transfer_type=TransferType.REPLICATION,
        file_ref="guid#f",
        source_provider_id=None,
        destination_provider_id="pid-dst",
        start_time=start_time,
        finish_time=finish_time,
    )


@pytest.mark.unit
def test_create_probe_file_writes_expected_bytes(tmp_path, monkeypatch):
    target = tmp_path / "probe.bin"
    payload = b"x" * nb.PROBE_SIZE_BYTES
    monkeypatch.setattr(nb.os, "urandom", lambda size: payload)

    nb._create_probe_file(str(target))

    assert target.read_bytes() == payload
    assert len(target.read_bytes()) == nb.PROBE_SIZE_BYTES


@pytest.mark.unit
def test_extract_transfer_time_seconds_returns_seconds():
    transfer = _make_transfer(start_time=100, finish_time=350)

    result = nb.extract_transfer_time_seconds(transfer)

    assert result == 250.0


@pytest.mark.unit
def test_extract_transfer_time_seconds_raises_when_missing_timestamps():
    transfer = _make_transfer(start_time=None, finish_time=350)

    with pytest.raises(nb.TransferNotCompletedException):
        nb.extract_transfer_time_seconds(transfer)


@pytest.mark.unit
def test_calculate_bandwidth_bytes_per_second_returns_expected_value():
    transfer = _make_transfer(start_time=0, finish_time=200)

    assert nb.calculate_bandwidth_bytes_per_second(transfer, 100) == 0.5


@pytest.mark.unit
@pytest.mark.parametrize(
    "start_time,finish_time,file_size",
    [
        (100, 100, 100),
        (0, 200, 0),
        (0, 200, -1),
    ],
)
def test_calculate_bandwidth_bytes_per_second_raises_on_invalid_input(
    start_time, finish_time, file_size
):
    transfer = _make_transfer(start_time=start_time, finish_time=finish_time)

    with pytest.raises(ValueError):
        nb.calculate_bandwidth_bytes_per_second(transfer, file_size)


@pytest.mark.unit
def test_mirror_array_across_hypothenuse_mirrors_upper_triangle_in_place():
    matrix = [
        [float("inf"), 11.0, 12.0],
        [float("inf"), float("inf"), 23.0],
        [float("inf"), float("inf"), float("inf")],
    ]

    result = nb.mirror_array_across_hypothenuse(matrix)

    assert result is matrix
    assert result == [
        [float("inf"), 11.0, 12.0],
        [11.0, float("inf"), 23.0],
        [12.0, 23.0, float("inf")],
    ]


def _patch_probe_creation(monkeypatch):
    def fake_create_probe_file(path):
        with open(path, "wb") as f:
            f.write(b"z" * nb.PROBE_SIZE_BYTES)

    monkeypatch.setattr(nb, "_create_probe_file", fake_create_probe_file)


@pytest.mark.unit
def test_check_network_bandwidth_list_input_success(monkeypatch):
    payload = b"z" * nb.PROBE_SIZE_BYTES
    monkeypatch.setattr(nb, "_create_probe_file", lambda _: None)
    monkeypatch.setattr("builtins.open", mock_open(read_data=payload))
    unlink_calls = []
    monkeypatch.setattr(nb.os, "unlink", lambda path: unlink_calls.append(path))

    provider0 = type(
        "Provider",
        (),
        {"provider_id": "pid-0", "name": "provider-0", "domain": "p0.example"},
    )()
    provider1 = type(
        "Provider",
        (),
        {"provider_id": "pid-1", "name": "provider-1", "domain": "p1.example"},
    )()
    zone_manager = MagicMock()
    zone_manager.list_providers.return_value = [provider0, provider1]

    transfer = _make_transfer(transfer_id="tr-1", start_time=0, finish_time=200)
    mgr0 = MagicMock()
    mgr0.space_id = "space-0"
    mgr0.upload_file.return_value = "guid#file-1"
    mgr0.replicate.return_value = transfer

    mgr1 = MagicMock()
    mgr1.space_id = "space-1"

    result = nb.check_network_bandwidth([mgr0, mgr1], zone_manager)

    expected_bw = nb.PROBE_SIZE_BYTES / 200
    assert result["pid-0"]["pid-1"] == expected_bw
    assert result["pid-1"]["pid-0"] == expected_bw
    assert result["pid-0"]["pid-0"] == float("inf")
    assert result["pid-1"]["pid-1"] == float("inf")
    mgr0.upload_file.assert_called_once_with(
        filename=nb.PROBE_FILENAME,
        content=payload,
    )
    mgr0.replicate.assert_called_once_with(
        file_ref="guid#file-1",
        destination_provider_id="pid-1",
        zone_manager=zone_manager,
    )
    assert len(unlink_calls) == 1
    assert Path(unlink_calls[0]).name == nb.PROBE_FILENAME


@pytest.mark.unit
def test_check_network_bandwidth_mapping_input_retries_after_wait_failure(monkeypatch):
    _patch_probe_creation(monkeypatch)
    sleep_calls = []
    unlink_calls = []
    monkeypatch.setattr(nb.time, "sleep", lambda value: sleep_calls.append(value))
    monkeypatch.setattr(nb.os, "unlink", lambda path: unlink_calls.append(path))

    provider0 = type(
        "Provider",
        (),
        {"provider_id": "pid-0", "name": "provider-0", "domain": "p0.example"},
    )()
    provider1 = type(
        "Provider",
        (),
        {"provider_id": "pid-1", "name": "provider-1", "domain": "p1.example"},
    )()
    zone_manager = MagicMock()
    zone_manager.list_providers.return_value = [provider0, provider1]

    transfer = _make_transfer(transfer_id="tr-retry", start_time=100, finish_time=300)
    mgr0 = MagicMock()
    mgr0.space_id = "space-0"
    mgr0.upload_file.return_value = "guid#file-1"
    mgr0.replicate.return_value = transfer
    mgr0.wait_for_transfer.side_effect = [RuntimeError("temporary"), transfer]

    mgr1 = MagicMock()
    mgr1.space_id = "space-1"

    result = nb.check_network_bandwidth({"a": mgr0, "b": mgr1}, zone_manager)

    expected_bw = nb.PROBE_SIZE_BYTES / 200
    assert result["pid-0"]["pid-1"] == expected_bw
    assert result["pid-1"]["pid-0"] == expected_bw
    assert sleep_calls == [0.4]
    assert len(unlink_calls) == 1
    assert Path(unlink_calls[0]).name == nb.PROBE_FILENAME


@pytest.mark.unit
def test_check_network_bandwidth_handles_replication_failure(monkeypatch):
    _patch_probe_creation(monkeypatch)
    monkeypatch.setattr(nb.os, "unlink", lambda _: None)

    provider0 = type(
        "Provider",
        (),
        {"provider_id": "pid-0", "name": "provider-0", "domain": "p0.example"},
    )()
    provider1 = type(
        "Provider",
        (),
        {"provider_id": "pid-1", "name": "provider-1", "domain": "p1.example"},
    )()
    zone_manager = MagicMock()
    zone_manager.list_providers.return_value = [provider0, provider1]

    mgr0 = MagicMock()
    mgr0.space_id = "space-0"
    mgr0.upload_file.return_value = "guid#file-1"
    mgr0.replicate.side_effect = RuntimeError("replication failed")

    mgr1 = MagicMock()
    mgr1.space_id = "space-1"

    result = nb.check_network_bandwidth([mgr0, mgr1], zone_manager)

    assert result == {
        "pid-0": {"pid-0": float("inf"), "pid-1": float("inf")},
        "pid-1": {"pid-0": float("inf"), "pid-1": float("inf")},
    }
    mgr0.delete_file.assert_called_once_with("guid#file-1")
    mgr1.delete_file.assert_called_once_with("guid#file-1")


@pytest.mark.unit
def test_check_network_bandwidth_handles_upload_failure(monkeypatch):
    _patch_probe_creation(monkeypatch)
    unlink_calls = []
    monkeypatch.setattr(nb.os, "unlink", lambda path: unlink_calls.append(path))

    provider0 = type(
        "Provider",
        (),
        {"provider_id": "pid-0", "name": "provider-0", "domain": "p0.example"},
    )()
    provider1 = type(
        "Provider",
        (),
        {"provider_id": "pid-1", "name": "provider-1", "domain": "p1.example"},
    )()
    zone_manager = MagicMock()
    zone_manager.list_providers.return_value = [provider0, provider1]

    mgr0 = MagicMock()
    mgr0.space_id = "space-0"
    mgr0.upload_file.side_effect = RuntimeError("upload failed")

    mgr1 = MagicMock()
    mgr1.space_id = "space-1"
    mgr1.upload_file.return_value = "guid#file-2"

    result = nb.check_network_bandwidth([mgr0, mgr1], zone_manager)

    assert result == {
        "pid-0": {"pid-0": float("inf"), "pid-1": float("inf")},
        "pid-1": {"pid-0": float("inf"), "pid-1": float("inf")},
    }
    mgr0.delete_file.assert_not_called()
    mgr1.delete_file.assert_called_once_with("guid#file-2")
    assert len(unlink_calls) == 1
    assert Path(unlink_calls[0]).name == nb.PROBE_FILENAME
