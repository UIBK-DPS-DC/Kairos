from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from src.network_monitoring.provider_latency import (
    check_provider_latency,
    check_all_providers_latency_with_names,
    predict_closest_provider,
    PROBE_SIZE_BYTES,
)


@pytest.fixture
def dummy_manager():
    mgr = MagicMock()
    mgr.space_id = "dummy-space"
    # Mock upload_file to return deterministic file ids
    mgr.upload_file.side_effect = (
        lambda filename, content, parent_id=None, specific_space_id=None: f"id_{filename}"
    )
    # Mock delete_file to do nothing
    mgr.delete_file.return_value = None
    return mgr


@pytest.mark.unit
@patch("os.urandom")
@patch("builtins.open")
@patch("os.unlink")
@patch("time.perf_counter")
def test_check_provider_latency(
    mock_perf, mock_unlink, mock_open, mock_urandom, dummy_manager
):
    """Unit test for check_provider_latency with mocks."""

    # Mock random bytes for probe creation
    mock_urandom.return_value = b"\x00" * PROBE_SIZE_BYTES

    # Simulate perf_counter for RTT measurement
    # Round-trip times of 0.01s, 0.02s, 0.03s
    mock_perf.side_effect = [0.0, 0.01, 0.01, 0.03, 0.03, 0.06]

    avg = check_provider_latency(dummy_manager, rounds=3)

    # Should return average of RTTs in milliseconds
    expected_rtts_ms = [(0.01 - 0.0) * 1000, (0.03 - 0.01) * 1000, (0.06 - 0.03) * 1000]
    expected_avg = sum(expected_rtts_ms) / len(expected_rtts_ms)
    assert avg == expected_avg

    # Ensure upload_file called 3 times
    assert dummy_manager.upload_file.call_count == 3

    # Ensure delete_file called for each uploaded file
    assert dummy_manager.delete_file.call_count == 3

    # Ensure local probe file is deleted
    mock_unlink.assert_called_once()
    deleted_name = Path(mock_unlink.call_args.args[0]).name
    assert deleted_name.startswith("latency_probe_")
    assert deleted_name.endswith(".bin")


@pytest.mark.unit
def test_check_all_providers_latency(dummy_manager):
    """Unit test for check_all_providers_latency."""
    managers = {"provider1": dummy_manager, "provider2": dummy_manager}

    # Patch check_provider_latency to return deterministic RTTs
    with patch(
        "src.network_monitoring.provider_latency.check_provider_latency"
    ) as mock_check:
        mock_check.side_effect = [10.0, 20.0]
        result = check_all_providers_latency_with_names(managers, rounds=1)

    assert result == {"provider1": 10.0, "provider2": 20.0}


@pytest.mark.unit
def test_predict_closest_provider(dummy_manager):
    """Unit test for predict_closest_provider."""
    managers = {"A": dummy_manager, "B": dummy_manager, "C": dummy_manager}

    with patch(
        "src.network_monitoring.provider_latency.check_all_providers_latency_with_names"
    ) as mock_all:
        mock_all.return_value = {"A": 15.0, "B": 5.0, "C": None}
        closest = predict_closest_provider(managers)

    # B has the lowest RTT
    assert closest == "B"


@pytest.mark.unit
def test_predict_closest_provider_all_fail(dummy_manager):
    """Return None if all providers fail."""
    managers = {"X": dummy_manager, "Y": dummy_manager}

    with patch(
        "src.network_monitoring.provider_latency.check_all_providers_latency_with_names"
    ) as mock_all:
        mock_all.return_value = {"X": None, "Y": None}
        closest = predict_closest_provider(managers)

    assert closest is None
