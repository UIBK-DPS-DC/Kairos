"""Smoke tests for Onedata provider latency and connectivity."""

import os
import pytest
from dotenv import load_dotenv

from src.network_monitoring.provider_latency import check_provider_latency
from src.rest_api.onedata.onedatamanager import OnedataDataManager
from src.rest_api.storage_interface import StorageInterface

# Load environment variables from .env
load_dotenv()


def get_onedata_config() -> dict:
    """Helper to retrieve and validate environment configuration."""
    space_id = os.getenv("ONEDATA_SPACE_ID")
    token = os.getenv("ONEDATA_TOKEN_PROVIDER")
    onezone_host = os.getenv("ONEDATA_ONEZONE_URL", "").replace("https://", "")
    oneprovider_host = os.getenv("ONEDATA_PROVIDER_HOST", "").replace("https://", "")

    if not all([space_id, token, oneprovider_host]):
        raise EnvironmentError(
            "Missing mandatory Onedata environment variables. "
            "Ensure ONEDATA_SPACE_ID, ONEDATA_TOKEN_PROVIDER, and ONEDATA_PROVIDER_HOST are set."
        )

    return {
        "space_id": space_id,
        "token": token,
        "onezone_host": onezone_host,
        "oneprovider_host": oneprovider_host,
    }


@pytest.fixture(scope="session", name="storage_manager")
def fixture_storage_manager() -> StorageInterface:
    """Create a real Onedata-backed storage manager using environment settings."""
    cfg = get_onedata_config()

    config = OnedataDataManager.load_config(
        token=cfg["token"],
        onezone_host=cfg["onezone_host"],
        oneprovider_host=cfg["oneprovider_host"],
    )

    return OnedataDataManager.build_storage_manager(
        config=config, space_id=cfg["space_id"]
    )


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_provider_latency_smoke(storage_manager: StorageInterface):
    """
    Validates that latency measurement executes successfully.
    """
    assert storage_manager is not None, "Storage manager failed to initialize."

    # Run the latency check
    result = check_provider_latency(storage_manager, rounds=3)

    if result is None:
        pytest.fail("check_provider_latency returned None.")

    # FIX: Handle both types of responses
    if isinstance(result, (float, int)):
        latency = float(result)
    elif isinstance(result, dict):
        latency = result.get("latency_ms")
        assert latency is not None, "Dictionary returned but missing 'latency_ms' key"
    else:
        pytest.fail(f"Unexpected return type from check_provider_latency: {type(result)}")

    # Validate the value
    assert 0 < latency < 10_000, f"Latency value {latency}ms is out of bounds."