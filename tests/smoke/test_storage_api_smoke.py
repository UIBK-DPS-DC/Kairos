"""Smoke tests for StorageInterface and OnedataDataManager."""
import os
import time
import pytest
import requests
from pathlib import Path
from dotenv import load_dotenv

from src.rest_api.onedata.onedatamanager import OnedataDataManager
from src.rest_api.storage_interface import StorageInterface

# Load .env file automatically
load_dotenv()

def load_env_config() -> dict:
    """Load Onedata credentials and host settings from environment variables."""
    # Retrieve required variables
    space_id = os.getenv("ONEDATA_SPACE_ID")
    token = os.getenv("ONEDATA_TOKEN_PROVIDER")
    onezone_url = os.getenv("ONEDATA_ONEZONE_URL", "")
    oneprovider_url = os.getenv("ONEDATA_PROVIDER_HOST")

    if not all([space_id, token, oneprovider_url]):
        raise RuntimeError("Missing required ONEDATA environment variables.")

    return {
        "onezone_url": onezone_url,
        "oneprovider_url": oneprovider_url,
        "space_id": space_id,
        "token": token,
    }

@pytest.fixture(scope="session", name="storage_manager")
def fixture_storage_manager() -> StorageInterface:
    """Create a real Onedata-backed storage manager from .env config."""
    cfg = load_env_config()

    # Clean URLs (remove https:// if provided in env)
    config = OnedataDataManager.load_config(
        token=cfg["token"],
        onezone_host=cfg["onezone_url"].replace("https://", ""),
        oneprovider_host=cfg["oneprovider_url"].replace("https://", ""),
    )
    return OnedataDataManager.build_storage_manager(
        config=config, space_id=cfg["space_id"]
    )

@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_storage_interface_smoke_contract(storage_manager: StorageInterface):
    """Validate a real storage manager satisfies the interface contract."""
    assert isinstance(storage_manager, StorageInterface)
    assert callable(storage_manager.list_providers)
    assert callable(storage_manager.upload_file)
    assert callable(storage_manager.delete_file)
    assert callable(storage_manager.has_file)


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_onedatamanager_provider_resolution_smoke(storage_manager: StorageInterface):
    """Validate provider discovery and provider-id resolution on real backend."""
    try:
        providers = storage_manager.list_providers()
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Onedata backend unavailable for smoke run: {exc}")

    assert isinstance(providers, list)
    assert len(providers) > 0, "No providers returned by list_providers()"

    first = providers[0]
    resolved_by_id = storage_manager.resolve_provider_id(first.provider_id)
    assert resolved_by_id == first.provider_id

    if first.domain:
        resolved_by_domain = storage_manager.resolve_provider_id(first.domain)
        assert resolved_by_domain == first.provider_id


@pytest.mark.smoke
@pytest.mark.integration
@pytest.mark.timeout(180)
def test_onedatamanager_file_lifecycle_smoke(storage_manager: StorageInterface):
    """Validate upload, existence check, and delete lifecycle on real backend."""
    filename = f"smoke_probe_{int(time.time())}.txt"
    payload = b"smoke-test-payload"
    file_id = None

    try:
        file_id = storage_manager.upload_file(filename=filename, content=payload)
        assert file_id, "upload_file() returned empty file id"
        assert storage_manager.has_file(file_id) is True
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Onedata backend unavailable for smoke run: {exc}")
    finally:
        if file_id:
            storage_manager.delete_file(file_id)
