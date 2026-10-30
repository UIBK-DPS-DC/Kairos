from abc import ABC, abstractmethod
from typing import Optional

from src.rest_api.onedata.onedata_helpers import Transfer


class StorageInterface(ABC):
    """Abstract interface for storage backends used by placement logic."""

    provider_name: str = ""

    @staticmethod
    @abstractmethod
    def load_config(
        token: str,
        onezone_host: str = "",
        oneprovider_host: str = "",
        dev_mode: bool = False,
    ):
        """Build backend-specific configuration object from plain values."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_storage_manager(
        cls,
        config,
        *,
        provider_url: Optional[str] = None,
        provider_name: str = "",
        space_name: str = "",
        space_id: str = "",
        use_guid: bool = True,
    ) -> "StorageInterface":
        """Build a concrete storage manager instance from backend config."""
        raise NotImplementedError

    @abstractmethod
    def list_providers(self) -> list:
        """Return reachable providers for the current credentials."""
        raise NotImplementedError

    @abstractmethod
    def resolve_provider_id(self, hostname_or_id: str) -> str:
        """Resolve a hostname/domain/provider alias to an opaque provider ID."""
        raise NotImplementedError

    @abstractmethod
    def get_replica_distribution(self, file_ref: str):
        """Return per-provider replica information for a file."""
        raise NotImplementedError

    @abstractmethod
    def replicate(
        self,
        file_ref: str,
        destination_provider_id: str,
        *,
        callback_url: Optional[str] = None,
        zone_manager=None,
    ):
        """Schedule replication of a file to the target provider."""
        raise NotImplementedError

    @abstractmethod
    def evict(
        self,
        file_ref: str,
        source_provider_id: str,
        *,
        callback_url: Optional[str] = None,
    ):
        """Schedule eviction of a file replica from the source provider."""
        raise NotImplementedError

    @abstractmethod
    def migrate(
        self,
        file_ref: str,
        source_provider_id: str,
        destination_provider_id: str,
        *,
        callback_url: Optional[str] = None,
    ):
        """Schedule migration from source provider to destination provider."""
        raise NotImplementedError

    @abstractmethod
    def wait_for_transfer(
        self, transfer, poll_interval: float = 5.0, timeout: float = 300.0
    ):
        """Block until transfer reaches a terminal state."""
        raise NotImplementedError

    @abstractmethod
    def upload_file(
        self,
        filename: str,
        content: bytes | str,
        parent_id: Optional[str] = None,
        specific_space_id: Optional[str] = None,
    ) -> str:
        """Upload a file and return its file ID."""
        raise NotImplementedError

    @abstractmethod
    def delete_file(self, file_id: str) -> None:
        """Delete a file by ID."""
        raise NotImplementedError

    @abstractmethod
    def read_file(self, file_id: str) -> bytes:
        """Read the content of a file through this provider."""
        raise NotImplementedError

    @abstractmethod
    def has_file(self, file_id: str) -> bool:
        """Check whether a file exists."""
        raise NotImplementedError

    @abstractmethod
    def update_transfer_status(self, transfer: Transfer) -> Transfer:
        """Update the status of a transfer."""
        raise NotImplementedError

    @abstractmethod
    def list_ended_transfer_details(self, limit: int = 1000) -> list[dict]:
        """Return transfer IDs available for interval metric collection."""
        raise NotImplementedError

    @abstractmethod
    def get_file_size(self, file_id: str) -> int:
        """Get the size of a file in bytes."""
        raise NotImplementedError

    @abstractmethod
    def get_data_distribution(self) -> dict:
        """Get the current data distribution across providers."""
        raise NotImplementedError

    @abstractmethod
    def get_file_popularity(self, zone_manager: "StorageInterface") -> dict:
        """Get the ordered popularity for all files."""
        raise NotImplementedError

    def get_transfer_status(self, transfer_id: str) -> dict:
        raise NotImplementedError("get_transfer_status is not implemented for this storage backend")