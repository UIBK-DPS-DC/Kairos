"""Shared support types and helpers for the Onedata integration.

This module hosts configuration helpers, transfer models, and helper
utilities that are used by `onedatamanager.py` and other external callers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class OnedataConfig:
    token: str
    onezone_url: str = ""
    oneprovider_url: str = ""
    verify_ssl: bool = True

    @property
    def auth_headers(self) -> dict:
        return {"X-Auth-Token": self.token, "Content-Type": "application/json"}


def load_config(
    token: str,
    onezone_host: str = "",
    oneprovider_host: str = "",
    dev_mode: bool = False,
) -> OnedataConfig:
    scheme = "https"
    onezone_url = f"{scheme}://{onezone_host}" if onezone_host else ""
    oneprovider_url = f"{scheme}://{oneprovider_host}" if oneprovider_host else ""
    return OnedataConfig(
        token=token,
        onezone_url=onezone_url,
        oneprovider_url=oneprovider_url,
        verify_ssl=not dev_mode,
    )


class TransferType(str, Enum):
    REPLICATION = "replication"
    EVICTION = "eviction"
    MIGRATION = "migration"


class TransferStatus(str, Enum):
    SCHEDULED = "scheduled"
    ENQUEUED = "enqueued"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


@dataclass
class Transfer:
    transfer_id: str
    transfer_type: TransferType
    file_ref: str
    source_provider_id: Optional[str]
    destination_provider_id: Optional[str]
    status: TransferStatus = TransferStatus.SCHEDULED
    bytes_replicated: int = 0
    bytes_evicted: int = 0
    files_processed: int = 0
    start_time: int = None
    finish_time: int = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            TransferStatus.COMPLETED,
            TransferStatus.FAILED,
            TransferStatus.CANCELLED,
            TransferStatus.SKIPPED,
        )


@dataclass
class ReplicaDistribution:
    file_ref: str
    distribution_per_provider: dict = field(default_factory=dict)

    @property
    def provider_ids(self) -> list:
        return [
            pid
            for pid, info in self.distribution_per_provider.items()
            if info.get("blocksPercentage", 0) > 0
        ]

    @property
    def replica_count(self) -> int:
        return len(self.provider_ids)

    @property
    def fully_replicated_providers(self) -> list:
        return [
            pid
            for pid, info in self.distribution_per_provider.items()
            if info.get("blocksPercentage", 0) >= 100.0
        ]

    def blocks_percentage(self, provider_id: str) -> float:
        info = self.distribution_per_provider.get(provider_id, {})
        pct = info.get("blocksPercentage")
        if pct is not None:
            return float(pct)
        return _compute_blocks_percentage(info)

    def virtual_size(self, provider_id: str) -> int:
        return self.distribution_per_provider.get(provider_id, {}).get("virtualSize", 0)

    @property
    def maximum_physical_size(self) -> int:
        """
        Return the largest physical size across all providers.

        The modern API nests physical sizes inside
        ``distributionPerStorageBackend`` (or the older ``distributionPerStorage``),
        not at the top level of the provider entry.  We sum across all storage
        backends for each provider and take the maximum.

        Falls back to ``virtualSize`` as a proxy when no physical size is
        reported (e.g. the provider is a caching layer with 0-byte blocks).
        """
        max_size = 0
        for info in self.distribution_per_provider.values():
            per_storage = (
                info.get("distributionPerStorageBackend")
                or info.get("distributionPerStorage")
                or {}
            )
            if per_storage:
                provider_physical = sum(
                    sinfo.get("physicalSize", 0) or 0
                    for sinfo in per_storage.values()
                )
            else:
                # Older API or missing key: try top-level fields
                provider_physical = info.get("physicalSize", 0) or info.get("logicalSize", 0) or 0

            if provider_physical == 0:
                # Last resort: use virtualSize as an approximation
                provider_physical = info.get("virtualSize", 0) or 0

            if provider_physical > max_size:
                max_size = provider_physical
        return max_size

@dataclass
class ProviderSummary:
    provider_id: str
    name: str
    domain: str

    @property
    def provider_url(self) -> str:
        d = self.domain.strip()
        if d and "." in d and " " not in d:
            return f"https://{d}"
        return ""


def _compute_blocks_percentage(provider_info: dict) -> float:
    virtual_size = provider_info.get("virtualSize", 0)
    # Support both the modern key name (distributionPerStorageBackend) and the
    # older one (distributionPerStorage) so the function works across API versions.
    per_storage = provider_info.get("distributionPerStorageBackend") or provider_info.get("distributionPerStorage", {})
    if per_storage:
        covered = 0
        for sinfo in per_storage.values():
            for block in sinfo.get("blocks", []):
                if len(block) >= 2:
                    covered += block[1]
        if virtual_size and virtual_size > 0:
            return min(100.0, covered / virtual_size * 100.0)
        if covered > 0:
            return 100.0

    for sinfo in per_storage.values():
        physical = sinfo.get("physicalSize", 0) or sinfo.get("logicalSize", 0)
        if physical and virtual_size and virtual_size > 0:
            return min(100.0, physical / virtual_size * 100.0)

    physical = provider_info.get("physicalSize", 0)
    if physical and virtual_size and virtual_size > 0:
        return min(100.0, physical / virtual_size * 100.0)

    return 0.0


__all__ = [
    "OnedataConfig",
    "load_config",
    "TransferType",
    "TransferStatus",
    "Transfer",
    "ReplicaDistribution",
    "ProviderSummary",
    "_compute_blocks_percentage",
]
