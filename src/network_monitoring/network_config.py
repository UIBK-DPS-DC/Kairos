import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

NETWORK_CONFIG_ENV = "KAIROS_NETWORK_CONFIG"


@dataclass(frozen=True)
class NetworkConfig:
    """Fixed network values that replace the probe measurements.

    ``pairwise_read_latency_ms``: ``latency[reader_id][holder_id]`` in ms, like
    :func:`check_pairwise_read_latency`.
    ``upload_rtt_ms``: provider ID -> upload RTT in ms, like
    :func:`check_all_providers_latency_with_ids`; only used for pairs missing
    from the pairwise matrix.
    ``replication_overhead_seconds`` / ``replication_bandwidth_bytes_per_second``:
    None keeps Kairos' defaults.
    """

    pairwise_read_latency_ms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    upload_rtt_ms: Dict[str, float] = field(default_factory=dict)
    replication_overhead_seconds: float | None = None
    replication_bandwidth_bytes_per_second: float | None = None


def load_network_config(path: str) -> NetworkConfig:
    """Read a :class:`NetworkConfig` from a JSON file. Keys starting with ``_`` are comments."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    def _optional_float(key: str) -> float | None:
        value = raw.get(key)
        return None if value is None else float(value)

    return NetworkConfig(
        pairwise_read_latency_ms={
            reader: {holder: float(ms) for holder, ms in row.items()}
            for reader, row in raw.get("pairwise_read_latency_ms", {}).items()
        },
        upload_rtt_ms={pid: float(ms) for pid, ms in raw.get("upload_rtt_ms", {}).items()},
        replication_overhead_seconds=_optional_float("replication_overhead_seconds"),
        replication_bandwidth_bytes_per_second=_optional_float("replication_bandwidth_bytes_per_second"),
    )


def network_config_from_env() -> NetworkConfig | None:
    """The config named by ``KAIROS_NETWORK_CONFIG``, or None to measure the network instead."""
    path = os.getenv(NETWORK_CONFIG_ENV, "").strip()
    if not path:
        return None
    config = load_network_config(path)
    logger.info("Using fixed network values from %s instead of probing: %s", path, config)
    return config
