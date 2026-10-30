import time
import os
import tempfile
from typing import Dict, Mapping, Optional
from src.rest_api.storage_interface import StorageInterface
import random as rand
import logging

PROBE_FILENAME = "latency_probe_1mb.bin"
PROBE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

logger = logging.getLogger(__name__)


def _create_probe_file(path: str):
    """Write 1 MB of deterministic bytes to a local temp file."""
    with open(path, "wb") as f:
        f.write(os.urandom(PROBE_SIZE_BYTES))


def check_provider_latency(
    manager: StorageInterface, devnull_id: Optional[str] = None, rounds: int = 3
):
    """
    Upload a 1 MB probe file to the manager's space root and measure
    the average upload RTT across `rounds` requests.
    """
    with tempfile.NamedTemporaryFile(
        prefix="latency_probe_",
        suffix=".bin",
        delete=False,
    ) as tmp_file:
        local_path = tmp_file.name

    _create_probe_file(local_path)

    avg = None

    with open(local_path, "rb") as f:
        content = f.read()

    rtts = []
    file_ids = []
    run_id = int(time.time() + rand.randint(0, 1000))
    for i in range(rounds):
        filename = f"latency_probe_{run_id}_round{i}.bin"
        try:
            t0 = time.perf_counter()
            file_id = manager.upload_file(
                filename=filename, content=content, specific_space_id=devnull_id
            )
            rtt = (time.perf_counter() - t0) * 1000
            rtts.append(rtt)
            file_ids.append(file_id)
            # print(f"  Round {i+1}: uploaded '{filename}' ({PROBE_SIZE_BYTES // 1024} KB) in {rtt:.2f} ms  [id={file_id}]")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("  Round %d: upload failed: %s", i + 1, exc)

    if rtts:
        avg = sum(rtts) / len(rtts)
        logger.info(
            "\nAverage upload RTT: %.2f ms over %d round(s) for %s on manager %s",
            avg,
            len(rtts),
            manager.space_id,
            manager.provider_name,
        )
    else:
        logger.error("All uploads failed.")

    # print("\nCleaning up probe files...")
    for file_id in file_ids:
        try:
            manager.delete_file(file_id)
            # print(f"  Deleted file {file_id}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("  Failed to delete %s: %s", file_id, exc)

    try:
        os.unlink(local_path)
    except OSError as exc:
        logger.warning("Failed to remove local probe file %s: %s", local_path, exc)

    return avg


def check_pairwise_provider_latency(
    zone_manager: StorageInterface,
    managers: Mapping[str, StorageInterface],
    rounds: int = 3,
    max_retries: int = 3,
    transfer_timeout: float = 300.0,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Measure directed transfer latency between every pair of providers.

    A retry is started only after the transfer remains unfinished for
    ``transfer_timeout`` seconds, preventing duplicate transfers from being
    queued while the original transfer is still active.
    """
    manager_items = list(managers.items())
    provider_ids = []
    for provider_key, manager in manager_items:
        provider_id = provider_key
        try:
            provider_id = zone_manager.resolve_provider_id(provider_key)
        except (ValueError, RuntimeError, AttributeError):
            manager_name = getattr(manager, "provider_name", "")
            if manager_name:
                try:
                    provider_id = zone_manager.resolve_provider_id(manager_name)
                except (ValueError, RuntimeError, AttributeError):
                    logger.error("Could not resolve provider ID for %r.", provider_key)
        provider_ids.append(provider_id)

    latency_map: Dict[str, Dict[str, Optional[float]]] = {
        source_id: {
            target_id: 0.0 if source_id == target_id else None
            for target_id in provider_ids
        }
        for source_id in provider_ids
    }

    with tempfile.NamedTemporaryFile(
        prefix="pairwise_latency_probe_", suffix=".bin", delete=False
    ) as tmp_file:
        local_path = tmp_file.name

    try:
        _create_probe_file(local_path)
        with open(local_path, "rb") as probe_file:
            content = probe_file.read()

        run_id = int(time.time() + rand.randint(0, 1000))
        for source_index, (_, source_manager) in enumerate(manager_items):
            source_id = provider_ids[source_index]
            file_id = None
            try:
                file_id = source_manager.upload_file(
                    filename=f"{PROBE_FILENAME}_{run_id}_{source_index}",
                    content=content,
                )
            except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                logger.error("Latency probe upload failed for %s: %s", source_id, exc)
                continue

            try:
                for target_index in range(len(manager_items)):
                    target_id = provider_ids[target_index]
                    if source_index == target_index:
                        continue

                    measurements = []
                    for round_index in range(rounds):
                        for retry_index in range(max_retries):
                            try:
                                started_at = time.perf_counter()
                                transfer = source_manager.replicate(
                                    file_ref=file_id,
                                    destination_provider_id=provider_ids[target_index],
                                    zone_manager=zone_manager,
                                )
                                source_manager.wait_for_transfer(
                                    transfer=transfer, timeout=transfer_timeout
                                )
                                measurements.append(
                                    (time.perf_counter() - started_at) * 1000
                                )
                                break
                            except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                                logger.error(
                                    "Latency probe failed from %s to %s on round %d "
                                    "(retry %d/%d): %s",
                                    source_id,
                                    target_id,
                                    round_index + 1,
                                    retry_index + 1,
                                    max_retries,
                                    exc,
                                )

                    if measurements:
                        latency_map[source_id][target_id] = sum(measurements) / len(
                            measurements
                        )
            except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                logger.error(
                    "Latency matrix update failed for %s: %s", source_id, exc
                )
            finally:
                if file_id is not None:
                    for _, manager in manager_items:
                        try:
                            manager.delete_file(file_id)
                        except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                            logger.debug("Failed to delete latency probe %s: %s", file_id, exc)
    finally:
        try:
            os.unlink(local_path)
        except OSError as exc:
            logger.warning("Failed to remove local probe file %s: %s", local_path, exc)

    return latency_map


def _resolve_provider_id(zone_manager: StorageInterface, provider_key: str, manager: StorageInterface) -> str:
    try:
        return zone_manager.resolve_provider_id(provider_key)
    except (ValueError, RuntimeError, AttributeError):
        manager_name = getattr(manager, "provider_name", "")
        if manager_name:
            try:
                return zone_manager.resolve_provider_id(manager_name)
            except (ValueError, RuntimeError, AttributeError):
                pass
        logger.error("Could not resolve provider ID for %r; using key as-is.", provider_key)
        return provider_key


def _wait_until_visible(manager: StorageInterface, file_id: str, timeout: float, poll_interval: float) -> bool:
    """Wait until the metadata of a file uploaded elsewhere reached this provider."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if manager.has_file(file_id):
                return True
        except Exception:  # pylint: disable=broad-except,broad-exception-caught
            pass
        time.sleep(poll_interval)
    return False


def _timed_read_ms(manager: StorageInterface, file_id: str) -> float:
    started_at = time.perf_counter()
    manager.read_file(file_id)
    return (time.perf_counter() - started_at) * 1000


def check_pairwise_read_latency(
    zone_manager: StorageInterface,
    managers: Mapping[str, StorageInterface],
    rounds: int = 3,
    probe_size_bytes: int = 4096,
    visibility_timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Measure the latency Onedata adds when a provider reads a file stored on another one.

    Returns ``latency[reader_id][holder_id]`` in ms: the extra time a read
    through ``reader`` takes when the file is only stored on ``holder``, which
    is the on-the-fly fetch of a task running next to ``reader``. A local read
    through ``reader`` is measured as well and subtracted, so the time between
    Kairos and ``reader`` cancels out.

    Every probe is read exactly once, so the fetch always comes from ``holder``
    and never from a provider that cached the probe in an earlier read.
    """
    manager_items = list(managers.items())
    provider_ids = [
        _resolve_provider_id(zone_manager, provider_key, manager)
        for provider_key, manager in manager_items
    ]
    latency_map: Dict[str, Dict[str, Optional[float]]] = {
        reader_id: {
            holder_id: 0.0 if reader_id == holder_id else None
            for holder_id in provider_ids
        }
        for reader_id in provider_ids
    }

    content = os.urandom(probe_size_bytes)
    run_id = int(time.time() + rand.randint(0, 1000))
    probe_index = 0

    def upload_probe(manager: StorageInterface) -> str:
        nonlocal probe_index
        probe_index += 1
        return manager.upload_file(
            filename=f"read_latency_probe_{run_id}_{probe_index}.bin", content=content
        )

    def delete_probe(manager: StorageInterface, file_id: str) -> None:
        try:
            manager.delete_file(file_id)
        except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
            logger.debug("Failed to delete read latency probe %s: %s", file_id, exc)

    for reader_index, (_, reader) in enumerate(manager_items):
        reader_id = provider_ids[reader_index]

        local_reads = []
        for _ in range(rounds):
            file_id = None
            try:
                file_id = upload_probe(reader)
                local_reads.append(_timed_read_ms(reader, file_id))
            except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                logger.error("Local read probe failed on %s: %s", reader_id, exc)
            finally:
                if file_id is not None:
                    delete_probe(reader, file_id)
        if not local_reads:
            continue
        local_latency = sum(local_reads) / len(local_reads)

        for holder_index, (_, holder) in enumerate(manager_items):
            holder_id = provider_ids[holder_index]
            if holder_index == reader_index:
                continue

            remote_reads = []
            for _ in range(rounds):
                file_id = None
                try:
                    file_id = upload_probe(holder)
                    if not _wait_until_visible(reader, file_id, visibility_timeout, poll_interval):
                        logger.error(
                            "Probe %s uploaded to %s not visible on %s after %.0f s.",
                            file_id, holder_id, reader_id, visibility_timeout,
                        )
                        continue
                    remote_reads.append(_timed_read_ms(reader, file_id))
                except Exception as exc:  # pylint: disable=broad-except,broad-exception-caught
                    logger.error("Remote read probe from %s on %s failed: %s", holder_id, reader_id, exc)
                finally:
                    if file_id is not None:
                        delete_probe(holder, file_id)

            if remote_reads:
                remote_latency = sum(remote_reads) / len(remote_reads)
                latency_map[reader_id][holder_id] = max(0.0, remote_latency - local_latency)

    logger.info("Pairwise read latency (ms, reader -> holder): %s", latency_map)
    return latency_map


def check_all_providers_latency_with_names(
    managers: Dict[str, StorageInterface],
    devnull_id: Optional[str] = None,
    rounds: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Check latency for multiple providers and return provider name -> average RTT.

    Providers that fail probing are returned with value None.
    """
    latency_map: Dict[str, Optional[float]] = {}
    for provider_name, manager in managers.items():
        avg_rtt = check_provider_latency(manager, rounds=rounds, devnull_id=devnull_id)
        latency_map[provider_name] = avg_rtt
    return latency_map


def check_all_providers_latency_with_ids(
    zone_manager: StorageInterface,
    managers: Dict[str, StorageInterface],
    devnull_id: Optional[str] = None,
    rounds: int = 3,
) -> Dict[str, Optional[float]]:
    """
    Check latency for multiple providers and return provider ID -> average RTT.

    Providers that fail probing are returned with value None.
    """
    latency_map: Dict[str, Optional[float]] = {}
    for provider_key, manager in managers.items():
        avg_rtt = check_provider_latency(manager, rounds=rounds, devnull_id=devnull_id)

        provider_id = provider_key
        try:
            provider_id = zone_manager.resolve_provider_id(provider_key)
        except (ValueError, RuntimeError, AttributeError):
            # Fallback: resolve from manager.provider_name when available.
            manager_name = getattr(manager, "provider_name", "")
            if manager_name:
                try:
                    provider_id = zone_manager.resolve_provider_id(manager_name)
                except (ValueError, RuntimeError, AttributeError):
                    logger.error(
                        "Could not resolve provider ID for %r; using key as-is.",
                        provider_key,
                    )
            else:
                logger.error(
                    "Could not resolve provider ID for %r; using key as-is.",
                    provider_key,
                )

        latency_map[provider_id] = avg_rtt
    return latency_map


def predict_closest_provider(
    managers: Dict[str, StorageInterface],
    devnull_id: Optional[str] = None,
    rounds: int = 3,
) -> Optional[str]:
    """Return the provider name with the lowest measured RTT."""
    latency_map = check_all_providers_latency_with_names(
        managers, devnull_id=devnull_id, rounds=rounds
    )
    valid = {name: rtt for name, rtt in latency_map.items() if rtt is not None}
    if not valid:
        return None
    return min(valid, key=valid.get)
