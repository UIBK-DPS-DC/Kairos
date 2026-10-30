import time
import os
from typing import Mapping, Optional
from src.rest_api.onedata.onedata_helpers import Transfer
from src.rest_api.storage_interface import StorageInterface
import logging

logger = logging.getLogger(__name__)

PROBE_FILENAME = "bw_probe_100b.bin"
PROBE_SIZE_BYTES = 1 * 100  # 100 Byte


class TransferNotCompletedException(Exception):
    pass


def _create_probe_file(path: str):
    """Write 1 MB of deterministic bytes to a local temp file."""
    with open(path, "wb") as f:
        f.write(os.urandom(PROBE_SIZE_BYTES))


def extract_transfer_time_seconds(transfer: Transfer) -> Optional[float]:

    # Onedata reports startTime/finishTime as Unix timestamps in whole seconds.
    difference_seconds = (
        transfer.finish_time - transfer.start_time
        if (transfer.start_time is not None and transfer.finish_time is not None)
        else None
    )
    if difference_seconds is None:
        raise TransferNotCompletedException(
            f"Transfer {transfer.transfer_id} has not completed yet."
        )

    logger.info(
        f"Transfer {transfer.transfer_id} took {difference_seconds} seconds (start: {transfer.start_time}, finish: {transfer.finish_time})."
    )

    return float(difference_seconds)


def calculate_bandwidth_bytes_per_second(
    transfer: Transfer, file_size_bytes: int
) -> Optional[float]:
    transfer_time = extract_transfer_time_seconds(transfer)
    if transfer_time <= 0 or file_size_bytes <= 0:
        raise ValueError(
            f"Invalid transfer time ({transfer_time} seconds) or file size ({file_size_bytes} bytes) for transfer {transfer.transfer_id}."
        )
    return file_size_bytes / transfer_time


def _has_valid_timing(transfer: Transfer) -> bool:
    return (
        transfer.start_time is not None
        and transfer.finish_time is not None
        and transfer.start_time > 0
        and transfer.finish_time > transfer.start_time
    )


def mirror_array_across_hypothenuse(array: list[list[float]]) -> list[list[float]]:
    """Given a 2D array, mirror the values across the hypothenuse to make it symmetric."""
    for i, row in enumerate(array):
        for j in range(i + 1, len(row)):
            array[j][i] = array[i][j]
    return array


def check_network_bandwidth(
    managers: list[StorageInterface] | Mapping[str, StorageInterface], zone_manager
):
    """
    Upload a 1 MB probe file to the manager's space root and measure
    the average upload RTT across `rounds` requests.
    """
    local_path = f"/tmp/{PROBE_FILENAME}"
    _create_probe_file(local_path)

    with open(local_path, "rb") as f:
        content = f.read()

    # Accept both dict[str, StorageInterface] and list[StorageInterface].
    list_of_providers = zone_manager.list_providers()
    if isinstance(managers, Mapping):
        manager_items = list(managers.items())
        manager_list = [mgr for _, mgr in manager_items]
    else:
        manager_items = []
        manager_list = list(managers)

    provider_id_by_alias: dict[str, str] = {}
    for provider in list_of_providers:
        aliases = {
            provider.provider_id,
            provider.name,
            provider.domain,
            provider.domain.split(".", 1)[0] if provider.domain else "",
        }
        for alias in aliases:
            alias = alias.strip()
            if alias:
                provider_id_by_alias[alias] = provider.provider_id

    provider_ids: list[str] = []
    for idx, mgr in enumerate(manager_list):
        manager_name = getattr(mgr, "provider_name", "").strip()
        provider_id = provider_id_by_alias.get(manager_name)

        # If managers were passed as a mapping, try matching by mapping key alias too.
        if provider_id is None and manager_items:
            mapping_key = str(manager_items[idx][0]).strip()
            provider_id = provider_id_by_alias.get(mapping_key)

        # Last-resort fallback: keep positional alignment with provider discovery.
        if provider_id is None and idx < len(list_of_providers):
            provider_id = list_of_providers[idx].provider_id

        if provider_id is None:
            raise RuntimeError(
                f"Unable to resolve provider ID for manager at index {idx}."
            )
        provider_ids.append(provider_id)

    # uploading the file to one providers and then replicate it to all the other providers to have the bandwidth estimation between providers
    bandwith_array = [
        [float("inf") for _ in range(len(manager_list))]
        for _ in range(len(manager_list))
    ]
    print("created probe file")
    had_successful_upload = False

    for i, mgr in enumerate(manager_list):
        # Skip a redundant last-source pass when at least one prior source upload succeeded.
        if i >= len(manager_list) - 1 and had_successful_upload:
            continue

        transfers = []
        transfer_to_destination: dict[int, int] = {}
        transfer_start_wall_time: dict[int, float] = {}
        # upload to first provider
        try:
            print(
                f"-upload probe to provider, {i} round, with name {mgr.provider_name}"
            )
            file_id = mgr.upload_file(filename=PROBE_FILENAME, content=content)
            had_successful_upload = True
        except Exception as exc:
            print(f"Failed to upload probe file to {mgr.space_id}: {exc}")
            continue

        # replicate to other providers
        for j, other_mgr in enumerate(manager_list[i + 1 :], start=i + 1):
            try:
                other_provider_id = provider_ids[j]
                print(
                    f"--replicate probe to provider, {j} round, name {other_mgr.provider_name}"
                )
                transfer = mgr.replicate(
                    file_ref=file_id,
                    destination_provider_id=other_provider_id,
                    zone_manager=zone_manager,
                )
                transfers.append(transfer)
                transfer_to_destination[transfer.transfer_id] = j
                transfer_start_wall_time[transfer.transfer_id] = time.perf_counter()

            except Exception as exc:
                print(f"Failed to replicate probe file to {other_mgr.space_id}: {exc}")
                continue

        # wait for replication to complete with exponential backoff
        backoff_time = 0.1
        transfer_attempts: dict[str, int] = {t.transfer_id: 0 for t in transfers}
        max_attempts_per_transfer = 5
        while transfers:
            for transfer in transfers.copy():
                try:
                    print(f"---wait for transfer {transfer.transfer_id} to complete")
                    mgr.wait_for_transfer(transfer=transfer, timeout=10.0)

                    # when successfully replicated, we can calculate the bandwidth and remove the transfer id from the set
                    print(
                        f"----transfer {transfer.transfer_id} completed, calculating bandwidth"
                    )

                    # Some providers report completion before timestamps are fully
                    # materialized in transfer status. Poll briefly for finalized
                    # timing metadata before computing throughput.
                    metadata_retries = 10
                    metadata_retry_sleep = 0.2
                    while metadata_retries > 0 and not _has_valid_timing(transfer):
                        try:
                            refreshed = mgr.update_transfer_status(transfer)
                            if isinstance(refreshed, Transfer):
                                transfer = refreshed
                        except Exception as metadata_exc:
                            logger.debug(
                                "Transfer %s metadata refresh failed: %s",
                                transfer.transfer_id,
                                metadata_exc,
                            )
                            break
                        if _has_valid_timing(transfer):
                            break
                        metadata_retries -= 1
                        if metadata_retries > 0:
                            time.sleep(metadata_retry_sleep)

                    j = transfer_to_destination[transfer.transfer_id]
                    try:
                        bandwith_array[i][j] = calculate_bandwidth_bytes_per_second(
                            transfer, PROBE_SIZE_BYTES
                        )
                    except ValueError:
                        elapsed = (
                            time.perf_counter()
                            - transfer_start_wall_time[transfer.transfer_id]
                        )
                        elapsed = max(elapsed, 1e-6)
                        bandwith_array[i][j] = PROBE_SIZE_BYTES / elapsed
                        logger.warning(
                            "Falling back to local elapsed timing for transfer %s: %.3fs",
                            transfer.transfer_id,
                            elapsed,
                        )
                    transfers.remove(transfer)
                except Exception as exc:
                    transfer_attempts[transfer.transfer_id] += 1
                    print(
                        f"Error while waiting for transfer {transfer.transfer_id}: {exc}"
                    )
                    # Prevent endless retries when a transfer never becomes measurable.
                    if (
                        transfer_attempts[transfer.transfer_id]
                        >= max_attempts_per_transfer
                    ):
                        transfers.remove(transfer)
                    continue
            if not transfers:
                break
            backoff_time = backoff_time * 4
            time.sleep(backoff_time)

        # delete original file from first provider to avoid filling up the space
        try:
            print(f"----delete probe file from og provider, {i} round")
            mgr.delete_file(file_id)
        except Exception as exc:
            print(f"Failed to delete probe file from {mgr.space_id}: {exc}")
            continue

        # delete all probe files from providers
        for j, other_mgr in enumerate(manager_list[i + 1 :], start=i + 1):
            try:
                print(f"----delete probe file from provider, {j} round")
                other_mgr.delete_file(file_id)
            except Exception as exc:
                print(f"Failed to delete probe file from {other_mgr.space_id}: {exc}")
                continue

    # delete local probe file
    os.unlink(local_path)

    mirrored = mirror_array_across_hypothenuse(bandwith_array)
    return {
        target_provider_id: {
            source_provider_id: mirrored[target_index][source_index]
            for source_index, source_provider_id in enumerate(provider_ids)
        }
        for target_index, target_provider_id in enumerate(provider_ids)
    }


def get_bandwidth_between_providers(
    network_bandwith_map: dict[str, dict[str, float]],
    source_provider_id: str,
    target_provider_id: str,
) -> Optional[float]:
    if network_bandwith_map is None:
        raise ValueError("Network bandwidth map has not been initialized.")
    if target_provider_id not in network_bandwith_map:
        raise KeyError(f"Unknown target provider ID: {target_provider_id}")
    if source_provider_id not in network_bandwith_map[target_provider_id]:
        raise KeyError(f"Unknown source provider ID: {source_provider_id}")
    return network_bandwith_map[target_provider_id][source_provider_id]
