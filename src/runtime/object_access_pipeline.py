import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict
import requests
from requests.auth import HTTPBasicAuth


from src.dag_monitoring.monitor_db import OnedataPrometheusMonitor
from src.data_placement.IPlacement import IPlacement
from src.data_placement.interrupt_listener import InterruptListener
from src.data_placement.strategies.Kairos import Kairos
from src.database.db_interface import DBInterface, Dag
from src.database.init_database import DB_PATH as DEFAULT_DB_PATH
from src.database.sqlite3_manager import SQLite3Manager
from src.network_monitoring.network_config import NetworkConfig
from src.rest_api.airflow.airflowmanager import AirflowManager
from src.rest_api.onedata.onedatamanager import OnedataDataManager
from src.dag_prediction.task_predictor import TaskPrediction, TaskPredictor
from src.rest_api.storage_interface import StorageInterface

logger = logging.getLogger(__name__)

# Mount path of the persistent volume in deployment/worker_job.template
CLUSTER_DB_PATH = "/data/kairos.sqlite3"


@dataclass
class PipelineConfig:
    prometheus_host: str
    db_path: Path
    lookback_minutes: int
    include_duration: bool
    max_predictions: int


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot object access pipeline: setup, ingest monitoring data to SQLite, "
            "predict object accesses, print results, and exit."
        )
    )
    parser.add_argument(
        "--prometheus-host",
        type=str,
        default=None,
        help="Prometheus host URL (env: PROMETHEUS_HOST, required).",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=f"SQLite database path (env: DATABASE_URL, default: {CLUSTER_DB_PATH}).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Lookback minutes for Prometheus query (env: MONITOR_LOOKBACK, default: 5).",
    )
    parser.add_argument(
        "--no-duration",
        action="store_true",
        help="Disable duration enrichment from Prometheus duration metrics.",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=20,
        help="Maximum number of predictions to print.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def resolve_config(args: argparse.Namespace, prometheus_host: str = None) -> PipelineConfig:
    prometheus_host = prometheus_host or args.prometheus_host or os.getenv("PROMETHEUS_HOST")
    if not prometheus_host:
        raise ValueError("PROMETHEUS_HOST is required via --prometheus-host or env var")

    # --db-path overrides DATABASE_URL, which overrides the persistent volume path
    db_path = _resolve_db_path(args.db_path or os.getenv("DATABASE_URL") or CLUSTER_DB_PATH)
    lookback_minutes = args.lookback or int(os.getenv("MONITOR_LOOKBACK", "5"))

    return PipelineConfig(
        prometheus_host=prometheus_host,
        db_path=db_path,
        lookback_minutes=lookback_minutes,
        include_duration=not args.no_duration,
        max_predictions=args.max_predictions,
    )


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH

    resolved = str(db_path).strip()
    if not resolved:
        return DEFAULT_DB_PATH

    if resolved.startswith("sqlite:///"):
        resolved = resolved.removeprefix("sqlite:///")

    return Path(resolved)


def setup_database_manager(config: PipelineConfig, init_db: bool = True) -> SQLite3Manager:
    return SQLite3Manager(config.db_path, initialize_db=init_db)


def run_monitoring_ingestion(
    config: PipelineConfig,
    database_manager: DBInterface,
    data_manager: StorageInterface,
    allowed_dag_execution_ids: set | None = None,
) -> int:
    monitor = OnedataPrometheusMonitor(
        prometheus_host=config.prometheus_host,
        data_manager=data_manager,
        db_path=config.db_path,
        lookback_minutes=config.lookback_minutes,
        include_duration=config.include_duration,
        test_mode=False,
        database_manager=database_manager,
        allowed_dag_execution_ids=allowed_dag_execution_ids,
    )
    reads = monitor.get_onedata_read_operations()
    writes = monitor.get_onedata_write_operations()
    operations = reads + writes
    logger.info("Ingested %d object access operations into SQLite - from that are %d reads and %d writes", len(operations), len(reads), len(writes))
    return len(operations)


def run_prediction(manager: SQLite3Manager):
    task_predictor = TaskPredictor(db_manager=manager, strategy="split")
    predictions: list[TaskPrediction] = task_predictor.predict_all_tasks()
    logger.debug("Generated predictions for %d objects", len(predictions))
    return predictions


def print_predictions(predictions: list[TaskPrediction]) -> None:
    if not predictions:
        logger.debug("No object predictions available.")
        return
    logger.debug(predictions)


def _clean_setting(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip(",")


def _load_onedata_settings() -> dict:
    onezone_host = _clean_setting(os.getenv("ONEDATA_ONEZONE_HOST", ""))
    oneprovider_host = _clean_setting(os.getenv("ONEDATA_PROVIDER_HOST", ""))
    space_id = _clean_setting(os.getenv("ONEDATA_SPACE_ID", ""))
    provider_tokens = []

    provider_token = _clean_setting(os.getenv("ONEDATA_TOKEN_PROVIDER"))
    if provider_token:
        provider_tokens.append(provider_token)

    zone_token = _clean_setting(os.getenv("ONEDATA_TOKEN_ZONE"))

    devnull_space_id = _clean_setting(os.getenv("DEVNULL_SPACE_ID"))

    deduped_provider_tokens = list(
        dict.fromkeys(token for token in provider_tokens if token)
    )
    if zone_token and zone_token not in deduped_provider_tokens:
        deduped_provider_tokens.append(zone_token)

    if not zone_token and deduped_provider_tokens:
        zone_token = deduped_provider_tokens[0]

    return {
        "onezone_host": onezone_host,
        "oneprovider_host": oneprovider_host,
        "space_id": space_id,
        "zone_token": zone_token,
        "provider_tokens": deduped_provider_tokens,
        "devnull_space_id": devnull_space_id,
    }

def _load_airflow_settings() -> dict:
    airflow_username = _clean_setting(os.getenv("AIRFLOW_USERNAME"))
    airflow_password = _clean_setting(os.getenv("AIRFLOW_PASSWORD"))
    airflow_api_service = _clean_setting(os.getenv("AIRFLOW_API_SERVICE"))

    return {
        "airflow_username": airflow_username,
        "airflow_password": airflow_password,
        "airflow_api_service": airflow_api_service,
    }


def _build_onedata_managers() -> (
    tuple[OnedataDataManager, OnedataDataManager, Dict[str, OnedataDataManager]]
):
    settings = _load_onedata_settings()
    zone_token = settings["zone_token"]
    provider_tokens = settings["provider_tokens"]
    devnull_space_id = settings["devnull_space_id"]

    if not zone_token:
        raise ValueError(
            "Missing Onedata token. Set ONEDATA_TOKEN_ZONE/ONEDATA_TOKEN_PROVIDER "
            "in the environment."
        )

    if not provider_tokens:
        raise ValueError("No provider token available. Set ONEDATA_TOKEN_PROVIDER.")

    if not settings["onezone_host"]:
        raise ValueError("Missing Onezone host. Set ONEDATA_ONEZONE_HOST.")

    zone_cfg = OnedataDataManager.load_config(
        token=zone_token,
        onezone_host=settings["onezone_host"],
        oneprovider_host=settings["oneprovider_host"],
    )
    zone_manager = OnedataDataManager.build_storage_manager(
        config=zone_cfg,
        space_id=settings["space_id"],
    )


    try:
        providers = zone_manager.list_providers()
    except (requests.RequestException, RuntimeError, ValueError):
        logger.exception("Failed to list providers for one of the available tokens")
        providers = []
        return 1
    provider_managers: Dict[str, OnedataDataManager] = {}
    for token in provider_tokens:
        token_cfg = OnedataDataManager.load_config(
            token=token,
            onezone_host=settings["onezone_host"],
            oneprovider_host=settings["oneprovider_host"],
        )

        for provider in providers:
            manager_key = provider.name or provider.provider_id
            if manager_key in provider_managers:
                continue

            provider_managers[manager_key] = OnedataDataManager.build_storage_manager(
                config=token_cfg,
                provider_url=provider.provider_url or None,
                provider_name=provider.name,
                space_id=settings["space_id"],
            )

    if not provider_managers:
        raise RuntimeError(
            "Could not build provider managers from available Onedata tokens."
        )

    return zone_manager, provider_managers, devnull_space_id

def build_kairos_dags(airflow_username: str, airflow_password: str, airflow_api_service: str) -> list[Kairos.dag]:
    """Build Kairos DAGs from the DAG definitions currently registered in Airflow."""

    # dev_mode disables TLS verification, as the in-cluster Airflow API was always called with verify=False
    airflow_cfg = AirflowManager.load_config(
        api_url=airflow_api_service,
        username=airflow_username,
        password=airflow_password,
        dev_mode=True,
    )
    airflow_manager = AirflowManager.build_manager(airflow_cfg)

    all_dags = []

    for dag_info in airflow_manager.list_dags():
        dag_id = dag_info["dag_id"]

        # TODO This might need to be updated, check Interfaces repo ()
        #  DAG Python files on a shared PVC (/opt/airflow/dags), written by awpl-api
        try:
            tasks_data = airflow_manager.list_tasks(dag_id)
        except requests.RequestException:
            logger.warning(f"Skipping {dag_id}, could not fetch tasks.")
            continue

        nodes_dict = {}
        edges_list = []

        for task in tasks_data:
            task_id = task["task_id"]
            nodes_dict[task_id] = Kairos.dag_node(task_id=task_id)

        for task in tasks_data:
            from_node = nodes_dict[task["task_id"]]

            for downstream_id in task.get("downstream_task_ids", []):
                to_node = nodes_dict.get(downstream_id)
                if to_node:
                    edges_list.append(Kairos.dag_edge(from_node=from_node, to_node=to_node))

        new_dag = Kairos.dag(
            dag_id=dag_id,
            nodes=list(nodes_dict.values()),
            edges=edges_list
        )
        all_dags.append(new_dag)

    return all_dags

def run_placement_strategy(
    provider_managers: dict[str, StorageInterface],
    zone_manager: StorageInterface,
    onedata_manager: StorageInterface,
    db_manager: SQLite3Manager,
    devnull_id: str,
    optimization_level: str,
    horizon_time_seconds: float = 60.0,
    prometheus_host: str = None,
    monitor_lookback_minutes: int = 5,
    interrupt_listener: InterruptListener | None = None,
    provider_latency_matrix: Dict[str, Dict[str, float | None]] | None = None,
    provider_bandwidth_matrix: Dict[str, Dict[str, float]] | None = None,
    network_config: NetworkConfig | None = None,
) -> None:
    strategy = Kairos(
        provider_managers=provider_managers,
        zone_manager=zone_manager,
        onedata_manager=onedata_manager,
        db_manager=db_manager,
        devnull_id=devnull_id,
        optimization_level=optimization_level,
        interrupt_listener=interrupt_listener,
        provider_latency_matrix=provider_latency_matrix,
        provider_bandwidth_matrix=provider_bandwidth_matrix,
        network_config=network_config,
    )
    logger.info("Starting Kairos placement strategy with optimization level: %s", optimization_level)

    try:
        settings = _load_airflow_settings()
        dags = build_kairos_dags(settings["airflow_username"], settings["airflow_password"], settings["airflow_api_service"])
        logger.info("Successfully built %d DAGs from Airflow.", len(dags))
        logger.debug("DAGs details: \n%s", dags)
    except Exception as e:
        logger.exception("Failed to build Kairos DAGs: %s", e)
        return

    def apply_placement(dag):
        logger.info("Applying Kairos placement strategy for DAG: %s", dag.dag_id)
        try:
            strategy.apply_placement(
                workflow_dag=dag,
                horizon_cost_budget=float("inf"),
                horizon_time_seconds=horizon_time_seconds,
                number_of_least_popular_to_evict=1,
                optimization_level=optimization_level
            )
            logger.info("Successfully applied Kairos placement for DAG: %s", dag.dag_id)
            return True
        except Exception as e:
            logger.exception("Failed to apply Kairos placement for DAG %s: %s", dag.dag_id, e)
            return False

    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=len(dags) or 1) as executor:
        futures = [executor.submit(apply_placement, dag) for dag in dags]
        for future in as_completed(futures):
            if future.result():
                succeeded += 1
            else:
                failed += 1

    logger.info(
        "Kairos placement complete — %d DAG(s) succeeded, %d DAG(s) failed.",
        succeeded, failed
    )