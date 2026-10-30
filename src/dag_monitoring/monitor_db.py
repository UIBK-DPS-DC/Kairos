# onedata_prometheus_monitor_container.py
# !/usr/bin/env python3
"""
Container-friendly Onedata Prometheus Monitor.

Fetches Onedata object access metrics from Prometheus and persists them in the sqlite3 database.

Supports environment variables for all configuration:
    - PROMETHEUS_HOST: Prometheus server URL (required)
    - DATABASE_URL: SQLite database path or sqlite:/// URL (default: data/kairos.sqlite3)
    - MONITOR_INTERVAL: Query interval in seconds (default: 60)
    - MONITOR_LOOKBACK: Lookback minutes for queries (default: 5)
    - LOG_LEVEL: Logging level (default: INFO)
"""

import argparse
import hashlib
import json
import logging
import os
import requests
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin
from src.database.init_database import initialize_database
from src.database.sqlite3_manager import SQLite3Manager
import re
from src.database.db_interface import Access, AccessType
import src.database.db_interface as db
from src.rest_api.storage_interface import StorageInterface

logger = logging.getLogger(__name__)
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "kairos.sqlite3"

# Global flag for graceful shutdown
_shutdown_requested = False


def configure_logging(log_level: str) -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )


def _sigterm_handler(_signum, _frame):
    """Handle SIGTERM for graceful shutdown in Kubernetes."""
    global _shutdown_requested  # pylint: disable=global-statement
    _shutdown_requested = True
    logger.warning("SIGTERM received; requesting graceful shutdown")


def setup_signal_handlers():
    """Register signal handlers for graceful termination."""
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)


def is_shutdown_requested() -> bool:
    """Check if graceful shutdown was requested."""
    return _shutdown_requested


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH

    resolved = str(db_path).strip()
    if not resolved:
        return DEFAULT_DB_PATH

    if resolved.startswith("sqlite:///"):
        resolved = resolved.removeprefix("sqlite:///")

    return Path(resolved)


def _build_access_id(record: Dict) -> str:
    """Stable id of one access: the labels of its Prometheus series.

    Every Prometheus series is one access of one task execution, so the id must
    not contain the sample timestamp, which shifts with each query window and
    would store the same access again on every ingestion. Accesses without a
    session cannot be told apart by their labels, so their timestamp is kept.
    """
    parts = [
        str(record.get("file_id", "unknown")),
        str(record.get("connection_id") or record.get("provider_id", "unknown")),
        str(record.get("space_id", "unknown")),
        str(record.get("operation", "read")),
        str(record.get("session_id", "unknown")),
    ]
    if record.get("task_execution_id") is None:
        parts.append(str(record.get("unix_timestamp", record.get("timestamp", "unknown"))))
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _parse_series(data: Dict, access_type: db.AccessType) -> List[Dict]:
    """One access record per Prometheus series of a ``*_duration_seconds_sum`` query.

    The timestamp is the series' first sample in the query window and the
    duration its last value (the counter's total).
    """
    results = []
    if data.get("resultType") not in ["matrix", "vector"]:
        return results
    for res in data.get("result", []):
        metric = res.get("metric", {})
        values = res.get("values", res.get("value", []))
        if data.get("resultType") == "vector" and values:
            values = [values]
        if not values:
            continue
        first_timestamp = min(float(ts) for ts, _ in values)
        _, last_value = max(values, key=lambda sample: float(sample[0]))

        session_id = metric.get("session_id", "unknown")
        session_id_parts = session_id.split(":")
        if len(session_id_parts) < 4:
            logger.warning(
                "session_id of access has not the wanted format so it might be a random access without a dag/task. It should look like 'dag_id:dag_execution_id:task_id:task_execution_id': %s",
                session_id,
            )
            dag_id = dag_execution_id = task_id = task_execution_id = None
        else:
            dag_id, dag_execution_id, task_id, task_execution_id = session_id_parts[:4]

        try:
            duration_seconds = float(last_value)
        except (TypeError, ValueError):
            duration_seconds = None

        results.append(
            {
                "timestamp": datetime.fromtimestamp(first_timestamp).isoformat(),
                "provider_id": metric.get("provider_id", "unknown"),
                "space_id": metric.get("space_id", "unknown"),
                "file_id": metric.get("file_id", "unknown"),
                "session_id": session_id,
                "duration_seconds": duration_seconds,
                "dag_id": dag_id,
                "dag_execution_id": dag_execution_id,
                "task_id": task_id,
                "task_execution_id": task_execution_id,
                "access_type": access_type,
                "operation": access_type.value,
            }
        )
    return results


def _insert_unique_new_helper_values(
    manager: db.DBInterface, accesses: List[Dict]
) -> bool:
    """Insert unique new values for DAGs, Tasks, Executions, Files, Providers, and Spaces."""

    dags: set[db.Dag] = set()
    tasks: set[db.Task] = set()
    dag_executions: set[db.Dag_Execution] = set()
    task_executions: set[db.Task_Execution] = set()
    files: set[db.File] = set()
    providers: set[db.Provider] = set()
    spaces: set[db.Space] = set()

    for access in accesses:
        dag_id = access.get("dag_id")
        task_id = access.get("task_id")
        dag_execution_id = access.get("dag_execution_id")
        task_execution_id = access.get("task_execution_id")
        # timepoint = access.get("timestamp", access.get("unix_timestamp"))
        file_id = access.get("file_id")
        provider_id = access.get("provider_id")
        space_id = access.get("space_id")

        if dag_id:
            dags.add(db.Dag(dag_id))
        if task_id and dag_id:
            tasks.add(db.Task(task_id, dag_id))
        if dag_execution_id and dag_id:
            dag_executions.add(db.Dag_Execution(dag_execution_id, dag_id))
        if task_execution_id and task_id:
            task_executions.add(
                db.Task_Execution(task_execution_id, task_id, dag_id, dag_execution_id)
            )
        if file_id:
            files.add(db.File(file_id))
        if provider_id:
            providers.add(db.Provider(provider_id))
        if space_id:
            spaces.add(db.Space(space_id))

    # Insert unique DAGs
    for dag in dags:
        manager.insert_dag(dag)

    # Insert unique Tasks
    for task in tasks:
        manager.insert_task(task)

    # Insert unique Executions
    for dag_execution in dag_executions:
        manager.insert_dag_execution(dag_execution)

    for task_execution in task_executions:
        manager.insert_task_execution(task_execution)

    # Insert unique Files
    for file in files:
        manager.insert_file(file)

    # Insert unique Providers
    for provider in providers:
        manager.insert_provider(provider)

    # Insert unique Spaces
    for space in spaces:
        manager.insert_space(space)

def restore_onedata_path_with_context(path: str, dag_id: str):
    trash_pattern = re.compile(r"^\/([^\/]+)\/\.trash\/([^@]+)@@[^\/]+\/(.*)$")
    
    match = trash_pattern.match(path)
    if match:
        one = match.group(1)
        three = match.group(2)
        four = match.group(3)

        clean_dag_id = "_".join(dag_id.split("_")[:2])
        
        # Reconstruct: /ProductionSpace + / + image_classification + / + grayscaled + /input/...
        return f"/{one}/{clean_dag_id}/{three}/{four}"

    return path # Return as-is if it's not a .trash path

    
def _persist_accesses(
    results: List[Dict],
    db_manager: db.DBInterface,
    data_manager: StorageInterface,
    allowed_dag_execution_ids: set | None = None,
) -> int:
    """Store read access events in sqlite and return the number written.

    If *allowed_dag_execution_ids* is provided, only accesses whose
    ``dag_execution_id`` is in that set are persisted.  Accesses without a
    ``dag_execution_id`` (i.e. not associated with any tracked DAG run) are
    always dropped when a filter is active.
    """
    if not results:
        return 0

    if allowed_dag_execution_ids is not None:
        results = [
            a for a in results
            if a.get("dag_execution_id") in allowed_dag_execution_ids
        ]
        if not results:
            logger.debug(
                "_persist_accesses: all accesses filtered out (none match current run IDs)"
            )
            return 0

    accesses = []

    # first always try to insert new values for dags, tasks, dag_executions, task_executions, files, providers, spaces
    # so find unique values for each of these and insert or ignore them first
    _insert_unique_new_helper_values(db_manager, results)

    for access in results:
        file_id = str(access.get("file_id", "")).strip()
        provider_id = str(
            access.get("connection_id") or access.get("provider_id", "")
        ).strip()
        space_id = str(access.get("space_id", "")).strip()
        timestamp = str(access.get("timestamp", "")).strip()
        duration_seconds = access.get("duration_seconds")
        task_execution_id = access.get("task_execution_id")
        dag_id = access.get("dag_id")

        if (
            not file_id
            or not provider_id
            or not space_id
            or not timestamp
        ):
            continue

        access_id = _build_access_id(access)
        known = db_manager.read_access(access_id)
        if known is not None:
            # Already stored by an earlier ingestion: no path lookup needed.
            cleaned_file_path = known.file_path
        else:
            # lookup the path of the file and store it in the database
            file_path = data_manager.get_file_path(file_id)
            if file_path is None:
                logger.debug("Skipping access for file %s: path lookup returned None", file_id)
                continue
            # clean the path to restore the original path if it was in trash
            cleaned_file_path = restore_onedata_path_with_context(file_path, dag_id)

        raw_access_type = access.get("access_type") or access.get("operation") or db.AccessType.READ
        if isinstance(raw_access_type, db.AccessType):
            access_type = raw_access_type
        else:
            try:
                access_type = db.AccessType(str(raw_access_type).lower())
            except ValueError:
                access_type = db.AccessType.READ

        accesses.append(
            db.Access(
                id=access_id,
                file_id=file_id,
                file_path=cleaned_file_path,
                provider_id=provider_id,
                space_id=space_id,
                timepoint=timestamp,
                access_type=access_type,
                duration_seconds=duration_seconds,
                task_execution_id=task_execution_id,
            )
        )

    return db_manager.insert_accesses(accesses)


class OnedataPrometheusMonitor:
    """Container-friendly Prometheus monitor for object access caching."""

    def __init__(
        self,
        prometheus_host: str,
        data_manager: StorageInterface,
        db_path: str | Path | None = None,
        database_manager: db.DBInterface | None = None,
        query_interval: int = 60,
        lookback_minutes: int = 5,
        include_duration: bool = True,
        test_mode: bool = False,
        allowed_dag_execution_ids: set | None = None,
    ):
        self.prometheus_host = prometheus_host.rstrip("/")
        self.data_manager = data_manager
        self.db_path = _resolve_db_path(db_path)
        self.query_interval = query_interval
        self.lookback_minutes = lookback_minutes
        self.include_duration = include_duration
        self.test_mode = test_mode
        self.api_url = urljoin(f"{self.prometheus_host}/", "api/v1/")
        self.database_manager = database_manager
        self.allowed_dag_execution_ids = allowed_dag_execution_ids

    # --------------------------
    # Internal Prometheus query
    # --------------------------
    def _query_prometheus(
        self, query: str, start_time: datetime = None, end_time: datetime = None
    ) -> Dict:
        try:
            if start_time and end_time:
                url = urljoin(self.api_url, "query_range")
                params = {
                    "query": query,
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                    "step": "15s",
                }
            else:
                url = urljoin(self.api_url, "query")
                params = {"query": query}

            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                logger.error("Prometheus query failed: %s", data)
                return {}
            return data.get("data", {})
        except Exception as err:  # pylint: disable=broad-except
            logger.error("Error querying Prometheus: %s", err)
            return {}

    # --------------------------
    # Fetch operations and cache
    # --------------------------
    def get_onedata_read_operations(self) -> List[Dict]:
        query = "onedata_read_duration_seconds_sum"
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=self.lookback_minutes)
        data = self._query_prometheus(query, start_time, end_time)
        results = _parse_series(data, db.AccessType.READ)

        written = _persist_accesses(results, self.database_manager, self.data_manager, self.allowed_dag_execution_ids)

        logger.info("Stored %d accesses in SQLite", written)

        logger.info(
            "Retrieved %d operations%s",
            len(results),
            " and cached them" if not self.test_mode else "",
        )
        return results
    
    def get_onedata_write_operations(self) -> List[Dict]:
        query = "onedata_write_duration_seconds_sum"
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=self.lookback_minutes)
        data = self._query_prometheus(query, start_time, end_time)
        results = _parse_series(data, db.AccessType.WRITE)

        written = _persist_accesses(results, self.database_manager, self.data_manager, self.allowed_dag_execution_ids)

        logger.info("Stored %d accesses in SQLite", written)

        logger.info(
            "Retrieved %d operations%s",
            len(results),
            " and cached them" if not self.test_mode else "",
        )
        return results

    def get_latest_operations(self) -> List[Dict]:
        """
        Fetch the latest operations using a short lookback (query_interval in seconds).
        Respects test_mode: does not write to database if test_mode=True.
        """
        original_lookback = self.lookback_minutes
        self.lookback_minutes = max(1, self.query_interval // 60)
        results = self.get_onedata_read_operations()
        results += self.get_onedata_write_operations()
        self.lookback_minutes = original_lookback
        return results


# --------------------------
# Example callback for container-friendly use
# --------------------------
def example_callback(ops: List[Dict]):
    """Print operations for debugging."""
    logger.info("Processing %d new operations", len(ops))
    for op in ops[:5]:
        logger.info(
            "FILE=%s OP=%s DURATION=%s",
            op.get("file_id"),
            op.get("operation"),
            op.get("duration_seconds"),
        )

