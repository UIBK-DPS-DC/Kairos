"""High-level Airflow workflow manager.

This module contains runtime API logic for authentication, DAG discovery,
and DAG run lifecycle methods against the Airflow v2 REST API.

Shared models and config helpers are defined in `airflow_helpers.py`.
"""

import logging
import time
from typing import Optional

import requests

from src.rest_api.airflow.airflow_helpers import (
    AirflowConfig,
    DagRunState,
    load_config as aux_load_config,
)

logger = logging.getLogger(__name__)


class AirflowManager:
    """
    Workflow manager for Apache Airflow.

    Parameters
    ----------
    config : AirflowConfig
        API URL plus either a pre-issued token or username/password. With
        credentials, a token is requested lazily on the first API call.
    """

    API_BASE = "/api/v2"

    @staticmethod
    def load_config(
        api_url: str,
        username: str = "",
        password: str = "",
        token: str = "",
        dev_mode: bool = False,
    ) -> AirflowConfig:
        """Create an AirflowConfig from API URL and credential values."""
        return aux_load_config(
            api_url=api_url,
            username=username,
            password=password,
            token=token,
            dev_mode=dev_mode,
        )

    @classmethod
    def build_manager(cls, config: AirflowConfig) -> "AirflowManager":
        """Create an Airflow manager from config."""
        return cls(config=config)

    def __init__(self, config: AirflowConfig):
        self.config = config
        self._session = self._build_session()
        if config.token:
            self._set_token(config.token)

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> str:
        """Request an access token with username/password and attach it to the session."""
        r = self._session.post(
            f"{self.config.api_url}/auth/token",
            json={"username": self.config.username, "password": self.config.password},
            timeout=(10, 30),
        )
        if not r.ok:
            raise RuntimeError(
                f"Failed to authenticate with Airflow ({r.status_code}): {r.text}"
            )
        token = r.json().get("access_token")
        if not token:
            raise RuntimeError("Airflow auth response contained no access_token.")
        self._set_token(token)
        logger.info("Successfully authenticated with Airflow.")
        return token

    def _ensure_authenticated(self) -> None:
        if "Authorization" not in self._session.headers:
            self.authenticate()

    def _set_token(self, token: str) -> None:
        self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # DAG discovery
    # ------------------------------------------------------------------

    def list_dags(self) -> list[dict]:
        """Return all DAGs visible to the current credentials."""
        return self._raw_get(self._api_url("/dags")).get("dags", [])

    def list_tasks(self, dag_id: str) -> list[dict]:
        """Return the tasks of a DAG, each with ``task_id`` and ``downstream_task_ids``."""
        return self._raw_get(self._api_url(f"/dags/{dag_id}/tasks")).get("tasks", [])

    # ------------------------------------------------------------------
    # DAG runs
    # ------------------------------------------------------------------

    def trigger_dag(self, dag_id: str) -> str:
        """Trigger a DAG run and return its dag_run_id."""
        # Python's None is sent as JSON null, i.e. "run now"
        body = self._raw_post(self._api_url(f"/dags/{dag_id}/dagRuns"), {"logical_date": None})
        return body.get("dag_run_id")

    def get_dag_run_state(self, dag_id: str, dag_run_id: str) -> str:
        """Return the current state of a DAG run."""
        url = self._api_url(f"/dags/{dag_id}/dagRuns/{dag_run_id}")
        return self._raw_get(url).get("state")

    def wait_for_completion(
        self,
        dag_id: str,
        dag_run_id: str,
        poll_interval: float = 10,
        timeout: Optional[float] = None,
    ) -> str:
        """Poll a DAG run until it reaches a terminal state and return that state.

        Raises TimeoutError if ``timeout`` seconds pass first.
        """
        logger.info("Monitoring DAG run: %s", dag_run_id)
        deadline = time.monotonic() + timeout if timeout is not None else None

        while True:
            state = self.get_dag_run_state(dag_id, dag_run_id)
            logger.debug("DAG run %s state: %s", dag_run_id, state)
            if state in (DagRunState.SUCCESS, DagRunState.FAILED):
                return state
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"DAG run '{dag_run_id}' did not finish within {timeout}s")
            time.sleep(poll_interval)

    def trigger_with_retries(
        self,
        dag_id: str,
        max_retries: int = 3,
        poll_interval: float = 10,
        retry_delay: float = 5,
    ) -> bool:
        """Trigger a DAG, wait for it, and retry on failure. Return True on success."""
        for attempt in range(1, max_retries + 1):
            logger.info("Triggering '%s' (attempt %d/%d)", dag_id, attempt, max_retries)
            try:
                dag_run_id = self.trigger_dag(dag_id)
                logger.info("Triggered DAG '%s', run ID: %s", dag_id, dag_run_id)

                if self.wait_for_completion(dag_id, dag_run_id, poll_interval) == DagRunState.SUCCESS:
                    logger.info("DAG '%s' completed successfully.", dag_id)
                    return True
                logger.warning("DAG run '%s' failed.", dag_run_id)

            except requests.RequestException as exc:
                body = exc.response.text if exc.response is not None else ""
                logger.warning("Airflow request failed: %s %s", exc, body)

            if attempt < max_retries:
                time.sleep(retry_delay)

        logger.error("DAG '%s' failed after %d attempts.", dag_id, max_retries)
        return False

    def list_dag_runs(self, dag_id: str, state: str = "success", limit: int = 100) -> list[dict]:
        """Return DAG runs filtered by state.

        Each dict contains at minimum: dag_run_id, state, start_date, end_date,
        logical_date. Returns an empty list on any HTTP error.
        """
        try:
            body = self._raw_get(
                self._api_url(f"/dags/{dag_id}/dagRuns"),
                params={"state": state, "limit": limit},
            )
        except requests.RequestException as exc:
            logger.warning("list_dag_runs failed for dag '%s': %s", dag_id, exc)
            return []
        return body.get("dag_runs", [])

    def get_task_instances(self, dag_id: str, dag_run_id: str) -> list[dict]:
        """Return all task instances for a DAG run.

        Each dict contains: task_id, state, start_date, end_date, try_number.
        Returns an empty list on any HTTP error.
        """
        try:
            body = self._raw_get(
                self._api_url(f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
            )
        except requests.RequestException as exc:
            logger.warning("get_task_instances failed for run '%s': %s", dag_run_id, exc)
            return []
        return body.get("task_instances", [])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _api_url(self, path: str) -> str:
        return f"{self.config.api_url}{self.API_BASE}{path}"

    def _raw_get(self, url: str, params: Optional[dict] = None) -> dict:
        self._ensure_authenticated()
        r = self._session.get(url, params=params, timeout=(10, 30))
        r.raise_for_status()
        return r.json()

    def _raw_post(self, url: str, body: Optional[dict] = None) -> dict:
        self._ensure_authenticated()
        r = self._session.post(url, json=body or {}, timeout=(10, 30))
        r.raise_for_status()
        return r.json()

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json"})
        s.verify = self.config.verify_ssl
        return s
