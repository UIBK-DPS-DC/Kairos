"""Shared support types and helpers for the Airflow integration.

This module hosts configuration helpers and DAG run models that are used
by `airflowmanager.py` and other external callers.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class AirflowConfig:
    api_url: str
    username: str = ""
    password: str = ""
    token: str = ""
    verify_ssl: bool = True


def load_config(
    api_url: str,
    username: str = "",
    password: str = "",
    token: str = "",
    dev_mode: bool = False,
) -> AirflowConfig:
    """Build an AirflowConfig from plain values.

    Either a pre-issued ``token`` or ``username``/``password`` is required;
    with credentials, the manager requests a token from ``/auth/token``.
    """
    if not api_url:
        raise ValueError("Airflow API URL is required.")
    if not token and not (username and password):
        raise ValueError("Airflow token or username and password are required.")
    return AirflowConfig(
        api_url=api_url.rstrip("/"),
        username=username,
        password=password,
        token=token,
        verify_ssl=not dev_mode,
    )


class DagRunState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
