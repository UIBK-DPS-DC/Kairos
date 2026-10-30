"""Smoke tests for AirflowManager against a real Airflow API."""
import os

import pytest
from dotenv import load_dotenv

from src.rest_api.airflow.airflowmanager import AirflowManager

# Load .env file automatically
load_dotenv()


@pytest.fixture(scope="session", name="airflow_manager")
def fixture_airflow_manager() -> AirflowManager:
    """Create a real Airflow manager from .env config."""
    api_url = os.getenv("AIRFLOW_API_SERVICE")
    username = os.getenv("AIRFLOW_USERNAME")
    password = os.getenv("AIRFLOW_PASSWORD")
    if not all([api_url, username, password]):
        pytest.skip("AIRFLOW_API_SERVICE, AIRFLOW_USERNAME or AIRFLOW_PASSWORD is not configured")

    config = AirflowManager.load_config(api_url=api_url, username=username, password=password)
    return AirflowManager.build_manager(config)


@pytest.mark.smoke
@pytest.mark.integration
def test_airflow_authentication(airflow_manager: AirflowManager):
    """Credentials are accepted and yield a non-empty access token."""
    assert airflow_manager.authenticate()


@pytest.mark.smoke
@pytest.mark.integration
def test_airflow_list_dags_and_tasks(airflow_manager: AirflowManager):
    """DAGs and their tasks can be listed with the expected fields."""
    dags = airflow_manager.list_dags()
    assert isinstance(dags, list)
    if not dags:
        pytest.skip("No DAGs registered in Airflow")

    tasks = airflow_manager.list_tasks(dags[0]["dag_id"])
    assert all("task_id" in task for task in tasks)
