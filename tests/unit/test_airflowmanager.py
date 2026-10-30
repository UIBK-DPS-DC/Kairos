from unittest.mock import MagicMock, patch

import pytest
import requests

from src.rest_api.airflow.airflow_helpers import AirflowConfig, load_config
from src.rest_api.airflow.airflowmanager import AirflowManager

API_URL = "http://mock-airflow:8080"


def _response(json_body=None, status_code=200):
    response = MagicMock()
    response.ok = status_code < 400
    response.status_code = status_code
    response.text = str(json_body)
    response.json.return_value = json_body or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    return response


@pytest.fixture
def manager():
    """Manager with a pre-issued token, so no auth request is made."""
    return AirflowManager.build_manager(AirflowConfig(api_url=API_URL, token="fake_token"))


@pytest.fixture
def credential_manager():
    """Manager that has to request a token with username/password first."""
    return AirflowManager.build_manager(
        AirflowManager.load_config(api_url=API_URL, username="admin", password="secret")
    )


@pytest.mark.unit
class TestLoadConfig:
    def test_strips_trailing_slash(self):
        assert load_config(f"{API_URL}/", token="t").api_url == API_URL

    def test_dev_mode_disables_ssl_verification(self):
        assert load_config(API_URL, token="t", dev_mode=True).verify_ssl is False

    def test_requires_url(self):
        with pytest.raises(ValueError, match="URL"):
            load_config("", token="t")

    def test_requires_token_or_credentials(self):
        with pytest.raises(ValueError, match="token or username and password"):
            load_config(API_URL, username="admin")


@pytest.mark.unit
class TestAuthentication:
    def test_token_from_config_is_used(self, manager):
        assert manager._session.headers["Authorization"] == "Bearer fake_token"

    def test_authenticates_lazily_on_first_call(self, credential_manager):
        session = credential_manager._session
        with patch.object(session, "post", return_value=_response({"access_token": "abc"})) as post, \
                patch.object(session, "get", return_value=_response({"dags": []})):
            credential_manager.list_dags()
            credential_manager.list_dags()

        post.assert_called_once()
        assert post.call_args.args[0] == f"{API_URL}/auth/token"
        assert post.call_args.kwargs["json"] == {"username": "admin", "password": "secret"}
        assert session.headers["Authorization"] == "Bearer abc"

    def test_failed_authentication_raises(self, credential_manager):
        with patch.object(credential_manager._session, "post", return_value=_response({}, 401)):
            with pytest.raises(RuntimeError, match="Failed to authenticate"):
                credential_manager.list_dags()


@pytest.mark.unit
class TestDagDiscovery:
    def test_list_dags(self, manager):
        with patch.object(manager._session, "get", return_value=_response({"dags": [{"dag_id": "a"}]})) as get:
            assert manager.list_dags() == [{"dag_id": "a"}]
        assert get.call_args.args[0] == f"{API_URL}/api/v2/dags"

    def test_list_tasks(self, manager):
        tasks = [{"task_id": "t1", "downstream_task_ids": ["t2"]}]
        with patch.object(manager._session, "get", return_value=_response({"tasks": tasks})) as get:
            assert manager.list_tasks("my_dag") == tasks
        assert get.call_args.args[0] == f"{API_URL}/api/v2/dags/my_dag/tasks"


@pytest.mark.unit
class TestDagRuns:
    def test_trigger_dag(self, manager):
        with patch.object(manager._session, "post", return_value=_response({"dag_run_id": "run_1"})) as post:
            assert manager.trigger_dag("my_dag") == "run_1"
        assert post.call_args.args[0] == f"{API_URL}/api/v2/dags/my_dag/dagRuns"
        assert post.call_args.kwargs["json"] == {"logical_date": None}

    @patch("src.rest_api.airflow.airflowmanager.time.sleep")
    def test_wait_for_completion_polls_until_terminal(self, _sleep, manager):
        states = [_response({"state": s}) for s in ("queued", "running", "success")]
        with patch.object(manager._session, "get", side_effect=states):
            assert manager.wait_for_completion("my_dag", "run_1", poll_interval=0) == "success"

    @patch("src.rest_api.airflow.airflowmanager.time.sleep")
    def test_wait_for_completion_times_out(self, _sleep, manager):
        with patch.object(manager._session, "get", return_value=_response({"state": "running"})):
            with pytest.raises(TimeoutError):
                manager.wait_for_completion("my_dag", "run_1", poll_interval=0, timeout=0)

    @patch("src.rest_api.airflow.airflowmanager.time.sleep")
    def test_trigger_with_retries_retries_failed_runs(self, _sleep, manager):
        with patch.object(manager, "trigger_dag", side_effect=["run_1", "run_2"]), \
                patch.object(manager, "wait_for_completion", side_effect=["failed", "success"]):
            assert manager.trigger_with_retries("my_dag", max_retries=3) is True

    @patch("src.rest_api.airflow.airflowmanager.time.sleep")
    def test_trigger_with_retries_gives_up(self, _sleep, manager):
        with patch.object(manager, "trigger_dag", side_effect=requests.ConnectionError("down")):
            assert manager.trigger_with_retries("my_dag", max_retries=2) is False

    def test_list_dag_runs_returns_empty_on_error(self, manager):
        with patch.object(manager._session, "get", return_value=_response({}, 500)):
            assert manager.list_dag_runs("my_dag") == []

    def test_get_task_instances(self, manager):
        instances = [{"task_id": "t1", "state": "success"}]
        with patch.object(manager._session, "get", return_value=_response({"task_instances": instances})) as get:
            assert manager.get_task_instances("my_dag", "run_1") == instances
        assert get.call_args.args[0] == f"{API_URL}/api/v2/dags/my_dag/dagRuns/run_1/taskInstances"
