import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data_placement.interrupt_listener import InterruptListener
from src.data_placement.strategies.Kairos import Kairos


class FakeSession:
    """Zenoh session stand-in: ``put`` calls the subscriber like Zenoh does."""

    def __init__(self):
        self.callback = None
        self.closed = False

    def declare_subscriber(self, key_expr, callback):
        self.callback = callback
        return MagicMock()

    def put(self, message: dict):
        payload = json.dumps(message).encode("utf-8")
        self.callback(SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda: payload)))

    def close(self):
        self.closed = True


def _wait_for(condition, timeout=2.0):
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        time.sleep(0.01)
    return condition()


@pytest.mark.unit
def test_interrupts_go_to_the_latest_handler_of_their_dag():
    session = FakeSession()
    received = []
    with InterruptListener(session_factory=lambda: session) as listener:
        session.put({"dag_id": "dag_1", "task_id": "before_any_cycle"})
        time.sleep(0.05)
        listener.set_handler("dag_1", lambda data: received.append(("cycle 1", data["task_id"])))
        session.put({"dag_id": "dag_1", "task_id": "t1"})
        session.put({"dag_id": "dag_2", "task_id": "other_dag"})
        assert _wait_for(lambda: len(received) == 1)
        listener.set_handler("dag_1", lambda data: received.append(("cycle 2", data["task_id"])))
        session.put({"dag_id": "dag_1", "task_id": "t2"})
        assert _wait_for(lambda: len(received) == 2)
    assert received == [("cycle 1", "t1"), ("cycle 2", "t2")]
    assert session.closed


@pytest.mark.unit
def test_slow_or_failing_handler_does_not_block_the_subscriber():
    session = FakeSession()
    release = threading.Event()
    handled = []

    def handler(data):
        if data["task_id"] == "slow":
            release.wait()
        if data["task_id"] == "boom":
            raise RuntimeError("boom")
        handled.append(data["task_id"])

    with InterruptListener(session_factory=lambda: session) as listener:
        listener.set_handler("dag_1", handler)
        session.put({"dag_id": "dag_1", "task_id": "slow"})  # returns at once
        session.put({"dag_id": "dag_1", "task_id": "boom"})
        session.put({"dag_id": "dag_1", "task_id": "after"})
        listener.dispatch(b"not json")
        release.set()
        assert _wait_for(lambda: handled == ["slow", "after"])


@pytest.mark.unit
def test_kairos_keeps_serving_interrupts_after_its_horizon():
    listener = MagicMock()
    kairos = Kairos.__new__(Kairos)  # skip __init__: it measures provider latencies
    kairos.interrupt_listener = listener
    kairos.horizon_end_time = time.time()
    kairos._on_interrupt = MagicMock()

    kairos.dynamic_orchestration(Kairos.dag("dag_1", [], []), 0, {"p": 1}, {"f": 2})

    dag_id, handler = listener.set_handler.call_args.args
    assert dag_id == "dag_1"
    handler({"task_id": "t1"})  # e.g. while the next cycle is still ingesting
    kairos._on_interrupt.assert_called_once_with(
        {"task_id": "t1"}, {"p": 1}, {"f": 2}, 0, {"remaining": float("inf")}
    )
