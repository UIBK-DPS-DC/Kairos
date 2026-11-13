import json
import logging
import os
import queue
import threading
from typing import Any, Callable, Dict

import zenoh

logger = logging.getLogger(__name__)

INTERRUPT_KEY = "airflow_interrupts"
# Endpoint of a Zenoh router, e.g. "tcp/zenoh-router:7447".
CONNECT_ENV = "ZENOH_CONNECT"

InterruptHandler = Callable[[Dict[str, Any]], None]


def zenoh_config() -> zenoh.Config:
    """Client of the router in ZENOH_CONNECT, or else a peer that scouts by multicast.

    Multicast scouting does not cross pod boundaries in Kubernetes, so there
    the predictor and Airflow have to meet at a router.
    """
    config = zenoh.Config()
    endpoint = os.getenv(CONNECT_ENV, "").strip()
    if endpoint:
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([endpoint]))
    return config


class InterruptListener:
    """Receives Airflow task-completion interrupts for the whole lifetime of the predictor.

    A subscriber that only exists during a placement cycle misses every
    interrupt sent while ingestion and static planning run. This listener stays
    subscribed across cycles; each cycle registers its handler per DAG with
    :meth:`set_handler`, which replaces the handler of the previous cycle.
    Until then the previous cycle's handler keeps serving its DAG.

    Interrupts are handled one at a time on a worker thread, so a slow handler
    (e.g. waiting for a file to appear) never blocks the Zenoh callback.
    """

    def __init__(self, key_expr: str = INTERRUPT_KEY, session_factory=None):
        self._key_expr = key_expr
        self._session_factory = session_factory or (lambda: zenoh.open(zenoh_config()))
        self._handlers: Dict[str, InterruptHandler] = {}
        self._lock = threading.Lock()
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._session = None
        self._subscriber = None
        self._worker: threading.Thread | None = None

    def start(self) -> "InterruptListener":
        self._worker = threading.Thread(target=self._run, name="interrupt-listener", daemon=True)
        self._worker.start()
        self._session = self._session_factory()
        self._subscriber = self._session.declare_subscriber(self._key_expr, self._receive)
        logger.info(
            "Listening for Airflow interrupts on '%s' via %s",
            self._key_expr,
            os.getenv(CONNECT_ENV) or "multicast scouting",
        )
        return self

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.undeclare()
            self._subscriber = None
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()
            self._worker = None

    def __enter__(self) -> "InterruptListener":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def set_handler(self, dag_id: str, handler: InterruptHandler) -> None:
        with self._lock:
            self._handlers[dag_id] = handler

    def _receive(self, sample: zenoh.Sample) -> None:
        self._queue.put(sample.payload.to_bytes())

    def _run(self) -> None:
        while (payload := self._queue.get()) is not None:
            self.dispatch(payload)

    def dispatch(self, payload: bytes) -> None:
        """Decode one interrupt and pass it to the handler of its DAG."""
        try:
            data = json.loads(payload.decode("utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to decode Zenoh interrupt message: %s", exc)
            return
        with self._lock:
            handler = self._handlers.get(data.get("dag_id"))
        if handler is None:
            # Before the first cycle of a DAG, or with DAG-level optimization.
            logger.info("No placement handler for interrupt %s. Ignoring it.", data)
            return
        try:
            handler(data)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Handling interrupt %s failed", data)
