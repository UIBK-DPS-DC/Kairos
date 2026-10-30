"""
Lightweight metrics and instrumentation for the data placement predictor.

Tracks placement cycles, errors, and cache activity. Can optionally export to Prometheus
if prometheus_client is installed.
"""

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Flag to check if prometheus_client is available for export
_prometheus_enabled = False
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    _prometheus_enabled = True
except ImportError:
    logger.debug("prometheus_client not installed; metrics will be in-memory only")


class MetricsCollector:
    """Metrics collector for placement predictor."""

    def __init__(self, enable_prometheus: bool = False):
        """
        Initialize metrics collector.

        Args:
            enable_prometheus: If True and prometheus_client is available, register Prometheus metrics.
        """
        self.enable_prometheus = enable_prometheus and _prometheus_enabled

        # In-memory counters
        self._placement_cycles_total = 0
        self._placement_errors_total = 0
        self._monitor_cycles_total = 0
        self._monitor_errors_total = 0
        self._last_successful_placement_timestamp = None
        self._last_successful_monitor_timestamp = None
        self._placement_duration_seconds = 0.0
        self._monitor_duration_seconds = 0.0

        # Prometheus metrics (if available)
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self.placement_cycles = Counter(
                "predictor_placement_cycles_total",
                "Total placement cycles executed",
                registry=self.registry,
            )
            self.placement_errors = Counter(
                "predictor_placement_errors_total",
                "Total placement cycle errors",
                registry=self.registry,
            )
            self.placement_duration = Histogram(
                "predictor_placement_duration_seconds",
                "Placement cycle duration in seconds",
                registry=self.registry,
            )
            self.monitor_cycles = Counter(
                "predictor_monitor_cycles_total",
                "Total monitor cycles executed",
                registry=self.registry,
            )
            self.monitor_errors = Counter(
                "predictor_monitor_errors_total",
                "Total monitor errors",
                registry=self.registry,
            )
            self.monitor_duration = Histogram(
                "predictor_monitor_duration_seconds",
                "Monitor cycle duration in seconds",
                registry=self.registry,
            )
            self.last_placement_timestamp = Gauge(
                "predictor_last_successful_placement_timestamp",
                "Unix timestamp of last successful placement cycle",
                registry=self.registry,
            )
            self.last_monitor_timestamp = Gauge(
                "predictor_last_successful_monitor_timestamp",
                "Unix timestamp of last successful monitor cycle",
                registry=self.registry,
            )
        else:
            self.registry = None

    def record_placement_cycle_start(self):
        """Record start of a placement cycle."""
        return _CycleTimer(self, "placement")

    def record_monitor_cycle_start(self):
        """Record start of a monitor cycle."""
        return _CycleTimer(self, "monitor")

    def record_placement_success(self, duration_seconds: float = 0.0):
        """Record successful placement cycle."""
        self._placement_cycles_total += 1
        self._last_successful_placement_timestamp = datetime.utcnow().timestamp()
        self._placement_duration_seconds = duration_seconds

        if self.enable_prometheus:
            self.placement_cycles.inc()
            self.placement_duration.observe(duration_seconds)
            self.last_placement_timestamp.set(self._last_successful_placement_timestamp)

        logger.info(
            "Placement cycle completed (cycle=%d, duration=%.2fs)",
            self._placement_cycles_total,
            duration_seconds,
        )

    def record_placement_error(self):
        """Record placement cycle error."""
        self._placement_errors_total += 1

        if self.enable_prometheus:
            self.placement_errors.inc()

        logger.warning(
            "Placement cycle failed (cycle=%d, errors=%d)",
            self._placement_cycles_total + 1,
            self._placement_errors_total,
        )

    def record_monitor_success(self, duration_seconds: float = 0.0):
        """Record successful monitor cycle."""
        self._monitor_cycles_total += 1
        self._last_successful_monitor_timestamp = datetime.utcnow().timestamp()
        self._monitor_duration_seconds = duration_seconds

        if self.enable_prometheus:
            self.monitor_cycles.inc()
            self.monitor_duration.observe(duration_seconds)
            self.last_monitor_timestamp.set(self._last_successful_monitor_timestamp)

        logger.debug(
            "Monitor cycle completed (cycle=%d, duration=%.2fs)",
            self._monitor_cycles_total,
            duration_seconds,
        )

    def record_monitor_error(self):
        """Record monitor cycle error."""
        self._monitor_errors_total += 1

        if self.enable_prometheus:
            self.monitor_errors.inc()

        logger.warning(
            "Monitor cycle failed (cycle=%d, errors=%d)",
            self._monitor_cycles_total + 1,
            self._monitor_errors_total,
        )

    def get_summary(self) -> dict:
        """Get current metrics summary."""
        return {
            "placement_cycles_total": self._placement_cycles_total,
            "placement_errors_total": self._placement_errors_total,
            "placement_error_rate": (
                self._placement_errors_total / self._placement_cycles_total
                if self._placement_cycles_total > 0
                else 0.0
            ),
            "placement_last_success_timestamp": self._last_successful_placement_timestamp,
            "placement_last_duration_seconds": self._placement_duration_seconds,
            "monitor_cycles_total": self._monitor_cycles_total,
            "monitor_errors_total": self._monitor_errors_total,
            "monitor_error_rate": (
                self._monitor_errors_total / self._monitor_cycles_total
                if self._monitor_cycles_total > 0
                else 0.0
            ),
            "monitor_last_success_timestamp": self._last_successful_monitor_timestamp,
            "monitor_last_duration_seconds": self._monitor_duration_seconds,
        }

    def export_prometheus(self) -> Optional[bytes]:
        """Export metrics in Prometheus format (requires prometheus_client)."""
        if not self.enable_prometheus or self.registry is None:
            return None
        return generate_latest(self.registry)

    def get_prometheus_content_type(self) -> str:
        """Get content type for Prometheus metrics."""
        return CONTENT_TYPE_LATEST if _prometheus_enabled else "text/plain"


class _CycleTimer:
    """Context manager for measuring cycle duration and recording metrics."""

    def __init__(self, collector: MetricsCollector, cycle_type: str):
        self.collector = collector
        self.cycle_type = cycle_type
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if exc_type is None:
            if self.cycle_type == "placement":
                self.collector.record_placement_success(duration)
            elif self.cycle_type == "monitor":
                self.collector.record_monitor_success(duration)
        else:
            if self.cycle_type == "placement":
                self.collector.record_placement_error()
            elif self.cycle_type == "monitor":
                self.collector.record_monitor_error()

        return False  # Re-raise exception if any


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_instance  # pylint: disable=global-statement
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(enable_prometheus=_prometheus_enabled)
    return _metrics_instance


def initialize_metrics(enable_prometheus: bool = False) -> MetricsCollector:
    """Initialize global metrics collector with specific settings."""
    global _metrics_instance  # pylint: disable=global-statement
    _metrics_instance = MetricsCollector(enable_prometheus=enable_prometheus)
    return _metrics_instance
