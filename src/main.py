import sys
from typing import Sequence

import logging
import os

from src.data_placement.interrupt_listener import InterruptListener
from src.network_monitoring.network_bandwith import check_network_bandwidth
from src.network_monitoring.network_config import network_config_from_env
from src.network_monitoring.provider_latency import check_pairwise_read_latency
from src.runtime.object_access_pipeline import (
    _build_onedata_managers,
    build_parser,
    configure_logging,
    resolve_config,
    setup_database_manager,
    run_monitoring_ingestion,
    run_placement_strategy,
    run_prediction,
    print_predictions,
)

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        config = resolve_config(args)
        logger.info("Configuration resolved successfully.")
    except ValueError as err:
        logger.exception(f"Configuration error: {err}")
        return 1

    try:
        db_manager = setup_database_manager(config)
        logger.info("SQLite database initialized successfully.")
    except Exception as err:
        logger.exception(f"Failed to initialize SQLite database: {err}")
        return 1

    try:
        zone_manager, provider_managers, devnull_space_id = _build_onedata_managers()
        logger.info("OneData managers built successfully.")
    except Exception as err:
        logger.exception(f"Failed to build OneData managers: {err}")
        return 1

    try:
        run_monitoring_ingestion(config, db_manager, provider_managers[list(provider_managers.keys())[0]])
        logger.info("Monitoring ingestion completed successfully.")
    except Exception as err:
        logger.exception(f"Failed to run monitoring ingestion: {err}")
        return 1

    predictions = []

    try:
        predictions = run_prediction(db_manager)
        logger.info("Prediction completed successfully.")
    except Exception as err:
        logger.exception(f"Failed to run prediction: {err}")
        
    try:
        print_predictions(predictions)
        logger.info("Predictions printed successfully.")
    except Exception as err:
        logger.exception(f"Failed to print prediction: {err}")

    try:
        # One subscription for all cycles, so interrupts sent while a cycle
        # ingests and plans are not lost.
        interrupt_listener = InterruptListener().start()
        # Measuring all provider pairs takes minutes and slows the workload down, so it
        # is refreshed only every few cycles, or replaced by fixed values.
        network_config = network_config_from_env()
        latency_refresh_cycles = int(os.getenv("PAIRWISE_LATENCY_REFRESH_CYCLES", "10"))
        provider_latency_matrix = None
        provider_bandwidth_matrix = None
        if network_config is not None:
            latency_refresh_cycles = 0
            provider_latency_matrix = network_config.pairwise_read_latency_ms
        cycle = 0
        while True:
            if latency_refresh_cycles > 0 and cycle % latency_refresh_cycles == 0:
                provider_latency_matrix = check_pairwise_read_latency(zone_manager, provider_managers)
                try:
                    provider_bandwidth_matrix = check_network_bandwidth(provider_managers, zone_manager)
                except Exception as err:
                    # Kairos then only requires the predicted start to lie in the future.
                    logger.exception(f"Failed to measure provider bandwidths: {err}")
            cycle += 1
            run_monitoring_ingestion(
                config,
                db_manager,
                provider_managers[list(provider_managers.keys())[0]],
            )
            run_placement_strategy(
                provider_managers=provider_managers,
                zone_manager=zone_manager,
                onedata_manager=provider_managers[list(provider_managers.keys())[0]],
                db_manager=db_manager,
                devnull_id=devnull_space_id,
                optimization_level=os.getenv("OPTIMIZATION_LEVEL", "dag"),
                prometheus_host=config.prometheus_host,
                monitor_lookback_minutes=config.lookback_minutes,
                interrupt_listener=interrupt_listener,
                provider_latency_matrix=provider_latency_matrix,
                provider_bandwidth_matrix=provider_bandwidth_matrix,
                network_config=network_config,
            )
    except Exception as e:
        logger.exception(f"Placement strategy failed: {e}")
        sys.exit(1)


    sys.exit(0)


if __name__ == "__main__":
    raise SystemExit(main())