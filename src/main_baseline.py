import sys
import time
from typing import Sequence

import logging

from src.runtime.object_access_pipeline import (
    _build_onedata_managers,
    build_parser,
    configure_logging,
    resolve_config,
)
from src.data_placement.strategies.Popularity_Baseline import Popularity_Baseline

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    try:
        resolve_config(args)
        logger.info("Configuration resolved successfully.")
    except ValueError as err:
        logger.exception(f"Configuration error: {err}")
        return 1


    try:
        zone_manager, provider_managers, devnull_space_id = _build_onedata_managers()
        logger.info("OneData managers built successfully.")
    except Exception as err:
        logger.exception(f"Failed to build OneData managers: {err}")
        return 1

    try:
        baseline_strategy = Popularity_Baseline(
            provider_managers=provider_managers,
            zone_manager=zone_manager,
            onedata_manager=provider_managers["uibk"],
            devnull_id=devnull_space_id,
        )
    except Exception as err:
        logger.exception(f"Failed to build baseline strategy: {err}")
        return 1

    try:
        while True:
            baseline_strategy.apply_placement()
            time.sleep(60)  # Wait for 60 seconds before the next placement
    except Exception as e:
        logger.exception(f"Placement strategy failed: {e}")
        sys.exit(1)


    sys.exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
