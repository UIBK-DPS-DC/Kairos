from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
import requests

from src.rest_api.storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class IPlacement(ABC):
    """Contract for all data placement strategy implementations."""

    @abstractmethod
    def apply_placement(self) -> int:
        """Execute one strategy pass and return structured result details."""
        raise NotImplementedError