# Compatibility shim: re-export IPlacement under the names used by strategy modules.
from src.data_placement.IPlacement import IPlacement

# Aliases used by existing strategy files
PlacementStrategyInterface = IPlacement
AbstractPlacementStrategy = IPlacement

__all__ = ["PlacementStrategyInterface", "AbstractPlacementStrategy", "IPlacement"]
