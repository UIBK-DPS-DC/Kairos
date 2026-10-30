from typing import Dict, Tuple
import math
import json
import logging

from src.data_placement.IPlacement import IPlacement
from src.rest_api.storage_interface import StorageInterface
import hashlib
import numpy as np


logger = logging.getLogger(__name__)


class Popularity_Baseline(IPlacement):
    """
    Baseline implementation of
    Popularity-Based Data Placement With Load Balancing in Edge Computing from Wei and Wang .
    """
    def __init__(
        self,
        provider_managers: Dict[str, StorageInterface],
        zone_manager: StorageInterface,
        onedata_manager: StorageInterface,
        devnull_id: str = None,
    ):
        self.provider_managers = provider_managers
        self.zone_manager = zone_manager
        self.onedata_manager = onedata_manager
        self.devnull_id = devnull_id
        # self.latency_map = check_pairwise_provider_latency(
        #     self.zone_manager, self.provider_managers
        # )
        self.latency_map = None
        self.provider_ids = list(self.provider_managers.keys())
        self.provider_api_ids = {
            provider.name or provider.provider_id: provider.provider_id
            for provider in self.zone_manager.list_providers()
        }
        # self.L = self._build_shortest_path_matrix() # change for hardcoded matrix because it takes quite long
        self.L = [[0.0, 17033.26329133324, 22210.319396333336, 17045.19473899988], [20204.92541633318, 0.0, 21875.62502966663, 20210.39367066669], [21732.91304899991, 16713.31890133327, 0.0, 21727.101789999928], [16772.7249836668, 16786.743021333376, 16777.690489666536, 0.0]]
        logger.info(
            "Popularity baseline shortest-path matrix (copy into code):\n"
            "provider_ids = %s\n"
            "L = %s",
            repr(self.provider_ids),
            repr(self.L),
        )
        self.provider_coordinates = {}
        self.data_object_coordinates = {}
        self.horizon_end_time = None

    @staticmethod
    def to_4byte_hash(data_object_id: str) -> bytes:
        return hashlib.sha256(data_object_id.encode()).digest()[:4]

    @staticmethod
    def calculate_coordinates_data_object(
        data_object_id: str, popularity: float, min_popularity: float, max_popularity: float
    ) -> Tuple[float, float]:
        # Min-Max scale the popularity so it spans exactly 0.0 to 1.0
        if max_popularity > min_popularity:
            normalized_pop = (popularity - min_popularity) / (max_popularity - min_popularity)
        else:
            normalized_pop = 1.0  # If all files have identical popularity, place at center

        r_object = 1.0 - normalized_pop
        theta_object = 2 * math.pi * int(Popularity_Baseline.to_4byte_hash(data_object_id).hex(), 16) / (2**32 - 1)

        x = r_object * math.cos(theta_object)
        y = r_object * math.sin(theta_object)
        return (x, y)

    def calculate_coordinates_providers(self) -> Dict[str, Tuple[float, float]]:
        # Handle edge cases for small networks
        N = len(self.provider_ids)
        if N == 0:
            return {}
        if N == 1:
            return {self.provider_ids[0]: (0.0, 0.0)}

        L = np.array(self.L)

        # Step 1: Compute the squared distance matrix L^(2)
        L2 = np.square(L)

        # Step 2: Compute the scalar product matrix B = -1/2 * J * L^(2) * J
        I = np.eye(N)
        A = np.ones((N, N))
        J = I - (1.0 / N) * A
        B = -0.5 * J.dot(L2).dot(J)

        # Step 3: Determine the two largest eigenvalues and corresponding eigenvectors of B
        # Using eigh since matrix B is symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort in descending order to isolate the top 2
        idx = np.argsort(eigenvalues)[::-1]
        top_eigenvalues = eigenvalues[idx][:2]
        top_eigenvectors = eigenvectors[:, idx][:, :2]

        # Step 4: Construct coordinates Q = Lambda_2^(1/2) * V_2^T
        # np.maximum prevents math errors if floating point precision yields a tiny negative zero
        lambda2_sqrt = np.diag(np.sqrt(np.maximum(top_eigenvalues, 0)))
        Q = lambda2_sqrt.dot(top_eigenvectors.T)

        # Step 5: Normalize Q so all coordinates sit within a circular region with unit radius
        q_max = np.max(np.abs(Q))
        if q_max > 0:
            Q_norm = (1.0 / (math.sqrt(2) * q_max)) * Q
        else:
            Q_norm = Q

        # Map the 2D coordinates back to provider IDs
        provider_coords = {}
        for i, provider_id in enumerate(self.provider_ids):
            # Q_norm is a 2xN matrix. Row 0 is X, Row 1 is Y.
            # Fallback to 0.0 for the Y axis if the network only has 2 nodes (1D line)
            x_val = float(Q_norm[0, i])
            y_val = float(Q_norm[1, i]) if Q_norm.shape[0] > 1 else 0.0
            
            provider_coords[provider_id] = (x_val, y_val)

        return provider_coords

    def _build_shortest_path_matrix(self) -> list[list[float | None]]:
        return [
            [self.latency_map[source_id][target_id] for target_id in self.provider_ids]
            for source_id in self.provider_ids
        ]

    def calculate_number_of_replicas(
        self,
        popularity: float, min_popularity: float, max_popularity: float,
        file_size: float, max_file_size: float,
        current_storage_usage: Dict[str, float],
        provider_capacities: Dict[str, float],
        alpha1=0.33, alpha2=0.33, alpha3=0.34, beta=0.5
    ) -> int:
        """Calculates n(d_i) based on data size, popularity, and available storage."""
        N = len(self.provider_ids)
        if N == 0:
            return 1
            
        norm_pop = (popularity - min_popularity) / (max_popularity - min_popularity) if max_popularity > min_popularity else 1.0
        norm_size = file_size / max_file_size if max_file_size > 0 else 1.0
        
        total_capacity = sum(provider_capacities.values())
        if total_capacity == float('inf'):
            norm_storage = 1.0
        elif total_capacity > 0:
            total_remaining = sum(
                provider_capacities[p] - current_storage_usage.get(p, 0.0) 
                for p in self.provider_ids
            )
            norm_storage = total_remaining / total_capacity
        else:
            norm_storage = 0.0

        # Calculate number of replicas (Eq 4)
        n_replicas = int((alpha1 * norm_size + alpha2 * norm_pop + alpha3 * norm_storage) * beta * N)
        
        # Bounding: At least 1 replica, at most a replica on every server
        return max(1, min(n_replicas, N))

    @staticmethod
    def calculate_coordinates_data_replicas(
        data_object_id: str, 
        popularity: float, 
        min_popularity: float, 
        max_popularity: float,
        n_replicas: int,
        t: float = 0.01
    ) -> list[Tuple[float, float]]:
        """Calculates coordinates for all replicas of a data object (Eq 5 & 6)."""
        if max_popularity > min_popularity:
            normalized_pop = (popularity - min_popularity) / (max_popularity - min_popularity)
        else:
            normalized_pop = 1.0 

        base_r = 1.0 - normalized_pop
        base_theta = 2 * math.pi * int(Popularity_Baseline.to_4byte_hash(data_object_id).hex(), 16) / (2**32 - 1)

        replicas_coords = []
        for k in range(n_replicas):
            # Shift the radius by 0.5 for replicas of highly popular data to prevent collision at the center
            if base_r < t and k > 0:
                r_object = base_r + 0.5
            else:
                r_object = base_r
                
            # Evenly distribute the angles
            theta_object = base_theta + k * (2 * math.pi / n_replicas)
            
            x = r_object * math.cos(theta_object)
            y = r_object * math.sin(theta_object)
            replicas_coords.append((x, y))
            
        return replicas_coords

    def reconcile_file_placement(
        self, file_id: str, desired_providers: set[str]
    ) -> dict[str, list]:
        """Schedule transfers needed to match a file's desired placement.

        Replications are scheduled before evictions so the file keeps at least
        one complete replica while the placement is being reconciled.
        """
        unknown_providers = desired_providers - set(self.provider_ids)
        if unknown_providers:
            raise ValueError(
                f"Unknown desired providers for file {file_id}: {unknown_providers}"
            )

        desired_provider_ids = {
            getattr(self, "provider_api_ids", {}).get(provider_id, provider_id)
            for provider_id in desired_providers
        }
        current_providers: set[str] = set()
        complete_providers: set[str] = set()
        for provider_id, manager in self.provider_managers.items():
            distribution = manager.get_replica_distribution(file_id)
            current_providers.update(distribution.provider_ids)
            for source_provider in distribution.fully_replicated_providers:
                complete_providers.add(source_provider)

        providers_to_replicate = desired_provider_ids - complete_providers
        providers_to_evict = current_providers - desired_provider_ids
        replications = []
        evictions = []

        for provider_id in sorted(providers_to_replicate):
            replications.append(
                self.onedata_manager.replicate(
                    file_id,
                    provider_id,
                    zone_manager=self.zone_manager,
                )
            )

        for provider_id in sorted(providers_to_evict):
            evict_by_id = getattr(
                self.onedata_manager,
                "evict_from_provider_id",
                self.onedata_manager.evict,
            )
            evictions.append(evict_by_id(file_id, provider_id))

        logger.info(
            "Reconciled file %s: replicated=%s evicted=%s unchanged=%s",
            file_id,
            sorted(providers_to_replicate),
            sorted(providers_to_evict),
            sorted(current_providers & desired_provider_ids),
        )
        return {"replications": replications, "evictions": evictions}

    def apply_placement(self):
        popularity_by_file = {}
        popularity_rows = self.onedata_manager.get_file_popularity_with_values(
            self.zone_manager
        )
        current_file_ids = set(self.onedata_manager.get_all_file_ids())
        for provider_rows in popularity_rows.values():
            for row in provider_rows:
                file_id = row.get("value")
                popularity = row.get("key")
                if (
                    file_id is None
                    or popularity is None
                    or file_id not in current_file_ids
                ):
                    continue
                popularity_by_file[file_id] = max(
                    popularity_by_file.get(file_id, 0.0), float(popularity)
                )

        max_popularity = max(popularity_by_file.values(), default=0.0)
        min_popularity = min(popularity_by_file.values(), default=0.0)

        if max_popularity > 0:
            self.data_object_coordinates = {
                file_id: self.calculate_coordinates_data_object(
                    file_id, popularity, min_popularity, max_popularity
                )
                for file_id, popularity in popularity_by_file.items()
            }
        else:
            self.data_object_coordinates = {}

        self.provider_coordinates = self.calculate_coordinates_providers()

        coordinates_for_plotting = {
            "providers": [
                {"id": provider_id, "x": x, "y": y}
                for provider_id, (x, y) in self.provider_coordinates.items()
            ],
            "data_objects": [
                {"id": file_id, "x": x, "y": y}
                for file_id, (x, y) in self.data_object_coordinates.items()
            ],
        }
        logger.info(
            "Popularity baseline coordinates (copy as JSON):\n%s",
            json.dumps(coordinates_for_plotting, indent=2, sort_keys=True),
        )

        # Data Placement in Order of Popularity
        current_storage_usage = {pid: 0.0 for pid in self.provider_ids}
        provider_capacities = {pid: float("inf") for pid in self.provider_ids}
        file_sizes = {
            file_id: float(self.onedata_manager.get_file_size(file_id))
            for file_id in popularity_by_file
        }
        max_file_size = max(file_sizes.values(), default=1.0)

        replica_assignments = []
        self.data_object_coordinates = {}

        for file_id, pop in popularity_by_file.items():
            n_replicas = self.calculate_number_of_replicas(
                pop, min_popularity, max_popularity,
                file_sizes[file_id], max_file_size,
                current_storage_usage, provider_capacities
            )
            
            coords = self.calculate_coordinates_data_replicas(
                file_id, pop, min_popularity, max_popularity, n_replicas
            )
            self.data_object_coordinates[file_id] = coords
            
            for k, (dx, dy) in enumerate(coords):
                replica_assignments.append({
                    'file_id': file_id,
                    'replica_idx': k,
                    'popularity': pop,
                    'file_size': file_sizes[file_id],
                    'x': dx,
                    'y': dy
                })

        for req in replica_assignments:
            nearest_provider = None
            min_dist = float('inf')
            
            for provider_id, (px, py) in self.provider_coordinates.items():
                dist = math.sqrt((px - req['x'])**2 + (py - req['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_provider = provider_id
                    
            req['target_provider'] = nearest_provider

        replica_assignments.sort(key=lambda req: req['popularity'], reverse=True)

        final_placement = {file_id: set() for file_id in popularity_by_file.keys()}

        for req in replica_assignments:
            file_id = req['file_id']
            target_provider = req['target_provider']
            file_size = req['file_size']
            
            if (current_storage_usage[target_provider] + file_size > provider_capacities[target_provider] 
                or target_provider in final_placement[file_id]):
                
                best_offload_provider = None
                best_dist = float('inf')
                
                for pid, (px, py) in self.provider_coordinates.items():
                    if (current_storage_usage[pid] + file_size <= provider_capacities[pid] 
                        and pid not in final_placement[file_id]):
                        
                        dist = math.sqrt((px - req['x'])**2 + (py - req['y'])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_offload_provider = pid
                            
                if best_offload_provider:
                    target_provider = best_offload_provider
                else:
                    logger.warning("No valid providers to offload replica for file %s.", file_id)
                    continue 
            
            final_placement[file_id].add(target_provider)
            current_storage_usage[target_provider] += file_size

        for file_id, desired_providers in final_placement.items():
            self.reconcile_file_placement(file_id, desired_providers)

        return final_placement