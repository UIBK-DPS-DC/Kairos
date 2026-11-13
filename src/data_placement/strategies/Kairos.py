import logging
import os
from typing import Any, Dict, List, Optional, Set
import json
from collections import defaultdict, deque
import threading
import time
from datetime import datetime, timedelta


from src.data_placement.IPlacement import IPlacement
from src.data_placement.interrupt_listener import InterruptListener
from src.rest_api.storage_interface import StorageInterface
from src.rest_api.onedata.onedatamanager import OnedataDataManager, ReplicaDistribution
from src.dag_prediction.task_predictor import TaskPrediction, TaskPredictor
import src.database.db_interface as db
from src.network_monitoring.provider_latency import check_all_providers_latency_with_ids
from src.network_monitoring.network_bandwith import PROBE_SIZE_BYTES
from src.network_monitoring.network_config import NetworkConfig


logger = logging.getLogger(__name__)

# Tolerated clock difference between Airflow and the Onedata providers.
FILE_MTIME_SLACK_SECONDS = 5.0

# Onedata needs ~5-15 s to schedule and finish even a tiny replication. Used when
# no probe transfer between the providers was measured.
DEFAULT_REPLICATION_OVERHEAD_SECONDS = float(os.getenv("KAIROS_REPLICATION_OVERHEAD_SECONDS", "10"))
# Throughput of the data itself; the probe is too small to measure it.
REPLICATION_BANDWIDTH_BYTES_PER_SECOND = float(os.getenv("KAIROS_REPLICATION_BANDWIDTH", str(20_000_000)))

# How long a replica Kairos created is kept after its task's predicted start.
# A fresh replica is the least popular one of its file, so budget expansion would
# otherwise evict it first, even before the task has read it.
REPLICA_PROTECTION_SECONDS = float(os.getenv("KAIROS_REPLICA_PROTECTION_SECONDS", "300"))
# (file_id, provider_id) -> unix time until which the replica is not evicted. Kairos is
# re-created every cycle, so the replicas are tracked for the whole process.
_protected_replicas: dict[tuple[str, str], float] = {}
# Interrupts replicate on the listener thread while budget expansion evicts on the main thread.
_replica_lock = threading.Lock()


def _protect_replica(file_id: str, provider_id: str, predicted_time: datetime | None) -> None:
    """Keep the new replica until ``REPLICA_PROTECTION_SECONDS`` after its task's predicted start."""
    start = max(time.time(), predicted_time.timestamp() if predicted_time is not None else 0.0)
    _protected_replicas[(file_id, provider_id)] = start + REPLICA_PROTECTION_SECONDS


def _is_protected(file_id: str, provider_id: str) -> bool:
    until = _protected_replicas.get((file_id, provider_id))
    if until is None:
        return False
    if until < time.time():
        del _protected_replicas[(file_id, provider_id)]
        return False
    return True


class Kairos(IPlacement):
    """Kairos placement strategy implementation."""

    def __init__(
        self,
        provider_managers: Dict[str, StorageInterface],
        zone_manager: StorageInterface,
        onedata_manager: StorageInterface,
        db_manager: db.DBInterface,
        devnull_id: str = None,
        optimization_level: str = "dag",
        interrupt_listener: InterruptListener | None = None,
        provider_latency_matrix: Dict[str, Dict[str, float | None]] | None = None,
        provider_bandwidth_matrix: Dict[str, Dict[str, float]] | None = None,
        network_config: NetworkConfig | None = None,
    ):
        """``interrupt_listener``: a listener that outlives this cycle; without
        one, interrupts are only received during the dynamic orchestration.

        ``provider_latency_matrix``: ``latency[reader_id][holder_id]`` in ms from
        :func:`check_pairwise_read_latency`, measured less often than once per cycle.

        ``provider_bandwidth_matrix``: ``bandwidth[target_id][source_id]`` in bytes/s
        of the probe transfers from :func:`check_network_bandwidth`. The probe is so
        small that its transfer time is Onedata's fixed replication overhead.

        ``network_config``: fixed upload RTTs and replication overhead/bandwidth;
        the upload RTTs are then not measured."""
        self.provider_managers = provider_managers
        self.zone_manager = zone_manager
        self.devnull_id = devnull_id
        self.onedata_manager = onedata_manager
        self.db_manager = db_manager
        self.network_config = network_config
        if network_config is not None:
            self.latency_map = dict(network_config.upload_rtt_ms)
        else:
            self.latency_map = check_all_providers_latency_with_ids(
                self.zone_manager, self.provider_managers, devnull_id=devnull_id
            )
        self.prices = self.read_price_config()
        self.horizon_end_time = None
        self.task_predictor: TaskPredictor | None = None
        self.optimization_level = optimization_level
        self.interrupt_listener = interrupt_listener
        self.provider_latency_matrix = provider_latency_matrix
        self.provider_bandwidth_matrix = provider_bandwidth_matrix

    def _get_global_replica_distribution(self, file_id: str) -> ReplicaDistribution:
        """
        Query all provider managers and merge results into a complete
        ReplicaDistribution.

        A single provider's ``/data/{id}/distribution`` endpoint only reports
        its own replica (or returns 404 if it doesn't host the file at all).
        This method queries every known provider and merges the per-provider
        entries so that ``replica_count`` reflects the true global state.
        """
        merged = ReplicaDistribution(file_ref=file_id)

        for provider_id, manager in self.provider_managers.items():
            try:
                dist = manager.get_replica_distribution(file_id)
            except Exception as exc:
                logger.debug(
                    "Provider %s returned no distribution for file %s: %s",
                    provider_id,
                    file_id,
                    exc,
                )
                continue

            for pid, info in dist.distribution_per_provider.items():
                existing = merged.distribution_per_provider.get(pid)
                if existing is None or info.get("blocksPercentage", 0) > existing.get("blocksPercentage", 0):
                    merged.distribution_per_provider[pid] = info

        if not merged.distribution_per_provider:
            logger.debug(
                "No provider returned distribution data for file %s; "
                "the file may have been deleted.",
                file_id,
            )

        return merged

    def _get_file_id_with_retry(
        self,
        file_path: str,
        initial_delay: float = 0.5,
        max_total_wait: float = 20.0,
        written_after: float | None = None,
    ) -> str | None:
        """
        Retrieve file ID for a path using exponential backoff retry.

        Useful when storage backend has not yet exposed newly created files/folders in its API.
        With ``written_after`` (unix seconds), an ID whose file was last modified
        before then is retried as well: a provider that has not synced a
        re-created file yet still resolves the path to the previous file.
        """
        retry_delay = initial_delay
        total_waited = 0.0

        while True:
            file_id = self.onedata_manager.get_file_id_from_path(file_path)
            if file_id is not None and written_after is None:
                return file_id
            if file_id is not None:
                mtime = self.onedata_manager.get_file_mtime(file_id)
                # Without an mtime the ID cannot be checked, so it is taken as is.
                if mtime is None or mtime >= written_after - FILE_MTIME_SLACK_SECONDS:
                    return file_id
                logger.debug(
                    "Path %s still resolves to %s, last modified %.1fs before the task started.",
                    file_path,
                    file_id,
                    written_after - mtime,
                )
            if total_waited >= max_total_wait:
                break
            sleep_duration = min(retry_delay, max_total_wait - total_waited)
            logger.debug(
                "File ID for path %s not yet available; retrying in %.1fs (total waited: %.1fs / %.1fs).",
                file_path,
                sleep_duration,
                total_waited,
                max_total_wait,
            )
            time.sleep(sleep_duration)
            total_waited += sleep_duration
            retry_delay = min(retry_delay * 2, max_total_wait)

        logger.warning(
            "Could not retrieve file ID for path %s after %.1fs of retries. Skipping replication for this file.",
            file_path,
            total_waited,
        )
        return None

    def read_price_config(self, config_file_path: Optional[str] = None) -> Dict[str, float]:
        """
        Read the provider price configuration from a file.

        Args:
            config_file_path (str): Path to the configuration file. Defaults to
                KAIROS_PRICE_CONFIG, or src/data_placement/strategies/price_conf.json.
                See price_conf.example.json for the format.

        Returns:
            Dict[str, float]: A dictionary mapping provider IDs to their respective prices.
        """
        # read the json file and return a dictionary mapping provider IDs to their respective static prices
        config_file_path = config_file_path or os.getenv(
            "KAIROS_PRICE_CONFIG", "src/data_placement/strategies/price_conf.json"
        )
        with open(config_file_path, "r") as f:
            config = json.load(f)
        return {provider_id: prices["static"] for provider_id, prices in config.items()}

    class transition_candidate:
        """Class to represent a transition in the Kairos placement strategy."""
        def __init__(
            self,
            file_id: str,
            to_provider: str,
            value_cost_ration: float,
            value: float,
            cost: float,
            predicted_time: datetime | None = None,
            holders: list[str] | None = None,
            size: int = 0,
            task_order: int = 0,
        ):
            self.file_id = file_id
            self.to_provider = to_provider
            self.value_cost_ration = value_cost_ration
            self.value = value
            self.cost = cost
            # Predicted start of the task that reads the file; the replication must finish before it.
            self.predicted_time = predicted_time
            self.holders = holders or []
            self.size = size
            # Topological position of the reading task, the tie-breaker for equal VCRs.
            self.task_order = task_order

    class dag_node:
        """Class to represent a node in the workflow DAG."""
        def __init__(self, task_id: str):
            self.task_id = task_id

        def __str__(self):
            return f"dag_node(task_id={self.task_id})"

        def __repr__(self):
            return self.__str__()
    class dag_edge:
        """Class to represent an edge in the workflow DAG."""
        def __init__(self, from_node: 'Kairos.dag_node', to_node: 'Kairos.dag_node'):
            self.from_node = from_node
            self.to_node = to_node

        def __str__(self):
            return f"dag_edge(from_node={self.from_node}, to_node={self.to_node})"

        def __repr__(self):
            return self.__str__()
    class dag:
        """Class to represent the workflow DAG."""
        def __init__(self, dag_id: str, nodes: list['Kairos.dag_node'], edges: list['Kairos.dag_edge']):
            self.dag_id = dag_id
            self.nodes: list['Kairos.dag_node'] = nodes
            self.edges: list['Kairos.dag_edge'] = edges

        def __str__(self):
            return f"\ndag(dag_id={self.dag_id},\nnodes={self.nodes},\nedges={self.edges})"

        def __repr__(self):
            return self.__str__()

    def get_static_dynamic_semiDynamic_file_paths(self, dag: dag) -> dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]]:
        """
        Get the file paths for static, dynamic, and semi-dynamic files for each task in the DAG.

        Args:
            dag (Kairos.dag): The workflow DAG to be used for placement decisions.

        Returns:
            dict: A dictionary mapping each task_id to a tuple of:
                - static (List[str]): Files read by this node that are not written by any
                                      OTHER node (a file this node both reads and writes is
                                      still static — it reads an existing version first).
                - dynamic (Dict[str, List[str]]): Files written by this node that are read by
                                                  at least one IMMEDIATE child. Mapped to the
                                                  list of immediate child task_ids that read it.
                - semi_dynamic (Dict[str, List[str]]): Files written by this node that are read
                                                       by at least one NON-IMMEDIATE downstream
                                                       node. Mapped to those task_ids.
                  Note: dynamic and semi_dynamic can coexist for the same file (e.g. both a
                  direct child and a grandchild read it).
        """
        task_files = {}

        # 1. Precompute reads/writes for each node and identify global writes
        global_writes: Set[str] = set()
        node_reads: Dict[str, Set[str]] = {}
        node_writes: Dict[str, Set[str]] = {}

        topologically_sorted_nodes = self.topological_sort(dag.nodes, dag.edges)

        for node in dag.nodes:
            reads = set(self.db_manager.read_paths_of_read_accessed_files_by_task(node.task_id, dag.dag_id))
            writes = set(self.db_manager.read_paths_of_write_accessed_files_by_task(node.task_id, dag.dag_id))

            node_reads[node.task_id] = reads
            node_writes[node.task_id] = writes
            global_writes.update(writes)

        # 2. Build a children map from dag.edges
        children_map: Dict[str, List[str]] = {node.task_id: [] for node in dag.nodes}
        for edge in dag.edges:
            children_map[edge.from_node.task_id].append(edge.to_node.task_id)

        # 3. Use Reverse Topological Sort to accumulate all descendant readers efficiently
        # downstream_readers[task_id][file_path] = set of downstream task_ids that read this file
        downstream_readers: Dict[str, Dict[str, Set[str]]] = {node.task_id: defaultdict(set) for node in dag.nodes}

        for node in reversed(topologically_sorted_nodes):
            tid = node.task_id
            for child_id in children_map[tid]:
                # a. If the child directly reads a file, add child to the readers set
                for f in node_reads[child_id]:
                    downstream_readers[tid][f].add(child_id)

                # b. Inherit readers from the child's descendants
                for f, descendant_readers in downstream_readers[child_id].items():
                    downstream_readers[tid][f].update(descendant_readers)

        # 4. Categorize files for each node
        for node in dag.nodes:
            tid = node.task_id
            children_ids = children_map[tid]

            # STATIC: Files read by this node that are not written by any OTHER node.
            # A file this node both reads and writes still counts as static for this node
            # (the task reads an existing version of it, then overwrites it).
            upstream_writes = global_writes - node_writes[tid]
            static: List[str] = [f for f in node_reads[tid] if f not in upstream_writes]

            dynamic: Dict[str, List[str]] = {}
            semi_dynamic: Dict[str, List[str]] = {}

            for f in node_writes[tid]:
                all_readers = downstream_readers[tid].get(f, set())

                if not all_readers:
                    continue  # No downstream node reads this file; it's a sink output

                # DYNAMIC: immediate children that read this file.
                child_readers = {c for c in children_ids if f in node_reads[c]}
                if child_readers:
                    dynamic[f] = list(child_readers)

                # SEMI-DYNAMIC: non-immediate downstream readers (grandchildren etc.).
                # Can coexist with dynamic for the same file.
                indirect_readers = all_readers - child_readers
                if indirect_readers:
                    semi_dynamic[f] = list(indirect_readers)

            task_files[tid] = (static, dynamic, semi_dynamic)


        logger.info("Static, dynamic, and semi-dynamic file paths for DAG %s: %s", dag.dag_id, task_files)
        return task_files
    
    def topological_sort(self, nodes: list['Kairos.dag_node'], edges: list['Kairos.dag_edge']) -> list['Kairos.dag_node']:
        """
        Perform a topological sort on the DAG nodes.

        Args:
            nodes (list[Kairos.dag_node]): List of DAG nodes.
            edges (list[Kairos.dag_edge]): List of DAG edges.

        Returns:
            list[Kairos.dag_node]: Topologically sorted list of DAG nodes.
        """
        node_by_id = {node.task_id: node for node in nodes}
        in_degree = {task_id: 0 for task_id in node_by_id}
        adjacency: dict[str, list[str]] = {task_id: [] for task_id in node_by_id}

        for edge in edges:
            from_task_id = edge.from_node.task_id
            to_task_id = edge.to_node.task_id

            if from_task_id not in node_by_id or to_task_id not in node_by_id:
                raise ValueError(
                    f"Edge references unknown node: {from_task_id} -> {to_task_id}"
                )

            adjacency[from_task_id].append(to_task_id)
            in_degree[to_task_id] += 1

        queue = deque(
            task_id for task_id, degree in in_degree.items() if degree == 0
        )
        sorted_nodes: list[Kairos.dag_node] = []

        while queue:
            current_task_id = queue.popleft()
            sorted_nodes.append(node_by_id[current_task_id])

            for downstream_task_id in adjacency[current_task_id]:
                in_degree[downstream_task_id] -= 1
                if in_degree[downstream_task_id] == 0:
                    queue.append(downstream_task_id)

        if len(sorted_nodes) != len(nodes):
            raise ValueError("Graph has at least one cycle, topological sort not possible.")

        return sorted_nodes

    def apply_placement(
            self,
            workflow_dag: 'Kairos.dag',
            horizon_cost_budget = float("inf"),
            horizon_time_seconds: float = 60,
            number_of_least_popular_to_evict: int = 1,
            optimization_level: str = "dag",
            prediction_strategy: str = "split",
            ) -> int:
        """
        Apply the Kairos placement strategy for one cycle of a dag.
        
        Args:
            horizon_cost_budget (float): The budget for the placement strategy in Dollars. Default is infinity.
            workflow_dag (Kairos.dag): The workflow DAG to be used for placement decisions. Default is None.
            number_of_least_popular_to_evict (int): The number of least popular files to evict from providers. Default is 1.
            optimization_level (str): The level of optimization to apply. Default is "dag".
            prediction_strategy (str): Which predictor to use for task-next-access estimation.
                ``"split"`` (default) predicts time (Ridge on the task's full series) and
                provider (Markov over the task's provider sequence) separately.
                ``"setar"`` uses the SETAR tree model with forward-roll to always
                return a future timestamp.  ``"linear"`` uses a stateless linear-regression
                extrapolation that is simpler but equally guaranteed to be in the future.
                Both pick the provider with the earliest per-provider forecast.
        Returns:
            float: The budget expanded after evicting and placing files.
            float: The budget after static orchestration.
            float: The budget after dynamic orchestration.
        """

        if optimization_level not in ["dag", "task"]:
            logger.error("Invalid optimization level: %s. Must be 'dag' or 'task'.", optimization_level)
            return 1

        if prediction_strategy not in ["setar", "linear", "split"]:
            logger.warning(
                "Unknown prediction_strategy '%s'. Falling back to 'split'.",
                prediction_strategy,
            )
            prediction_strategy = "split"

        # Phase 1
        logger.info("Starting budget expansion for DAG %s", workflow_dag.dag_id)
        self.horizon_end_time = time.time() + horizon_time_seconds

        file_paths = self.get_static_dynamic_semiDynamic_file_paths(workflow_dag)

        expanded_budget = self.budget_expansion(
            number_of_least_popular_to_evict,
            self.prices,
            horizon_time_seconds,
            protected_file_ids=self._static_file_ids(file_paths),
        )
        expanded_budget += horizon_cost_budget

        # Phase 2
        logger.info("Starting static orchestration for DAG %s", workflow_dag.dag_id)
        try:
            self.task_predictor = TaskPredictor(db_manager=self.db_manager, strategy=prediction_strategy)
            logger.debug("Using prediction strategy: %s", prediction_strategy)
            predictions = self.task_predictor.predict_all_tasks_for_dag(workflow_dag.dag_id)
        except Exception as e:
            logger.error("Failed to initialize predictions (strategy: %s). Skipping predictions for this run. Error: %s", prediction_strategy, e)
            self.task_predictor = None
            predictions = {}

        # The budget freed by the evictions is spent in the static orchestration.
        static_orchestrated_budget = self.static_orchestration(workflow_dag, expanded_budget, horizon_time_seconds, predictions, file_paths)

        # Phase 3
        logger.info("Starting dynamic orchestration for DAG %s", workflow_dag.dag_id)

        if optimization_level == "task":
            logger.debug("Task-level optimization is enabled. Running dynamic orchestration.")
            dynamic_orchestrated_budget = self.dynamic_orchestration(
                workflow_dag, horizon_time_seconds, predictions, file_paths, static_orchestrated_budget
            )
        else:
            logger.debug("Task-level optimization is not enabled. Skipping dynamic orchestration.")
            dynamic_orchestrated_budget = static_orchestrated_budget
            # Wait out the horizon as with task level, so that both levels plan
            # (and evict) at the same rate and differ only in the dynamic phase.
            time.sleep(max(0.0, self.horizon_end_time - time.time()))

        return expanded_budget, static_orchestrated_budget, dynamic_orchestrated_budget

    def budget_expansion(
        self,
        number_of_least_popular_to_evict,
        provider_prices: Dict[str, float],
        horizon_time_seconds: float,
        protected_file_ids: Set[str] = frozenset(),
    ) -> float:
        """
        Expand the budget based on the current configuration.
        
        Args:
            number_of_least_popular_to_evict (int): The number of least popular files to evict from providers.
            provider_prices (Dict[str, float]): A dictionary mapping provider IDs to their respective prices.
            horizon_time_seconds (float): The horizon duration in seconds used to scale hourly prices.
            protected_file_ids (Set[str]): Files the DAG reads; none of their replicas is evicted.
                Replicas Kairos created recently are never evicted either.
        Returns:
            float: The expanded budget.
        """

        logger.debug("Starting budget expansion")

        popularity = self.onedata_manager.get_file_popularity_with_values(self.zone_manager)

        logger.debug("Popularity data retrieved: %s", popularity)

        if number_of_least_popular_to_evict <= 0:
            logger.debug("No evictions requested; budget expansion is 0.")
            return 0.0

        provider_prices = provider_prices or {}
        popularity_by_file: dict[str, dict[str, float]] = {}

        for provider_id, provider_popularity in popularity.items():
            for popularity_entry in provider_popularity:
                file_id = popularity_entry.get("value")
                popularity_value = popularity_entry.get("key")
                if file_id is None or popularity_value is None:
                    continue
                try:
                    score = float(popularity_value)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid popularity score %r for file %s on provider %s",
                        popularity_value,
                        file_id,
                        provider_id,
                    )
                    continue
                popularity_by_file.setdefault(file_id, {})[provider_id] = score

        # Sort all files by their minimum popularity score across providers (ascending =
        # least popular first), then limit to a small candidate pool before making
        # per-file API calls. Without this, we would call _get_global_replica_distribution
        # for every file in the popularity index (hundreds of files × all providers).
        candidate_pool_size = max(number_of_least_popular_to_evict * 10, 100)
        least_popular_files = sorted(
            popularity_by_file.items(),
            key=lambda item: min(item[1].values()),  # ascending: least popular first
        )[:candidate_pool_size]

        logger.debug(
            "Considering %d least-popular file candidates for eviction (pool=%d)",
            len(least_popular_files),
            candidate_pool_size,
        )

        evicted_size_per_provider: Dict[str, int] = {}
        horizon_hours = horizon_time_seconds / 3600.0

        for file_id, provider_popularity in least_popular_files:
            if file_id in protected_file_ids:
                logger.debug("Skipping eviction for file %s because the DAG reads it.", file_id)
                continue
            replica_distribution = self._get_global_replica_distribution(file_id)

            if replica_distribution.replica_count == 0:
                logger.debug(
                    "No distribution data for file %s; skipping eviction.",
                    file_id,
                )
                continue

            if replica_distribution.replica_count <= 1:
                logger.debug(
                    "Skipping eviction for file %s because it has only one replica.",
                    file_id,
                )
                continue
            else:
                logger.info(
                    "File %s has %d replicas. Continuing with eviction process.",
                    file_id,
                    replica_distribution.replica_count,
                )

            # Held until the evictions are started, so an interrupt cannot replicate
            # a file to a provider in between and see its new replica evicted.
            with _replica_lock:
                replicas_with_popularity = sorted(
                    (
                        (provider_id, provider_popularity.get(provider_id, 0.0))
                        for provider_id in replica_distribution.provider_ids
                        if not _is_protected(file_id, provider_id)
                    ),
                    key=lambda item: item[1],
                )

                if not replicas_with_popularity:
                    logger.debug(
                        "No unprotected replica with popularity data for file %s; skipping eviction.",
                        file_id,
                    )
                    continue

                max_evictions = min(
                    number_of_least_popular_to_evict,
                    replica_distribution.replica_count - 1,
                    len(replicas_with_popularity),
                )
                least_popular_providers = [
                    provider_id for provider_id, _ in replicas_with_popularity[:max_evictions]
                ]

                file_size = replica_distribution.maximum_physical_size
                logger.debug(
                    "File %s physical size: %d bytes (%.2f MB).",
                    file_id,
                    file_size,
                    file_size / 1_000_000,
                )
                for provider_id in least_popular_providers:
                    evicted_size_per_provider[provider_id] = (
                        evicted_size_per_provider.get(provider_id, 0) + file_size
                    )
                    logger.info(
                        "Evicting file %s from provider %s due to low popularity.",
                        file_id,
                        provider_id,
                    )
                    self.onedata_manager.evict_from_provider_id(file_id, provider_id)

        bytes_per_gb = 1_000_000_000
        logger.info("Evicted size per provider: %s", evicted_size_per_provider)
        logger.debug("Provider prices: %s", provider_prices)
        logger.debug("Horizon hours: %f", horizon_hours)
        expanded_budget = sum(
            (
                evicted_size_per_provider[provider_id]
                / bytes_per_gb
                * float(provider_prices.get(provider_id, 0.0))
                * horizon_hours
            )
            for provider_id in evicted_size_per_provider
        )
        logger.info("Expanded budget for the horizon by: %f $", expanded_budget)
        return expanded_budget

    def static_orchestration(self, dag: 'Kairos.dag', horizon_cost_budget: float, horizon_time_seconds: float, predictions: Dict[str, TaskPrediction], file_paths: dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]])-> float:
        """
        Perform static orchestration based on the current configuration.
        
        Args:
            dag (Kairos.dag): The workflow DAG to be used for placement decisions.
            horizon_cost_budget (float): The budget for the placement strategy in Dollars.
            horizon_time_seconds (float): The horizon duration in seconds used to scale hourly prices.
            predictions (Dict[str, TaskPrediction]): A dictionary mapping task IDs to their predicted executions.
            file_paths (dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]]): A dictionary mapping task IDs to their file paths.
        Returns:
            float: The remaining budget after static orchestration.
        """

        candidates: list[Kairos.transition_candidate] = []

        horizon_hours = horizon_time_seconds / 3600.0

        sorted_nodes = self.topological_sort(dag.nodes, dag.edges)
        logger.info("Topologically sorted nodes for DAG %s: %s", dag.dag_id, sorted_nodes)

        static_file_paths: list[str] = []

        for task_order, node in enumerate(sorted_nodes):
            logger.debug("Processing node %s in DAG %s", node.task_id, dag.dag_id)
            static_file_paths_of_node, _, _ = file_paths.get(node.task_id)
            static_file_paths.extend(static_file_paths_of_node)
            logger.info("Found %d static files for task %s", len(static_file_paths_of_node), node.task_id)
            
            predictions_for_node = predictions.get(f"{node.task_id}_{dag.dag_id}")
            logger.debug("Predictions for node %s in DAG %s: %s", node.task_id, dag.dag_id, predictions_for_node)
            if not predictions_for_node:
                logger.warning(
                    "No predictions found for task %s in DAG %s. Skipping placement.",
                    node.task_id,
                    dag.dag_id,
                )
                continue

            pred_provider_id = predictions_for_node.predicted_provider_id
            timestamp = predictions_for_node.predicted_next_access
            logger.debug(
                "Predicted next access time for task %s in DAG %s is %s on provider %s",
                node.task_id,
                dag.dag_id,
                timestamp,
                pred_provider_id,
            )
            
            current_time = datetime.fromtimestamp(time.time())
            

            if timestamp is None or current_time.__gt__(timestamp):
                logger.warning(
                    "Predicted next access time for task %s in DAG %s is in the past or None. Skipping placement.",
                    node.task_id,
                    dag.dag_id,
                )
                logger.warning(
                    "Current time: %s, Predicted time: %s -- therefore skipping placement",
                    current_time,
                    timestamp,
                )
                continue
            logger.debug("Not skipping placement because predicted time is in the future: %s", timestamp)



            for file_path in static_file_paths_of_node:

                file_id = self.onedata_manager.get_file_id_from_path(file_path)
                if file_id is None:
                    logger.warning(
                        "Could not retrieve file ID for path %s. Skipping placement for this file.",
                        file_path,
                    )
                    continue
                logger.debug("Processing static file %s (ID: %s) for task %s in DAG %s", file_path, file_id, node.task_id, dag.dag_id)
                candidate = self._transition_candidate_for(
                    file_id, pred_provider_id, timestamp, horizon_hours, task_order
                )
                if candidate is not None:
                    candidates.append(candidate)

        return self._execute_transitions(candidates, horizon_cost_budget)

    def _transition_candidate_for(
        self,
        file_id: str,
        pred_provider_id: str,
        predicted_time: datetime | None,
        horizon_hours: float,
        task_order: int = 0,
    ) -> 'Kairos.transition_candidate | None':
        """Replication of ``file_id`` to ``pred_provider_id`` with its value/cost ratio.

        Value: the remote read latency (ms) the task avoids by reading locally.
        Cost: the storage cost ($) the new replica induces over the horizon.
        None if no replication is needed or its value cannot be determined.
        """
        bytes_per_gb = 1_000_000_000
        distribution: ReplicaDistribution = self._get_global_replica_distribution(file_id)
        # A freshly written dynamic file may not be reported as fully replicated yet.
        holders = distribution.fully_replicated_providers or distribution.provider_ids

        if not holders:
            logger.debug("File %s is not stored anywhere. Skipping transition.", file_id)
            return None

        if pred_provider_id in holders:
            logger.debug(
                "Predicted provider %s for file %s is already storing the file. No transition needed.",
                pred_provider_id,
                file_id,
            )
            return None

        # The task runs next to its predicted provider: after the replication it
        # reads locally (~0 ms), without it Onedata fetches the file on the fly
        # from the provider storing it that is closest to the predicted provider.
        current_latency = self._remote_read_latency(pred_provider_id, holders)
        if current_latency is None:
            logger.warning(
                "No latency information available from predicted provider %s to the providers of file %s. Skipping.",
                pred_provider_id,
                file_id,
            )
            return None
        value = current_latency - 0.0

        if value <= 0:
            logger.warning(
                "Transitioning file %s to provider %s would result in negative or zero latency improvement (%s ms). Skipping.",
                file_id, pred_provider_id, value
            )
            return None

        size = distribution.maximum_physical_size
        cost = (size / bytes_per_gb) * self.prices.get(pred_provider_id, 0) * horizon_hours
        vcr = value / cost if cost > 0 else float("inf")

        logger.info("Transition candidate: file %s to provider %s (VCR: %s, Latency Impr: %s ms, Cost: %s $)", file_id, pred_provider_id, vcr, value, cost)
        return self.transition_candidate(
            file_id, pred_provider_id, vcr, value, cost,
            predicted_time=predicted_time, holders=list(holders), size=size, task_order=task_order,
        )

    def _static_file_ids(
        self, file_paths: dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]]
    ) -> Set[str]:
        """IDs of the files the DAG's tasks read without writing them first."""
        paths = {path for static, _, _ in file_paths.values() for path in static}
        file_ids = set()
        for path in paths:
            file_id = self.onedata_manager.get_file_id_from_path(path)
            if file_id is not None:
                file_ids.add(file_id)
        return file_ids

    def _replication_time(self, to_provider: str, holders: list[str], size: int) -> float:
        """Seconds to replicate ``size`` bytes to ``to_provider`` from the fastest holder.

        Onedata's fixed overhead (the probe transfer time, or the configured
        overhead if no probe between the providers was measured) plus the time to
        move ``size`` bytes.
        """
        measured = (self.provider_bandwidth_matrix or {}).get(to_provider, {})
        overheads = [
            PROBE_SIZE_BYTES / measured[h] for h in holders
            if measured.get(h) is not None and 0 < measured[h] < float("inf")
        ]
        config = getattr(self, "network_config", None)
        default_overhead = DEFAULT_REPLICATION_OVERHEAD_SECONDS
        bandwidth = REPLICATION_BANDWIDTH_BYTES_PER_SECOND
        if config is not None:
            if config.replication_overhead_seconds is not None:
                default_overhead = config.replication_overhead_seconds
            if config.replication_bandwidth_bytes_per_second is not None:
                bandwidth = config.replication_bandwidth_bytes_per_second
        overhead = min(overheads) if overheads else default_overhead
        return overhead + size / bandwidth

    def _finishes_in_time(self, c: 'Kairos.transition_candidate', now: datetime) -> bool:
        """Whether the replication of ``c`` completes before its task is predicted to start."""
        if c.predicted_time is None:
            return False
        t_rep = self._replication_time(c.to_provider, c.holders, c.size)
        logger.debug("Estimated replication time of file %s to %s: %.1f s", c.file_id, c.to_provider, t_rep)
        return now + timedelta(seconds=t_rep) <= c.predicted_time

    def _execute_transitions(self, candidates: list['Kairos.transition_candidate'], budget: float) -> float:
        """Execute the candidates greedily by descending VCR within ``budget``; returns the remaining budget.

        Equal VCRs are ordered by the topological position of the reading task. A
        candidate is executed only if its replication finishes before the task's
        predicted start and its cost fits the remaining budget.
        """
        candidates = sorted(candidates, key=lambda c: (-c.value_cost_ration, c.task_order))
        for i, c in enumerate(candidates):
            if budget <= 0:
                logger.info("Budget exhausted; %d remaining transitions are not executed.", len(candidates) - i)
                break
            if not self._finishes_in_time(c, datetime.now()):
                logger.info(
                    "Replication of file %s to %s would not finish before the predicted start %s. Skipping.",
                    c.file_id,
                    c.to_provider,
                    c.predicted_time,
                )
                continue
            if c.cost > budget:
                logger.info(
                    "Insufficient budget to transition file %s to %s (Cost: %.2f $, Remaining Budget: %.2f $)",
                    c.file_id,
                    c.to_provider,
                    c.cost,
                    budget,
                )
                continue
            logger.info(
                "Transitioning file %s to provider %s with value/cost ratio %.2f (value: %.2f ms, cost: %.2f $)",
                c.file_id,
                c.to_provider,
                c.value_cost_ration,
                c.value,
                c.cost,
            )
            with _replica_lock:
                _protect_replica(c.file_id, c.to_provider, c.predicted_time)
                self.onedata_manager.replicate_to_provider(c.file_id, c.to_provider)
            budget -= c.cost
        return budget

    def _remote_read_latency(self, reader_id: str, holder_ids: list[str]) -> float | None:
        """Latency of reading a file through ``reader_id`` from the closest of ``holder_ids``.

        Uses the measured provider-to-provider read latencies. For pairs without
        a measurement the path is estimated via Kairos: reader -> Kairos -> holder.
        """
        measured = (self.provider_latency_matrix or {}).get(reader_id, {})
        latencies = [measured[h] for h in holder_ids if measured.get(h) is not None]
        if latencies:
            return min(latencies)

        reader_latency = self.latency_map.get(reader_id)
        holder_latencies = [self.latency_map[h] for h in holder_ids if self.latency_map.get(h) is not None]
        if reader_latency is None or not holder_latencies:
            return None
        return reader_latency + min(holder_latencies)

    def _current_prediction(self, task_id: str, dag_id: str, parent_start: datetime) -> TaskPrediction | None:
        """Predict the execution of a child task in the DAG run that started at ``parent_start``.

        Predictions made at the start of the horizon can be several DAG runs
        old when an interrupt arrives.
        """
        if self.task_predictor is None:
            return None
        try:
            return self.task_predictor.predict_next_task_execution(
                task_id, dag_id, now=parent_start
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Re-predicting task %s in DAG %s failed: %s", task_id, dag_id, exc)
            return None

    def _on_interrupt(
        self,
        data: dict,
        predictions: dict[str, TaskPrediction],
        file_paths: dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]],
        horizon_time_seconds: float,
        budget: dict[str, float],
    ) -> None:
        """Replicate the dynamic files of a finished task to its children's predicted providers.

        The replications are ranked by VCR and executed within ``budget["remaining"]``,
        which is shared by all interrupts of the cycle.
        """
        logger.info("Received interrupt via Zenoh: %s at time %s", data, time.time())
        task_id = data.get("task_id")
        dag_id = data.get("dag_id")
        task_start = data.get("task_start")
        parent_start = datetime.fromtimestamp(task_start) if task_start is not None else datetime.now()
        horizon_hours = horizon_time_seconds / 3600.0

        if task_id not in file_paths:
            return
        _, dynamic_files, _ = file_paths[task_id]
        logger.info(
            "Dynamic files for task %s in DAG %s: %s",
            task_id,
            dag_id,
            dynamic_files,
        )
        if not dynamic_files:
            logger.warning(
                "No dynamic files found for task %s in DAG %s. No replication needed.",
                task_id,
                dag_id,
            )
            return

        candidates: list[Kairos.transition_candidate] = []
        for file_path, child_task_ids in dynamic_files.items():
            # The task re-created its output files, so they are newer than its start.
            file_id = self._get_file_id_with_retry(file_path, written_after=task_start)
            if file_id is None:
                continue
            for child_task_id in child_task_ids:
                predictions_for_child = self._current_prediction(
                    child_task_id, dag_id, parent_start
                ) or predictions.get(f"{child_task_id}_{dag_id}")
                if not predictions_for_child or not predictions_for_child.predicted_provider_id:
                    logger.warning(
                        "No predicted provider for child task %s in DAG %s. Skipping replication for file %s.",
                        child_task_id,
                        dag_id,
                        file_id,
                    )
                    continue
                candidate = self._transition_candidate_for(
                    file_id,
                    predictions_for_child.predicted_provider_id,
                    predictions_for_child.predicted_next_access,
                    horizon_hours,
                )
                if candidate is not None:
                    candidates.append(candidate)

        # The listener handles interrupts on a single thread, so updating the shared budget is safe.
        budget["remaining"] = self._execute_transitions(candidates, budget["remaining"])

    def dynamic_orchestration(
        self,
        workflow_dag: 'Kairos.dag',
        horizon_time_seconds: float,
        predictions: dict[str, TaskPrediction],
        file_paths: dict[str, tuple[list[str], dict[str, list[str]], dict[str, list[str]]]],
        horizon_cost_budget: float = float("inf"),
    ) -> float:
        """Perform dynamic orchestration based on the current configuration.

        Reacts to Airflow task-completion interrupts (``airflow_interrupts``
        Zenoh key expression) with this cycle's predictions and file paths, and
        waits until the end of the planning horizon.

        Returns:
            float: The budget remaining at the end of the horizon.
        """
        logger.info(
            "Starting dynamic orchestration for %s - Time is %s, for a maximum of %s seconds",
            workflow_dag.dag_id,
            time.time(),
            horizon_time_seconds,
        )

        budget = {"remaining": horizon_cost_budget}
        # Interrupts of this DAG go to this cycle from now on, also while the
        # next cycle ingests and plans, until that cycle registers itself.
        handler = lambda data: self._on_interrupt(data, predictions, file_paths, horizon_time_seconds, budget)
        if self.interrupt_listener is not None:
            self.interrupt_listener.set_handler(workflow_dag.dag_id, handler)
            time.sleep(max(0.0, self.horizon_end_time - time.time()))
        else:
            with InterruptListener() as listener:
                listener.set_handler(workflow_dag.dag_id, handler)
                time.sleep(max(0.0, self.horizon_end_time - time.time()))

        logger.info("Finished dynamic orchestration for %s - Time is %s", workflow_dag.dag_id, time.time())
        return budget["remaining"]


if __name__ == "__main__":
    """Test script for get_static_dynamic_semiDynamic_file_paths method."""
    import sys
    from unittest.mock import Mock, MagicMock
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create mock objects
    mock_db_manager = Mock()
    mock_provider_managers = {}
    mock_zone_manager = Mock()
    mock_onedata_manager = Mock()
    
    # Initialize Kairos instance with mocks
    kairos = Kairos(
        provider_managers=mock_provider_managers,
        zone_manager=mock_zone_manager,
        onedata_manager=mock_onedata_manager,
        db_manager=mock_db_manager,
    )
    
    # Create test DAG nodes
    node1 = Kairos.dag_node("task_1")
    node2 = Kairos.dag_node("task_2")
    node3 = Kairos.dag_node("task_3")
    
    # Create test DAG edges
    edge1 = Kairos.dag_edge(node1, node2)
    edge2 = Kairos.dag_edge(node2, node3)
    
    # Create test DAG
    test_dag = Kairos.dag("test_dag_id", [node1, node2, node3], [edge1, edge2])
    
    # Mock the database manager methods
    mock_db_manager.read_paths_of_read_accessed_files_by_task.side_effect = lambda task_id, dag_id: {
        "task_1": ["file_a.txt"],
        "task_2": ["file_b.txt", "file_e.txt"],
        "task_3": ["file_b.txt", "file_c.txt", "file_d.txt"],
    }.get(task_id, [])
    
    mock_db_manager.read_paths_of_write_accessed_files_by_task.side_effect = lambda task_id, dag_id: {
        "task_1": ["file_b.txt"],
        "task_2": ["file_c.txt"],
        "task_3": ["file_f.txt"],
    }.get(task_id, [])
    
    # Call the method
    print("Testing get_static_dynamic_semiDynamic_file_paths method...")
    result = kairos.get_static_dynamic_semiDynamic_file_paths(test_dag)
    
    # Print results
    print("\nResults:")
    for task_id, (static, dynamic, semi_dynamic) in result.items():
        print(f"\nTask: {task_id}")
        print(f"  Static read files: {static}")
        print(f"  Dynamic output files: {dynamic}")
        print(f"  Semi-dynamic output files: {semi_dynamic}")
    
    print("\nTest completed successfully!")

