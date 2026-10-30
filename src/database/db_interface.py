from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum


class Dag:
    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return f"Dag(id={self.id})"


class Dag_Execution:
    def __init__(self, id: str, dag_id: str):
        self.id = id
        self.dag_id = dag_id

    def __str__(self):
        return f"Dag_Execution(id={self.id}, dag_id={self.dag_id})"


class Task:
    def __init__(self, id: str, dag_id: str):
        self.id = id
        self.dag_id = dag_id

    def __str__(self):
        return f"Task(id={self.id}, dag_id={self.dag_id})"


class Task_Execution:
    def __init__(
        self, id: str, task_id: str, task_dag_id: str, dag_execution_id: str = None
    ):
        self.id = id
        self.task_id = task_id
        self.task_dag_id = task_dag_id
        self.dag_execution_id = dag_execution_id

    def __str__(self):
        return f"Task_Execution(id={self.id}, task_id={self.task_id}, task_dag_id={self.task_dag_id}, dag_execution_id={self.dag_execution_id})"


class File:
    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return f"File(id={self.id})"


class Provider:
    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return f"Provider(id={self.id})"


class Space:
    def __init__(self, id: str):
        self.id = id

    def __str__(self):
        return f"Space(id={self.id})"


class AccessType(Enum):
    READ = "read"
    WRITE = "write"


class Access:
    def __init__(
        self,
        file_id: str,
        file_path: str,
        provider_id: str,
        space_id: str,
        timepoint: str,
        access_type: AccessType,
        duration_seconds: float = None,
        task_execution_id: str = None,
        id: str = None,
    ):
        self.id = id
        self.file_id = file_id
        self.file_path = file_path
        self.provider_id = provider_id
        self.space_id = space_id
        self.timepoint = timepoint
        self.duration_seconds = duration_seconds
        if isinstance(access_type, str):
            self.access_type = AccessType(access_type)
        else:
            self.access_type = access_type
        self.task_execution_id = task_execution_id

    def __str__(self):
        return f"Access(id={self.id}, file_id={self.file_id}, access_type={self.access_type}, file_path={self.file_path}, provider_id={self.provider_id}, space_id={self.space_id}, timepoint={self.timepoint}, duration_seconds={self.duration_seconds}, task_execution_id={self.task_execution_id})"


class DBInterface(ABC):
    """
    Abstract interface for DAG metric caching backends.
    It defines the methods to read and manipulate stored data. For that there are the following tables:
    - Dags: stores the DAGs that have been executed.
    - Tasks: stores the tasks that have been executed.
    - Executions: stores the executions of tasks.
    - Files: stores the files that have been accessed.
    - Providers: stores the providers of file access.
    - Spaces: stores the spaces where files are located.
    - Accesses: stores the access events for files.
    - Execution_Files: stores the relationship between executions and files.
    Each table has its own set of methods to insert, read, update, and delete records
    """

    # dag

    @abstractmethod
    def insert_dag(self, dag: Dag) -> None:
        pass

    @abstractmethod
    def read_dag(self, dag_id: str) -> Dag:
        pass

    @abstractmethod
    def delete_dag(self, dag_id: str) -> None:
        pass

    @abstractmethod
    def read_all_dags(self) -> List[Dag]:
        pass

        # task

    @abstractmethod
    def insert_task(self, task: Task) -> None:
        pass

    @abstractmethod
    def read_task(self, task_id: str, dag_id: str) -> Task:
        pass

    @abstractmethod
    def update_task(self, task: Task) -> None:
        pass

    @abstractmethod
    def delete_task(self, task_id: str, dag_id: str) -> None:
        pass

    @abstractmethod
    def read_all_tasks(self) -> List[Task]:
        pass

        # dag execution

    @abstractmethod
    def insert_dag_execution(self, execution: Dag_Execution) -> None:
        pass

    @abstractmethod
    def read_dag_execution(self, execution_id: str) -> Dag_Execution:
        pass

    @abstractmethod
    def update_dag_execution(self, execution: Dag_Execution) -> None:
        pass

    @abstractmethod
    def delete_dag_execution(self, execution_id: str) -> None:
        pass

    @abstractmethod
    def read_all_dag_executions(self) -> List[Dag_Execution]:
        pass

        # task execution

    @abstractmethod
    def insert_task_execution(self, execution: Task_Execution) -> None:
        pass

    @abstractmethod
    def read_task_execution(self, execution_id: str) -> Task_Execution:
        pass

    @abstractmethod
    def update_task_execution(self, execution: Task_Execution) -> None:
        pass

    @abstractmethod
    def delete_task_execution(self, execution_id: str) -> None:
        pass

    @abstractmethod
    def read_all_task_executions(self) -> List[Task_Execution]:
        pass

        # file

    @abstractmethod
    def insert_file(self, file: File) -> None:
        pass

    @abstractmethod
    def read_file(self, file_id: str) -> File:
        pass

    @abstractmethod
    def delete_file(self, file_id: str) -> None:
        pass

    @abstractmethod
    def read_all_files(self) -> List[File]:
        pass

        # provider

    @abstractmethod
    def insert_provider(self, provider_id: str) -> None:
        pass

    @abstractmethod
    def read_provider(self, provider_id: str) -> str:
        pass

    @abstractmethod
    def delete_provider(self, provider_id: str) -> None:
        pass

    @abstractmethod
    def read_all_providers(self) -> List[str]:
        pass

        # space

    @abstractmethod
    def insert_space(self, space_id: str) -> None:
        pass

    @abstractmethod
    def read_space(self, space_id: str) -> str:
        pass

    @abstractmethod
    def delete_space(self, space_id: str) -> None:
        pass

    @abstractmethod
    def read_all_spaces(self) -> List[str]:
        pass

        # access

    @abstractmethod
    def insert_access(self, access: Access) -> None:
        pass

    @abstractmethod
    def read_access(self, access_id: str) -> Access:
        pass

    @abstractmethod
    def update_access(self, access: Access) -> None:
        pass

    @abstractmethod
    def delete_access(self, access_id: str) -> None:
        pass

    @abstractmethod
    def read_all_accesses(self) -> List[Access]:
        pass

    @abstractmethod
    def read_x_accesses(self, x: int) -> List[Access]:
        pass
    
    # further methods for prediction purposes

    @abstractmethod
    def read_accesses_by_file(self, file_id: str) -> List[Access]:
        pass

    @abstractmethod
    def read_accesses_by_task_execution(self, task_execution_id: str) -> List[Access]:
        pass

    @abstractmethod
    def read_accesses_by_task(self, task_id: str, dag_id: str) -> List[Access]:
        pass

    @abstractmethod
    def read_task_executions_by_task(
        self, task_id: str, dag_id: str
    ) -> List[Task_Execution]:
        pass

    @abstractmethod
    def read_last_x_task_executions_by_task(
        self, task_id: str, dag_id: str, x: int
    ) -> List[Task_Execution]:
        pass

    @abstractmethod
    def read_providers_used_by_task(self, task_id: str, dag_id: str) -> List[str]:
        pass

    @abstractmethod
    def read_task_executions_by_provider_and_task(
        self, provider_id: str, task_id: str, dag_id: str
    ) -> List[Task_Execution]:
        pass

    @abstractmethod
    def read_access_counts_by_task(
        self, task_id: str, dag_id: str
    ) -> List[tuple[str, str, int]]:
        """(task_execution_id, provider_id, number of accesses) for every execution of a task."""
        pass

    @abstractmethod
    def read_first_access_times_by_task(
        self, task_id: str, dag_id: str
    ) -> Dict[str, str]:
        """task_execution_id -> Timepoint (ISO format) of its earliest access, for every execution of a task."""
        pass

    @abstractmethod
    def read_last_x_task_executions_by_provider_and_task(
        self, provider_id: str, task_id: str, dag_id: str, x: int
    ) -> List[Task_Execution]:
        pass

    @abstractmethod
    def read_independent_accesses_by_task(self, task_id: str, dag_id: str, previous_task_id: str) -> List[str]:
        pass

    @abstractmethod
    def read_dependent_accesses_between_tasks(self, task_id: str, dag_id: str, previous_task_id: str) -> List[str]:
        pass

    @abstractmethod
    def read_paths_of_read_accessed_files_by_task(
        self, task_id: str, dag_id: str
    ) -> list[str]:
        """Return the distinct file paths accessed during a specific task execution.

        This is used at interrupt time after re-ingesting Prometheus data: given the
        run_id (= dag_execution_id) and the just-completed task_id, retrieve all file paths
        that were touched in that particular run so we can replicate them immediately.
        """
        pass

    @abstractmethod
    def read_paths_of_write_accessed_files_by_task(
        self, task_id: str, dag_id: str
    ) -> list[str]:
        """Return the distinct file paths accessed during a specific task execution.

        This is used at interrupt time after re-ingesting Prometheus data: given the
        run_id (= dag_execution_id) and the just-completed task_id, retrieve all file paths
        that were touched in that particular run so we can replicate them immediately.
        """
        pass

    @abstractmethod
    def read_file_ids_by_dag_execution_and_task(
        self, dag_execution_id: str, task_id: str, dag_id: str
    ) -> List[str]:
        """Return the distinct file GUIDs accessed during a specific dag execution by a given task.

        This is used at interrupt time after re-ingesting Prometheus data: given the
        run_id (= dag_execution_id) and the just-completed task_id, retrieve all file IDs
        that were touched in that particular run so we can replicate them immediately.
        """
        pass