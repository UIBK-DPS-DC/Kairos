from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from src.database.db_interface import (
    Access,
    AccessType,
    Dag,
    DBInterface,
    Task_Execution,
    File,
    Task,
    Dag_Execution,
    Provider,
    Space,
)
from src.database.init_database import DB_PATH as DEFAULT_DB_PATH
from src.database.init_database import initialize_database, delete_database

import logging

logger = logging.getLogger(__name__)


def _access_type_from_value(value: str | None) -> AccessType | None:
    if value is None:
        return None
    try:
        return AccessType(value)
    except ValueError:
        return None


def _resolve_db_path(db_path: str | Path | None) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH

    resolved = str(db_path).strip()
    if not resolved:
        return DEFAULT_DB_PATH

    if resolved.startswith("sqlite:///"):
        resolved = resolved.removeprefix("sqlite:///")

    return Path(resolved)




class SQLite3Manager(DBInterface):
    """SQLite-backed implementation of the DB cache interface."""

    def __init__(self, db_path: str | Path = None, initialize_db: bool = True):
        self.db_path = _resolve_db_path(db_path)
        if initialize_db:
            logger.info("Initializing database at %s and therefore deleting old one", self.db_path)
            delete_database(self.db_path)
            initialize_database(self.db_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON;")
            yield connection

    def _row_exists(self, table: str, record_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT 1 FROM {table} WHERE Id = ?", (record_id,)
            ).fetchone()
        return row is not None

    def _read_single_value(self, query: str, params: tuple[str, ...]) -> str | None:
        with self._connection() as connection:
            row = connection.execute(query, params).fetchone()
        if row is None:
            return None
        return row[0]

    # DAG methods

    def insert_dag(self, dag: Dag) -> None:
        with self._connection() as connection:
            connection.execute("INSERT OR IGNORE INTO Dags (Id) VALUES (?)", (dag.id,))

    def read_dag(self, dag_id: str) -> Dag | None:
        dag_id_value = self._read_single_value(
            "SELECT * FROM Dags WHERE Id = ?", (dag_id,)
        )
        return Dag(dag_id_value) if dag_id_value is not None else None

    def delete_dag(self, dag_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM Dags WHERE Id = ?", (dag_id,))

    def read_all_dags(self) -> List[Dag]:
        with self._connection() as connection:
            rows = connection.execute("SELECT Id FROM Dags").fetchall()
        return [Dag(row[0]) for row in rows]

    # Dag execution methods

    def insert_dag_execution(self, execution: Dag_Execution) -> None:
        self.insert_dag(Dag(execution.dag_id))
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Dag_Executions (Id, Dag_Id) VALUES (?, ?)",
                (execution.id, execution.dag_id),
            )

    def read_dag_execution(self, execution_id: str) -> Dag_Execution | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM Dag_Executions WHERE Id = ?", (execution_id,)
            ).fetchone()
        if row is None:
            return None
        return Dag_Execution(id=row[0], dag_id=row[1])

    def update_dag_execution(self, execution: Dag_Execution) -> None:
        self.insert_dag(Dag(execution.dag_id))
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE Dag_Executions SET Dag_Id = ? WHERE Id = ?",
                (execution.dag_id, execution.id),
            ).rowcount
            if updated == 0:
                connection.execute(
                    "INSERT OR REPLACE INTO Dag_Executions (Id, Dag_Id) VALUES (?, ?)",
                    (execution.id, execution.dag_id),
                )

    def delete_dag_execution(self, execution_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM Dag_Executions WHERE Id = ?", (execution_id,)
            )

    def read_all_dag_executions(self) -> List[Dag_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, Dag_Id FROM Dag_Executions"
            ).fetchall()
        return [Dag_Execution(id=row[0], dag_id=row[1]) for row in rows]

    # Task methods

    def insert_task(self, task: Task) -> None:
        self.insert_dag(Dag(task.dag_id))
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Tasks (Id, Dag_Id) VALUES (?, ?)",
                (task.id, task.dag_id),
            )

    def read_task(self, task_id: str, dag_id: str) -> Task | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM Tasks WHERE Id = ? AND Dag_Id = ?", (task_id, dag_id)
            ).fetchone()
        if row is None:
            return None
        return Task(id=row[0], dag_id=row[1])

    def update_task(self, task: Task) -> None:
        self.insert_dag(Dag(task.dag_id))
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE Tasks SET Dag_Id = ? WHERE Id = ?", (task.dag_id, task.id)
            ).rowcount
            if updated == 0:
                connection.execute(
                    "INSERT OR REPLACE INTO Tasks (Id, Dag_Id) VALUES (?, ?)",
                    (task.id, task.dag_id),
                )

    def delete_task(self, task_id: str, dag_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM Tasks WHERE Id = ? AND Dag_Id = ?", (task_id, dag_id)
            )

    def read_all_tasks(self) -> List[Task]:
        with self._connection() as connection:
            rows = connection.execute("SELECT Id, Dag_Id FROM Tasks").fetchall()
        return [Task(id=row[0], dag_id=row[1]) for row in rows]

    # Task execution methods

    def insert_task_execution(self, execution: Task_Execution) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Task_Executions (Id, Task_Id, Task_Dag_Id, Dag_Execution_Id) VALUES (?, ?, ?, ?)",
                (
                    execution.id,
                    execution.task_id,
                    execution.task_dag_id,
                    execution.dag_execution_id,
                ),
            )

    def read_task_execution(self, execution_id: str) -> Task_Execution | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM Task_Executions WHERE Id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            return None
        return Task_Execution(
            id=row[0], task_id=row[1], task_dag_id=row[2], dag_execution_id=row[3]
        )

    def update_task_execution(self, execution: Task_Execution) -> None:
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE Task_Executions SET Task_Id = ?, Task_Dag_Id = ?, Dag_Execution_Id = ? WHERE Id = ?",
                (
                    execution.task_id,
                    execution.task_dag_id,
                    execution.dag_execution_id,
                    execution.id,
                ),
            ).rowcount
            if updated == 0:
                connection.execute(
                    "INSERT OR REPLACE INTO Task_Executions (Id, Task_Id, Task_Dag_Id, Dag_Execution_Id) VALUES (?, ?, ?, ?)",
                    (
                        execution.id,
                        execution.task_id,
                        execution.task_dag_id,
                        execution.dag_execution_id,
                    ),
                )

    def delete_task_execution(self, execution_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM Task_Executions WHERE Id = ?", (execution_id,)
            )

    def read_all_task_executions(self) -> List[Task_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, Task_Id, Task_Dag_Id, Dag_Execution_Id FROM Task_Executions"
            ).fetchall()
        return [
            Task_Execution(
                id=row[0], task_id=row[1], task_dag_id=row[2], dag_execution_id=row[3]
            )
            for row in rows
        ]

    # File methods

    def insert_file(self, file: File) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Files (Id) VALUES (?)", (file.id,)
            )

    def read_file(self, file_id: str) -> File | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM Files WHERE Files.Id = ?
                """,
                (file_id,),
            ).fetchone()
        if row is None:
            return None
        return File(id=row[0])

    def delete_file(self, file_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM Files WHERE Id = ?", (file_id,))

    def read_all_files(self) -> List[File]:
        with self._connection() as connection:
            rows = connection.execute("SELECT Id FROM Files").fetchall()
        return [File(id=row[0]) for row in rows]

    # Provider methods

    def insert_provider(self, provider: Provider) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Providers (Id) VALUES (?)", (provider.id,)
            )

    def read_provider(self, provider_id: str) -> Provider | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT Id FROM Providers WHERE Id = ?",
                (provider_id,),
            ).fetchone()
        if row is None:
            return None
        return Provider(id=row[0])

    def delete_provider(self, provider_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM Providers WHERE Id = ?", (provider_id,))

    def read_all_providers(self) -> List[Provider]:
        with self._connection() as connection:
            rows = connection.execute("SELECT Id FROM Providers").fetchall()
        return [Provider(id=row[0]) for row in rows]

    # Space methods

    def insert_space(self, space: Space) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO Spaces (Id) VALUES (?)", (space.id,)
            )

    def read_space(self, space_id: str) -> Space | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT Id FROM Spaces WHERE Id = ?",
                (space_id,),
            ).fetchone()
        if row is None:
            return None
        return Space(id=row[0])

    def delete_space(self, space_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM Spaces WHERE Id = ?", (space_id,))

    def read_all_spaces(self) -> List[Space]:
        with self._connection() as connection:
            rows = connection.execute("SELECT Id FROM Spaces").fetchall()
        return [Space(id=row[0]) for row in rows]

    # Access methods

    def insert_access(self, access: Access) -> None:
        access_type_val = (
            access.access_type.value
            if isinstance(access.access_type, AccessType)
            else (str(access.access_type) if access.access_type is not None else None)
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO Accesses (
                    Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(Id) DO UPDATE SET
                    Timepoint = MIN(Timepoint, excluded.Timepoint),
                    Duration_Seconds = MAX(
                        COALESCE(Duration_Seconds, excluded.Duration_Seconds),
                        COALESCE(excluded.Duration_Seconds, Duration_Seconds)
                    )
                """,
                (
                    access.id,
                    access.file_id,
                    access_type_val,
                    access.file_path,
                    access.provider_id,
                    access.space_id,
                    access.timepoint,
                    access.duration_seconds,
                    access.task_execution_id,
                ),
            )

    def read_access(self, access_id: str) -> Access | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id
                FROM Accesses
                WHERE Id = ?
                """,
                (access_id,),
            ).fetchone()
        if row is None:
            return None
        return Access(
            id=row[0],
            file_id=row[1],
            access_type=_access_type_from_value(row[2]),
            file_path=row[3],
            provider_id=row[4],
            space_id=row[5],
            timepoint=row[6],
            duration_seconds=row[7],
            task_execution_id=row[8],
        )

    def update_access(self, access: Access) -> None:
        access_type_val = (
            access.access_type.value
            if isinstance(access.access_type, AccessType)
            else (str(access.access_type) if access.access_type is not None else None)
        )
        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE Accesses
                SET File_Id = ?, Access_Type = ?, File_Path = ?, Provider_Id = ?, Space_Id = ?, Timepoint = ?, Duration_Seconds = ?, Task_Execution_Id = ?
                WHERE Id = ?
                """,
                (
                    access.file_id,
                    access_type_val,
                    access.file_path,
                    access.provider_id,
                    access.space_id,
                    access.timepoint,
                    access.duration_seconds,
                    access.task_execution_id,
                    access.id,
                ),
            ).rowcount
            if updated == 0:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO Accesses (
                        Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        access.id,
                        access.file_id,
                        access_type_val,
                        access.file_path,
                        access.provider_id,
                        access.space_id,
                        access.timepoint,
                        access.duration_seconds,
                        access.task_execution_id,
                    ),
                )

    def delete_access(self, access_id: str) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM Accesses WHERE Id = ?", (access_id,))

    def read_all_accesses(self) -> List[Access]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id FROM Accesses"
            ).fetchall()
        return [
            Access(
                id=row[0],
                file_id=row[1],
                access_type=_access_type_from_value(row[2]),
                file_path=row[3],
                provider_id=row[4],
                space_id=row[5],
                timepoint=row[6],
                duration_seconds=row[7],
                task_execution_id=row[8],
            )
            for row in rows
        ]

    def read_x_accesses(self, x: int) -> List[Access]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id FROM Accesses LIMIT ?",
                (x,),
            ).fetchall()
        return [
            Access(
                id=row[0],
                file_id=row[1],
                access_type=_access_type_from_value(row[2]),
                file_path=row[3],
                provider_id=row[4],
                space_id=row[5],
                timepoint=row[6],
                duration_seconds=row[7],
                task_execution_id=row[8],
            )
            for row in rows
        ]

    def insert_accesses(self, accesses: Iterable[Access]) -> int:
        written = 0
        for access in accesses:
            self.insert_access(access)
            written += 1
        return written

    def read_accesses_by_file(self, file_id: str) -> List[Access]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id FROM Accesses WHERE File_Id = ?",
                (file_id,),
            ).fetchall()
        return [
            Access(
                id=row[0],
                file_id=row[1],
                access_type=_access_type_from_value(row[2]),
                file_path=row[3],
                provider_id=row[4],
                space_id=row[5],
                timepoint=row[6],
                duration_seconds=row[7],
                task_execution_id=row[8],
            )
            for row in rows
        ]

    def read_accesses_by_task_execution(self, task_execution_id: str) -> List[Access]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT Id, File_Id, Access_Type, File_Path, Provider_Id, Space_Id, Timepoint, Duration_Seconds, Task_Execution_Id FROM Accesses WHERE Task_Execution_Id = ?",
                (task_execution_id,),
            ).fetchall()
        return [
            Access(
                id=row[0],
                file_id=row[1],
                access_type=_access_type_from_value(row[2]),
                file_path=row[3],
                provider_id=row[4],
                space_id=row[5],
                timepoint=row[6],
                duration_seconds=row[7],
                task_execution_id=row[8],
            )
            for row in rows
        ]

    def read_accesses_by_task(self, task_id, dag_id) -> List[Access]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.Id, a.File_Id, a.Access_Type, a.File_Path, a.Provider_Id, a.Space_Id, a.Timepoint, a.Duration_Seconds, a.Task_Execution_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                """,
                (task_id, dag_id),
            ).fetchall()
        return [
            Access(
                id=row[0],
                file_id=row[1],
                access_type=_access_type_from_value(row[2]),
                file_path=row[3],
                provider_id=row[4],
                space_id=row[5],
                timepoint=row[6],
                duration_seconds=row[7],
                task_execution_id=row[8],
            )
            for row in rows
        ]

    def read_task_executions_by_task(
        self, task_id: str, dag_id: str
    ) -> List[Task_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT Id, Task_Id, Task_Dag_Id, Dag_Execution_Id
                FROM Task_Executions
                WHERE Task_Id = ? AND Task_Dag_Id = ?
                """,
                (task_id, dag_id),
            ).fetchall()
        return [
            Task_Execution(
                id=row[0],
                task_id=row[1],
                task_dag_id=row[2],
                dag_execution_id=row[3],
            )
            for row in rows
        ]

    def read_last_x_task_executions_by_task(
        self, task_id: str, dag_id: str, x: int
    ) -> List[Task_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT Id, Task_Id, Task_Dag_Id, Dag_Execution_Id
                FROM Task_Executions
                WHERE Task_Id = ? AND Task_Dag_Id = ?
                ORDER BY Id DESC
                LIMIT ?
                """,
                (task_id, dag_id, x),
            ).fetchall()
        return [
            Task_Execution(
                id=row[0],
                task_id=row[1],
                task_dag_id=row[2],
                dag_execution_id=row[3],
            )
            for row in rows
        ]

    def read_providers_used_by_task(self, task_id, dag_id):
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT a.Provider_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                """,
                (task_id, dag_id),
            ).fetchall()
        return [row[0] for row in rows]

    def read_task_executions_by_provider_and_task(
        self, provider_id: str, task_id: str, dag_id: str
    ) -> List[Task_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT te.Id, te.Task_Id, te.Task_Dag_Id, te.Dag_Execution_Id
                FROM Task_Executions AS te
                JOIN Accesses AS a ON a.Task_Execution_Id = te.Id
                WHERE a.Provider_Id = ? AND te.Task_Id = ? AND te.Task_Dag_Id = ?
                """,
                (provider_id, task_id, dag_id),
            ).fetchall()
        return [
            Task_Execution(
                id=row[0],
                task_id=row[1],
                task_dag_id=row[2],
                dag_execution_id=row[3],
            )
            for row in rows
        ]

    def read_access_counts_by_task(
        self, task_id: str, dag_id: str
    ) -> List[tuple[str, str, int]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT te.Id, a.Provider_Id, COUNT(*)
                FROM Task_Executions AS te
                JOIN Accesses AS a ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                GROUP BY te.Id, a.Provider_Id
                """,
                (task_id, dag_id),
            ).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def read_first_access_times_by_task(
        self, task_id: str, dag_id: str
    ) -> Dict[str, str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT te.Id, MIN(a.Timepoint)
                FROM Task_Executions AS te
                JOIN Accesses AS a ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                GROUP BY te.Id
                """,
                (task_id, dag_id),
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def read_last_x_task_executions_by_provider_and_task(
        self, provider_id: str, task_id: str, dag_id: str, x: int
    ) -> List[Task_Execution]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT te.Id, te.Task_Id, te.Task_Dag_Id, te.Dag_Execution_Id
                FROM Task_Executions AS te
                JOIN Accesses AS a ON a.Task_Execution_Id = te.Id
                WHERE a.Provider_Id = ? AND te.Task_Id = ? AND te.Task_Dag_Id = ?
                ORDER BY te.Id DESC
                LIMIT ?
                """,
                (provider_id, task_id, dag_id, x),
            ).fetchall()
        return [
            Task_Execution(
                id=row[0],
                task_id=row[1],
                task_dag_id=row[2],
                dag_execution_id=row[3],
            )
            for row in rows
        ]

    def read_independent_accesses_by_task(self, task_id: str, dag_id: str, previous_task_id: str) -> List[str]:
        if previous_task_id is None:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT a.File_Id
                    FROM Accesses AS a
                    JOIN Task_Executions AS te
                    ON a.Task_Execution_Id = te.Id
                    WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                    AND te.Id = (
                        SELECT MAX(te2.Id)
                        FROM Task_Executions AS te2
                        WHERE te2.Task_Id = ? AND te2.Task_Dag_Id = ?
                    )
                    """,
                    (
                        task_id,
                        dag_id,
                        task_id,
                        dag_id,
                    ),).fetchall()

                return [row[0] for row in rows]

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.File_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te
                ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                AND te.Id = (
                    SELECT MAX(te2.Id)
                    FROM Task_Executions AS te2
                    WHERE te2.Task_Id = ? AND te2.Task_Dag_Id = ?
                )

                EXCEPT

                SELECT a.File_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te
                ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                AND te.Id = (
                    SELECT MAX(te2.Id)
                    FROM Task_Executions AS te2
                    WHERE te2.Task_Id = ? AND te2.Task_Dag_Id = ?
                )
                """,
                (
                    task_id,
                    dag_id,
                    task_id,
                    dag_id,
                    previous_task_id,
                    dag_id,
                    previous_task_id,
                    dag_id,
                ),).fetchall()

            return [row[0] for row in rows]

    def read_dependent_accesses_between_tasks(self, task_id: str, dag_id: str, previous_task_id: str) -> List[str]:
        if previous_task_id is None:
            return []

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.File_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te
                ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                AND te.Id = (
                    SELECT MAX(te2.Id)
                    FROM Task_Executions AS te2
                    WHERE te2.Task_Id = ? AND te2.Task_Dag_Id = ?
                )

                INTERSECT

                SELECT a.File_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te
                ON a.Task_Execution_Id = te.Id
                WHERE te.Task_Id = ? AND te.Task_Dag_Id = ?
                AND te.Id = (
                    SELECT MAX(te2.Id)
                    FROM Task_Executions AS te2
                    WHERE te2.Task_Id = ? AND te2.Task_Dag_Id = ?
                )
                """,
                (
                    task_id,
                    dag_id,
                    task_id,
                    dag_id,
                    previous_task_id,
                    dag_id,
                    previous_task_id,
                    dag_id,
                ),).fetchall()

            return [row[0] for row in rows]

    def read_paths_of_read_accessed_files_by_task(self, task_id: str, dag_id: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT a.File_Path
                FROM Accesses AS a
                JOIN Task_Executions AS te ON a.Task_Execution_Id = te.Id
                WHERE a.Access_Type = 'read' AND te.Task_Id = ? AND te.Task_Dag_Id = ?
                """,
                (task_id, dag_id),
            ).fetchall()
        return [row[0] for row in rows]

    def read_paths_of_write_accessed_files_by_task(self, task_id: str, dag_id: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT a.File_Path
                FROM Accesses AS a
                JOIN Task_Executions AS te ON a.Task_Execution_Id = te.Id
                WHERE a.Access_Type = 'write' AND te.Task_Id = ? AND te.Task_Dag_Id = ?
                """,
                (task_id, dag_id),
            ).fetchall()
        return [row[0] for row in rows]

    def read_file_ids_by_dag_execution_and_task(
        self, dag_execution_id: str, task_id: str, dag_id: str
    ) -> list[str]:
        """Return distinct file GUIDs accessed by a task in a specific dag execution.

        Used at interrupt time after re-ingesting Prometheus data so that the
        interrupt handler can discover the live file IDs for the current run.
        """
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT a.File_Id
                FROM Accesses AS a
                JOIN Task_Executions AS te ON a.Task_Execution_Id = te.Id
                WHERE te.Dag_Execution_Id = ?
                  AND te.Task_Id        = ?
                  AND te.Task_Dag_Id    = ?
                """,
                (dag_execution_id, task_id, dag_id),
            ).fetchall()
        return [row[0] for row in rows]