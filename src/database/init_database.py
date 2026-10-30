from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "kairos.sqlite3"


def initialize_database(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS Dags (
                Id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS Dag_Executions (
                Id TEXT PRIMARY KEY,
                Dag_Id TEXT NOT NULL,
                FOREIGN KEY (Dag_Id) REFERENCES Dags(Id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Tasks (
                Id TEXT,
                Dag_Id TEXT,
                PRIMARY KEY (Id, Dag_Id),
                FOREIGN KEY (Dag_Id) REFERENCES Dags(Id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Task_Executions (
                Id TEXT PRIMARY KEY,
                Task_Id TEXT NOT NULL,
                Task_Dag_Id TEXT NOT NULL,
                Dag_Execution_Id TEXT NOT NULL,
                FOREIGN KEY (Task_Id, Task_Dag_Id) REFERENCES Tasks(Id, Dag_Id) ON DELETE CASCADE,
                FOREIGN KEY (Dag_Execution_Id) REFERENCES Dag_Executions(Id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Files (
                Id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS Providers (
                Id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS Spaces (
                Id TEXT PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS Accesses (
                Id TEXT PRIMARY KEY,
                File_Id TEXT NOT NULL,
                Access_Type TEXT NOT NULL,
                File_Path TEXT NOT NULL,
                Provider_Id TEXT NOT NULL,
                Space_Id TEXT NOT NULL,
                Timepoint TEXT NOT NULL,
                Duration_Seconds REAL,
                Task_Execution_Id TEXT,
                FOREIGN KEY (File_Id) REFERENCES Files(Id) ON DELETE CASCADE,
                FOREIGN KEY (Provider_Id) REFERENCES Providers(Id) ON DELETE CASCADE,
                FOREIGN KEY (Space_Id) REFERENCES Spaces(Id) ON DELETE CASCADE,
                FOREIGN KEY (Task_Execution_Id) REFERENCES Task_Executions(Id) ON DELETE SET NULL
            );
            """)

        cursor = connection.execute("PRAGMA table_info(Accesses);")
        columns = [row[1] for row in cursor.fetchall()]
        if columns:
            if "Access_Type" not in columns:
                connection.execute(
                    "ALTER TABLE Accesses ADD COLUMN Access_Type TEXT NOT NULL DEFAULT 'read';"
                )
            if "File_Path" not in columns:
                connection.execute(
                    "ALTER TABLE Accesses ADD COLUMN File_Path TEXT NOT NULL DEFAULT '';"
                )

def delete_database(db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()


if __name__ == "__main__":
    initialize_database()
