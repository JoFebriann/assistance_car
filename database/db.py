import sqlite3
from pathlib import Path
from config.settings import DB_PATH, SCHEMA_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database(schema_path: Path | None = None):
    """
    Initialize database using schema.sql
    """
    if schema_path is None:
        schema_path = SCHEMA_PATH

    conn = get_connection()
    cur = conn.cursor()

    with open(schema_path, "r", encoding="utf-8") as f:
        cur.executescript(f.read())

    conn.commit()
    conn.close()
    
def reset_database():

    conn = get_connection()
    cur = conn.cursor()

    # Delete child tables first so FK relationships are respected
    cur.execute("DELETE FROM detections")
    cur.execute("DELETE FROM optical_flow")
    cur.execute("DELETE FROM lane_metrics")
    cur.execute("DELETE FROM scene_metrics")
    cur.execute("DELETE FROM frames")

    # Reset autoincrement counter for detections so IDs restart from 1
    cur.execute(
        "DELETE FROM sqlite_sequence WHERE name='detections'"
    )

    conn.commit()
    conn.close()