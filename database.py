"""
database.py
-----------
Manages the SQLite database for storing experiment results.

Schema
------
experiments
    id              INTEGER PRIMARY KEY
    date            TEXT
    algorithm       TEXT
    bandwidth       REAL
    traffic_load    REAL
    congestion_level TEXT
    simulation_time REAL

results
    id              INTEGER PRIMARY KEY
    experiment_id   INTEGER  (FK → experiments.id)
    traffic_type    TEXT
    avg_latency     REAL     (ms)
    jitter          REAL     (ms)
    packet_loss     REAL     (%)
    throughput      REAL     (Mbps)
    bw_utilization  REAL     (%)
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "qos_results.db")


def _get_connection():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # allows dict-like access
    return conn


def initialise_db():
    """Create tables if they do not already exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT,
            algorithm       TEXT,
            bandwidth       REAL,
            traffic_load    REAL,
            congestion_level TEXT,
            simulation_time REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id   INTEGER,
            traffic_type    TEXT,
            avg_latency     REAL,
            jitter          REAL,
            packet_loss     REAL,
            throughput      REAL,
            bw_utilization  REAL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)

    conn.commit()
    conn.close()


def save_experiment(
    algorithm: str,
    bandwidth: float,
    traffic_load: float,
    congestion_level: str,
    simulation_time: float,
    metrics_result: dict,
) -> int:
    """
    Save an experiment and its results to the database.

    Parameters
    ----------
    metrics_result : dict with keys "per_type" and "overall"
        As returned by simulation.run_simulation()

    Returns
    -------
    int : The new experiment id
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # Insert experiment record
    cursor.execute("""
        INSERT INTO experiments
            (date, algorithm, bandwidth, traffic_load, congestion_level, simulation_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        algorithm,
        bandwidth,
        traffic_load,
        congestion_level,
        simulation_time,
    ))
    experiment_id = cursor.lastrowid

    # Insert per-type results
    for traffic_type, m in metrics_result["per_type"].items():
        cursor.execute("""
            INSERT INTO results
                (experiment_id, traffic_type, avg_latency, jitter,
                 packet_loss, throughput, bw_utilization)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            traffic_type,
            m["avg_latency"],
            m["jitter"],
            m["packet_loss"],
            m["throughput"],
            m["bw_utilization"],
        ))

    # Insert overall result with traffic_type = "Overall"
    m = metrics_result["overall"]
    cursor.execute("""
        INSERT INTO results
            (experiment_id, traffic_type, avg_latency, jitter,
             packet_loss, throughput, bw_utilization)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        experiment_id,
        "Overall",
        m["avg_latency"],
        m["jitter"],
        m["packet_loss"],
        m["throughput"],
        m["bw_utilization"],
    ))

    conn.commit()
    conn.close()
    return experiment_id


def get_all_experiments() -> list:
    """Return all experiments as a list of dicts."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_results_for_experiment(experiment_id: int) -> list:
    """Return all result rows for a given experiment."""
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM results WHERE experiment_id = ?", (experiment_id,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_comparison_data(
    bandwidth: float,
    traffic_load: float,
    congestion_level: str,
    simulation_time: float,
) -> list:
    """
    Fetch the most recent result for each algorithm matching the given params.
    Returns a list of dicts suitable for building a comparison table.
    """
    conn = _get_connection()
    cursor = conn.cursor()

    algorithms = ["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"]
    comparison = []

    for algo in algorithms:
        # Get the most recent experiment matching these parameters
        cursor.execute("""
            SELECT e.id FROM experiments e
            WHERE e.algorithm = ?
              AND e.bandwidth = ?
              AND e.traffic_load = ?
              AND e.congestion_level = ?
              AND e.simulation_time = ?
            ORDER BY e.id DESC
            LIMIT 1
        """, (algo, bandwidth, traffic_load, congestion_level, simulation_time))

        row = cursor.fetchone()
        if row:
            exp_id = row["id"]
            cursor.execute("""
                SELECT * FROM results
                WHERE experiment_id = ? AND traffic_type = 'Overall'
            """, (exp_id,))
            result_row = cursor.fetchone()
            if result_row:
                d = dict(result_row)
                d["algorithm"] = algo
                comparison.append(d)

    conn.close()
    return comparison
