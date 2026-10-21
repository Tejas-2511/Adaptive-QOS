"""
simulation.py
-------------
Handles packet generation and the main simulation loop.

Implements high-performance vectorized scheduling based on the Adaptive QoS
technical reference (using parallel NumPy arrays for speed and fidelity),
while maintaining full compatibility with app.py and database.py.
"""

import math
import random
from typing import Dict, List, Tuple, Any
import numpy as np

import qos_algorithms as qa
from qos_algorithms import (
    CLASSES,
    SIZE_RANGE,
    WFQ_W,
    CBWFQ_S,
    MIXES,
    MULT,
    INIT_SCORE,
    BUFFER,
    ADAPT_W,
    generate,
    fifo,
    pq,
    wfq,
    cbwfq,
    adaptive,
    metrics,
    run_all,
)

# ---------------------------------------------------------------------------
# Traffic type definitions (preserved for app.py compatibility)
# ---------------------------------------------------------------------------

TRAFFIC_TYPES = {
    "Video Conferencing": {
        "priority": 4,              # highest
        "size_range": (100, 300),   # bytes – small real-time packets
        "weight": 4,
        "cbwfq_share": 0.35,
    },
    "Gaming": {
        "priority": 3,
        "size_range": (50, 200),
        "weight": 3,
        "cbwfq_share": 0.30,
    },
    "Video Streaming": {
        "priority": 2,
        "size_range": (800, 1500),  # larger chunks
        "weight": 2,
        "cbwfq_share": 0.20,
    },
    "Background": {
        "priority": 1,              # lowest
        "size_range": (500, 1400),
        "weight": 1,
        "cbwfq_share": 0.15,
    },
}

# Default traffic mix proportions
TRAFFIC_MIX = {
    "Video Conferencing": 0.20,
    "Gaming": 0.25,
    "Video Streaming": 0.30,
    "Background": 0.25,
}

# Congestion level overload multipliers
CONGESTION_OVERLOAD = MULT   # {"Low": 0.55, "Moderate": 1.10, "High": 1.65}


# ---------------------------------------------------------------------------
# Backward-compatible packet generation (dict-based)
# ---------------------------------------------------------------------------

def generate_packets(
    simulation_time: float,   # seconds
    bandwidth_mbps: float,    # link capacity in Mbps
    traffic_load: float,      # 0-1 UI slider (scales total offered load)
    congestion_level: str,    # "Low" | "Moderate" | "High"
    seed: int = 42,
    profile: str = "Default",
) -> list:
    """
    Generate a list of packet dicts (for backwards compatibility).
    """
    pt, sz, arr = generate(
        bw=bandwidth_mbps,
        load=traffic_load,
        level=congestion_level,
        sim_time=simulation_time,
        seed=seed,
        profile=profile,
    )
    packets = []
    for i in range(len(pt)):
        c_name = CLASSES[int(pt[i])]
        packets.append({
            "packet_id": i,
            "traffic_type": c_name,
            "size": int(sz[i]),
            "arrival_time": round(float(arr[i]), 6),
            "priority": TRAFFIC_TYPES[c_name]["priority"],
        })
    return packets


# ---------------------------------------------------------------------------
# Algorithm mapping
# ---------------------------------------------------------------------------

ALGORITHM_MAP = {
    "FIFO": fifo,
    "PQ": pq,
    "WFQ": wfq,
    "CBWFQ": cbwfq,
    "Adaptive QoS": adaptive,
}


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    algorithm: str,
    simulation_time: float,
    bandwidth_mbps: float,
    traffic_load: float,
    congestion_level: str,
    seed: int = 42,
    profile: str = "Default",
    **adaptive_kw,
) -> dict:
    """
    Run a simulation for a given algorithm using the vectorized engine and
    return a results dict matching app.py and database.py schema.

    Returns
    -------
    {
        "per_type": {traffic_type: {metric: value, ...}, ...},
        "overall":  {metric: value, ...},
        "adaptive_log": [...],
        "total_packets": int,
    }
    """
    # 1. Generate packet traces using parallel numpy arrays
    pt, sz, arr = generate(
        bw=bandwidth_mbps,
        load=traffic_load,
        level=congestion_level,
        sim_time=simulation_time,
        seed=seed,
        profile=profile,
    )

    adaptive_meta = None

    # 2. Schedule packets using the vectorized algorithms
    if algorithm == "FIFO":
        dep = fifo(pt, sz, arr, bandwidth_mbps)
    elif algorithm == "PQ":
        dep = pq(pt, sz, arr, bandwidth_mbps)
    elif algorithm == "WFQ":
        dep = wfq(pt, sz, arr, bandwidth_mbps)
    elif algorithm == "CBWFQ":
        dep = cbwfq(pt, sz, arr, bandwidth_mbps)
    elif algorithm == "Adaptive QoS":
        dep, adaptive_meta = adaptive(
            pt, sz, arr, bandwidth_mbps, congestion_level, **adaptive_kw
        )
    else:
        raise ValueError(f"Unknown QoS algorithm: '{algorithm}'")

    # 3. Compute metrics
    m_raw = metrics(pt, sz, arr, dep, bandwidth_mbps, simulation_time)

    # 4. Format metrics for dashboard & database storage
    per_type = {}
    for c_name in CLASSES:
        cm = m_raw.get(c_name, {})
        n = int(cm.get("n", 0))
        loss = float(cm.get("loss", 0.0))
        deliv = int(round((1.0 - loss / 100.0) * n))
        lat = cm.get("lat", 0.0)
        p99 = cm.get("p99", 0.0)

        per_type[c_name] = {
            "avg_latency": round(lat, 3) if not (math.isnan(lat) or math.isinf(lat)) else 0.0,
            "p99_latency": round(p99, 3) if not (math.isnan(p99) or math.isinf(p99)) else 0.0,
            "jitter": round(float(cm.get("jit", 0.0)), 3),
            "packet_loss": round(loss, 2),
            "throughput": round(float(cm.get("thr", 0.0)), 4),
            "bw_utilization": round(float(cm.get("util", 0.0)), 2),
            "total_packets": n,
            "delivered_packets": deliv,
            "dropped_packets": n - deliv,
        }

    ov = m_raw.get("Overall", {})
    ov_n = int(ov.get("n", 0))
    ov_loss = float(ov.get("loss", 0.0))
    ov_deliv = int(round((1.0 - ov_loss / 100.0) * ov_n))
    ov_lat = ov.get("lat", 0.0)
    ov_p99 = ov.get("p99", 0.0)

    overall = {
        "avg_latency": round(ov_lat, 3) if not (math.isnan(ov_lat) or math.isinf(ov_lat)) else 0.0,
        "p99_latency": round(ov_p99, 3) if not (math.isnan(ov_p99) or math.isinf(ov_p99)) else 0.0,
        "jitter": round(float(ov.get("jit", 0.0)), 3),
        "packet_loss": round(ov_loss, 2),
        "throughput": round(float(ov.get("thr", 0.0)), 4),
        "bw_utilization": round(float(ov.get("util", 0.0)), 2),
        "total_packets": ov_n,
        "delivered_packets": ov_deliv,
        "dropped_packets": ov_n - ov_deliv,
    }

    adaptive_log = []
    if adaptive_meta and "log" in adaptive_meta:
        for entry in adaptive_meta["log"]:
            # entry format: (t, score, lev, strict, bw_pct, loss_pct, avg_lat)
            t, score, lev, strict, bw_pct, loss_pct, avg_lat = entry
            w_tuple = ADAPT_W.get(lev, (4, 3, 2, 1))
            adaptive_log.append({
                "time": round(float(t), 4),
                "congestion_score": round(float(score), 4),
                "weights": {
                    "Video Conferencing": w_tuple[0],
                    "Gaming": w_tuple[1],
                    "Video Streaming": w_tuple[2],
                    "Background": w_tuple[3],
                },
                "strict_priority": bool(strict),
                "bw_utilization": round(float(bw_pct), 2),
                "packet_loss": round(float(loss_pct), 2),
                "avg_latency": round(float(avg_lat), 3),
            })

    results = {
        "per_type": per_type,
        "overall": overall,
        "adaptive_log": adaptive_log,
        "total_packets": len(pt),
    }

    if adaptive_meta:
        results["switches"] = adaptive_meta.get("switches", 0)
        results["evicted"] = adaptive_meta.get("evicted", 0)

    return results
