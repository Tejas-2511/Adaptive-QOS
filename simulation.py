"""
simulation.py
-------------
Handles packet generation and the main simulation loop.

A 'packet' is just a Python dict with fields:
    packet_id, traffic_type, size (bytes), arrival_time, priority

The simulator feeds packets through a chosen QoS algorithm and
collects per-packet departure times and drop flags.
"""

import random
from qos_algorithms import (
    fifo_schedule,
    pq_schedule,
    wfq_schedule,
    cbwfq_schedule,
    adaptive_qos_schedule,
)
from metrics import compute_metrics

# ---------------------------------------------------------------------------
# Traffic type definitions
# ---------------------------------------------------------------------------

TRAFFIC_TYPES = {
    "Video Conferencing": {
        "priority": 4,        # highest
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

# How much of the total traffic each type contributes (must sum to 1.0)
TRAFFIC_MIX = {
    "Video Conferencing": 0.20,
    "Gaming": 0.25,
    "Video Streaming": 0.30,
    "Background": 0.25,
}

# Congestion level → offered load multiplier relative to link capacity.
# Values > 1.0 mean we inject more traffic than the link can drain,
# which creates real queuing delay and packet drops so that different
# QoS algorithms produce visibly different results.
CONGESTION_OVERLOAD = {
    "Low":      0.55,   # ~55% of capacity – no real stress
    "Moderate": 1.10,   # 10% overloaded  → noticeable queuing / some drops
    "High":     1.65,   # 65% overloaded  → heavy drops / large queues
}


# ---------------------------------------------------------------------------
# Packet generation
# ---------------------------------------------------------------------------

def generate_packets(
    simulation_time: float,   # seconds
    bandwidth_mbps: float,    # link capacity in Mbps
    traffic_load: float,      # 0-1 UI slider (scales total offered load)
    congestion_level: str,    # "Low" | "Moderate" | "High"
    seed: int = 42,
) -> list:
    """
    Generate a list of packets arriving over [0, simulation_time].

    Offered load = traffic_load * CONGESTION_OVERLOAD[level] * bandwidth.
    At Moderate/High congestion the offered load intentionally exceeds the
    link capacity so that queues build up and algorithms are differentiated.
    """
    rng = random.Random(seed)

    overload = CONGESTION_OVERLOAD.get(congestion_level, 1.10)

    # Total bytes to inject (may exceed what the link can drain)
    total_bytes_target = (
        overload * traffic_load * bandwidth_mbps * 1e6 / 8
        * simulation_time
    )

    packets = []
    packet_id = 0
    bytes_generated = 0

    while bytes_generated < total_bytes_target:
        # Pick traffic type according to mix probabilities
        traffic_type = rng.choices(
            list(TRAFFIC_MIX.keys()),
            weights=list(TRAFFIC_MIX.values()),
        )[0]

        info = TRAFFIC_TYPES[traffic_type]
        size = rng.randint(*info["size_range"])

        # Arrival time: uniformly spread over simulation window.
        # Slightly skew towards earlier arrivals to create initial burst pressure.
        fraction_done = bytes_generated / max(total_bytes_target, 1)
        arrival_time = (fraction_done ** 0.85) * simulation_time + rng.uniform(0, 0.002)
        arrival_time = min(arrival_time, simulation_time)

        packets.append({
            "packet_id": packet_id,
            "traffic_type": traffic_type,
            "size": size,           # bytes
            "arrival_time": round(arrival_time, 6),
            "priority": info["priority"],
        })

        bytes_generated += size
        packet_id += 1

    # Sort by arrival time – ensures FIFO ordering is correct
    packets.sort(key=lambda p: p["arrival_time"])

    # Re-assign sequential IDs after sorting
    for i, pkt in enumerate(packets):
        pkt["packet_id"] = i

    return packets


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------

ALGORITHM_MAP = {
    "FIFO": fifo_schedule,
    "PQ": pq_schedule,
    "WFQ": wfq_schedule,
    "CBWFQ": cbwfq_schedule,
    "Adaptive QoS": adaptive_qos_schedule,
}


def run_simulation(
    algorithm: str,
    simulation_time: float,
    bandwidth_mbps: float,
    traffic_load: float,
    congestion_level: str,
    seed: int = 42,
) -> dict:
    """
    Run a full simulation for a given algorithm and return a results dict.

    Returns
    -------
    {
        "per_type": {traffic_type: {metric: value, ...}, ...},
        "overall":  {metric: value, ...},
        "adaptive_log": [...],   # only for Adaptive QoS
    }
    """
    # 1. Generate packets (same seed → fair comparison)
    packets = generate_packets(
        simulation_time=simulation_time,
        bandwidth_mbps=bandwidth_mbps,
        traffic_load=traffic_load,
        congestion_level=congestion_level,
        seed=seed,
    )

    # 2. Schedule packets through the chosen QoS algorithm
    schedule_fn = ALGORITHM_MAP[algorithm]
    scheduled = schedule_fn(
        packets=packets,
        bandwidth_mbps=bandwidth_mbps,
        simulation_time=simulation_time,
        congestion_level=congestion_level,
    )

    # scheduled is a list of dicts:
    # {packet_id, traffic_type, size, arrival_time, priority,
    #  departure_time, dropped}
    # Plus optional "adaptive_log" key on the returned tuple.

    adaptive_log = []
    if isinstance(scheduled, tuple):
        scheduled, adaptive_log = scheduled

    # 3. Compute metrics
    results = compute_metrics(
        scheduled_packets=scheduled,
        simulation_time=simulation_time,
        bandwidth_mbps=bandwidth_mbps,
    )
    results["adaptive_log"] = adaptive_log
    results["total_packets"] = len(packets)

    return results
