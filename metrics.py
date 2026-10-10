"""
metrics.py
----------
Computes QoS metrics from a list of scheduled (processed) packets.

Each packet dict must have:
    traffic_type, size, arrival_time, departure_time, dropped (bool)
"""

import numpy as np


def compute_metrics(
    scheduled_packets: list,
    simulation_time: float,
    bandwidth_mbps: float,
) -> dict:
    """
    Compute per-traffic-type and overall metrics.

    Returns
    -------
    {
        "per_type": {
            "Gaming": {
                "avg_latency": float,   # ms
                "jitter": float,        # ms
                "packet_loss": float,   # %
                "throughput": float,    # Mbps
                "bw_utilization": float # %
            },
            ...
        },
        "overall": { same keys }
    }
    """
    # Group packets by traffic type
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for pkt in scheduled_packets:
        groups[pkt["traffic_type"]].append(pkt)

    per_type = {}
    for ttype, pkts in groups.items():
        per_type[ttype] = _compute_group_metrics(
            pkts, simulation_time, bandwidth_mbps
        )

    overall = _compute_group_metrics(scheduled_packets, simulation_time, bandwidth_mbps)

    return {"per_type": per_type, "overall": overall}


def _compute_group_metrics(pkts: list, simulation_time: float, bandwidth_mbps: float) -> dict:
    """Compute metrics for a single group (traffic type or all packets)."""
    total = len(pkts)
    if total == 0:
        return {
            "avg_latency": 0.0,
            "jitter": 0.0,
            "packet_loss": 0.0,
            "throughput": 0.0,
            "bw_utilization": 0.0,
        }

    dropped = [p for p in pkts if p.get("dropped", False)]
    delivered = [p for p in pkts if not p.get("dropped", False)]

    # Packet loss
    packet_loss = (len(dropped) / total) * 100.0

    if not delivered:
        return {
            "avg_latency": 0.0,
            "jitter": 0.0,
            "packet_loss": packet_loss,
            "throughput": 0.0,
            "bw_utilization": 0.0,
        }

    # Latency for each delivered packet (convert seconds → ms)
    latencies = np.array([
        (p["departure_time"] - p["arrival_time"]) * 1000.0
        for p in delivered
    ])
    avg_latency = float(np.mean(latencies))
    p99_latency = float(np.percentile(latencies, 99))

    # Jitter = mean of absolute successive latency differences
    if len(latencies) > 1:
        diffs = np.abs(np.diff(latencies))
        jitter = float(np.mean(diffs))
    else:
        jitter = 0.0

    # Throughput = total delivered bytes / simulation_time → convert to Mbps
    total_bytes = sum(p["size"] for p in delivered)
    throughput_mbps = (total_bytes * 8) / (simulation_time * 1e6)

    # Bandwidth utilization
    bw_utilization = (throughput_mbps / bandwidth_mbps) * 100.0

    return {
        "avg_latency": round(avg_latency, 3),
        "p99_latency": round(p99_latency, 3),
        "jitter": round(jitter, 3),
        "packet_loss": round(packet_loss, 2),
        "throughput": round(throughput_mbps, 4),
        "bw_utilization": round(bw_utilization, 2),
    }
