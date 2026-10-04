"""
qos_algorithms.py
-----------------
Implements five packet-scheduling algorithms:

    1. FIFO          – First In, First Out
    2. PQ            – Priority Queuing
    3. WFQ           – Weighted Fair Queuing
    4. CBWFQ         – Class-Based WFQ
    5. Adaptive QoS  – Rule-based congestion-adaptive scheduling

Each function receives the same arguments:
    packets         : list of packet dicts (sorted by arrival_time)
    bandwidth_mbps  : link capacity in Mbps
    simulation_time : total simulation window in seconds
    congestion_level: "Low" | "Moderate" | "High"

Each function returns a list of processed packet dicts, each containing
all original fields PLUS:
    departure_time  : float  (seconds) – when the packet left the queue
    dropped         : bool   – True if the packet was dropped (buffer full)

Adaptive QoS returns a tuple: (processed_packets, adaptive_log)
    adaptive_log    : list of dicts recording how priority weights changed
"""

import copy

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Queue buffer size (number of packets).  Packets beyond this are dropped.
BUFFER_SIZE = 200   # tighter buffer → realistic drops under overload

# Transmission time helper
def _tx_time(size_bytes: float, bandwidth_mbps: float) -> float:
    """Return transmission time in seconds for a packet of given size."""
    return (size_bytes * 8) / (bandwidth_mbps * 1e6)


# ---------------------------------------------------------------------------
# Helper: build a base result packet
# ---------------------------------------------------------------------------

def _make_result(pkt: dict, departure_time: float, dropped: bool) -> dict:
    r = copy.copy(pkt)
    r["departure_time"] = departure_time
    r["dropped"] = dropped
    return r


# ---------------------------------------------------------------------------
# 1. FIFO
# ---------------------------------------------------------------------------

def fifo_schedule(packets, bandwidth_mbps, simulation_time, congestion_level, **_):
    """
    First-In First-Out scheduling.

    Packets are served strictly in arrival order.
    If the buffer is full, the incoming packet is dropped.
    """
    results = []
    current_time = 0.0   # when the link becomes free
    queue_len = 0        # number of packets currently buffered

    for pkt in packets:
        arrival = pkt["arrival_time"]

        # If link was idle, advance to arrival time
        if current_time < arrival:
            current_time = arrival
            queue_len = max(queue_len - 1, 0)   # some packets drained

        # Buffer check
        if queue_len >= BUFFER_SIZE:
            results.append(_make_result(pkt, departure_time=current_time, dropped=True))
            continue

        queue_len += 1
        tx = _tx_time(pkt["size"], bandwidth_mbps)
        current_time += tx
        queue_len -= 1
        results.append(_make_result(pkt, departure_time=current_time, dropped=False))

    return results


# ---------------------------------------------------------------------------
# 2. Priority Queuing (PQ)
# ---------------------------------------------------------------------------

def pq_schedule(packets, bandwidth_mbps, simulation_time, congestion_level, **_):
    """
    Strict Priority Queuing.

    Maintains four priority queues (4=highest … 1=lowest).
    At each scheduling decision, the highest non-empty queue is served.
    Lower-priority traffic may starve if high-priority traffic is heavy.
    """
    from collections import deque

    # Priority queues: key = priority level (4,3,2,1)
    queues = {4: deque(), 3: deque(), 2: deque(), 1: deque()}
    results = []
    current_time = 0.0
    pkt_index = 0
    n = len(packets)

    while pkt_index < n or any(queues.values()):
        # Enqueue all packets that have arrived by current_time
        while pkt_index < n and packets[pkt_index]["arrival_time"] <= current_time:
            pkt = packets[pkt_index]
            q = queues[pkt["priority"]]
            if len(q) < BUFFER_SIZE:
                q.append(pkt)
            else:
                results.append(_make_result(pkt, departure_time=current_time, dropped=True))
            pkt_index += 1

        # Find highest non-empty queue
        served = None
        for p in [4, 3, 2, 1]:
            if queues[p]:
                served = queues[p].popleft()
                break

        if served is None:
            # No packet in queue – jump to next arrival
            if pkt_index < n:
                current_time = packets[pkt_index]["arrival_time"]
            continue

        tx = _tx_time(served["size"], bandwidth_mbps)
        current_time += tx
        results.append(_make_result(served, departure_time=current_time, dropped=False))

    return results


# ---------------------------------------------------------------------------
# 3. WFQ – Weighted Fair Queuing
# ---------------------------------------------------------------------------

# Weights per traffic type
WFQ_WEIGHTS = {
    "Video Conferencing": 4,
    "Gaming": 3,
    "Video Streaming": 2,
    "Background": 1,
}

def wfq_schedule(packets, bandwidth_mbps, simulation_time, congestion_level, **_):
    """
    Weighted Fair Queuing.

    Each traffic class gets a share of bandwidth proportional to its weight.
    We implement a simplified WFQ using a finish-time based virtual clock:
    each packet's virtual finish time = virtual_start + (size / weight).
    Packets with the smallest virtual finish time are served first.
    """
    import heapq

    total_weight = sum(WFQ_WEIGHTS.values())

    # Per-class virtual time tracker
    virtual_time = {t: 0.0 for t in WFQ_WEIGHTS}
    current_time = 0.0
    results = []
    pkt_index = 0
    n = len(packets)

    # Priority heap: (virtual_finish_time, arrival_time, pkt)
    heap = []
    buffer_count = 0

    while pkt_index < n or heap:
        # Enqueue arrivals up to current_time
        while pkt_index < n and packets[pkt_index]["arrival_time"] <= current_time:
            pkt = packets[pkt_index]
            if buffer_count < BUFFER_SIZE:
                w = WFQ_WEIGHTS.get(pkt["traffic_type"], 1)
                # Virtual start = max(virtual_time[class], current_time)
                v_start = max(virtual_time[pkt["traffic_type"]], current_time)
                v_finish = v_start + (pkt["size"] / w)
                virtual_time[pkt["traffic_type"]] = v_finish
                heapq.heappush(heap, (v_finish, pkt["arrival_time"], pkt))
                buffer_count += 1
            else:
                results.append(_make_result(pkt, departure_time=current_time, dropped=True))
            pkt_index += 1

        if not heap:
            if pkt_index < n:
                current_time = packets[pkt_index]["arrival_time"]
            continue

        _, _, served = heapq.heappop(heap)
        buffer_count -= 1
        tx = _tx_time(served["size"], bandwidth_mbps)
        current_time += tx
        results.append(_make_result(served, departure_time=current_time, dropped=False))

    return results


# ---------------------------------------------------------------------------
# 4. CBWFQ – Class-Based WFQ
# ---------------------------------------------------------------------------

# Guaranteed bandwidth share per class (must sum to 1.0)
CBWFQ_SHARES = {
    "Video Conferencing": 0.35,
    "Gaming": 0.30,
    "Video Streaming": 0.20,
    "Background": 0.15,
}

def cbwfq_schedule(packets, bandwidth_mbps, simulation_time, congestion_level, **_):
    """
    Class-Based Weighted Fair Queuing.

    Each class has a guaranteed bandwidth allocation.
    We implement it similarly to WFQ but use the bandwidth share to
    compute virtual finish times (weight ∝ bandwidth share).
    Excess bandwidth is shared fairly among active classes.
    """
    import heapq

    # Use shares as weights for virtual clock calculation
    shares = CBWFQ_SHARES
    current_time = 0.0
    virtual_time = {t: 0.0 for t in shares}
    results = []
    pkt_index = 0
    n = len(packets)
    heap = []
    buffer_count = 0

    while pkt_index < n or heap:
        while pkt_index < n and packets[pkt_index]["arrival_time"] <= current_time:
            pkt = packets[pkt_index]
            if buffer_count < BUFFER_SIZE:
                share = shares.get(pkt["traffic_type"], 0.10)
                v_start = max(virtual_time[pkt["traffic_type"]], current_time)
                # Smaller share → larger virtual finish → served later
                v_finish = v_start + (pkt["size"] / share)
                virtual_time[pkt["traffic_type"]] = v_finish
                heapq.heappush(heap, (v_finish, pkt["arrival_time"], pkt))
                buffer_count += 1
            else:
                results.append(_make_result(pkt, departure_time=current_time, dropped=True))
            pkt_index += 1

        if not heap:
            if pkt_index < n:
                current_time = packets[pkt_index]["arrival_time"]
            continue

        _, _, served = heapq.heappop(heap)
        buffer_count -= 1
        tx = _tx_time(served["size"], bandwidth_mbps)
        current_time += tx
        results.append(_make_result(served, departure_time=current_time, dropped=False))

    return results


# ---------------------------------------------------------------------------
# 5. Adaptive QoS
# ---------------------------------------------------------------------------

def _compute_congestion_score(
    bw_utilization: float,   # 0-1
    packet_loss: float,      # 0-1
    latency: float,          # 0-1 (normalised)
) -> float:
    """
    Congestion score formula from the project spec:
        Score = 0.5 × BW_util + 0.3 × Packet_loss + 0.2 × Latency
    All inputs must already be normalised to [0, 1].
    """
    return 0.5 * bw_utilization + 0.3 * packet_loss + 0.2 * latency


def _get_adaptive_weights(congestion_score: float) -> dict:
    """
    Rule-based weight assignment based on congestion level.

    Thresholds (easy to change):
        0.0 – 0.3  → Low
        0.3 – 0.6  → Moderate
        0.6 – 0.8  → High
        0.8 – 1.0  → Severe
    """
    if congestion_score < 0.3:
        # Low congestion – normal allocation
        return {
            "Video Conferencing": 4,
            "Gaming": 3,
            "Video Streaming": 2,
            "Background": 1,
        }
    elif congestion_score < 0.6:
        # Moderate – slightly boost real-time
        return {
            "Video Conferencing": 6,
            "Gaming": 5,
            "Video Streaming": 2,
            "Background": 1,
        }
    elif congestion_score < 0.8:
        # High – strongly prioritise real-time
        return {
            "Video Conferencing": 8,
            "Gaming": 7,
            "Video Streaming": 2,
            "Background": 1,
        }
    else:
        # Severe – maximum priority for real-time, minimum for background
        return {
            "Video Conferencing": 10,
            "Gaming": 9,
            "Video Streaming": 3,
            "Background": 1,   # minimum guaranteed share
        }


# Map congestion_level string → approximate initial score.
# These now reflect overloaded traffic (Moderate=1.1x, High=1.65x capacity).
_INITIAL_SCORE = {
    "Low":      0.15,
    "Moderate": 0.60,
    "High":     0.85,
}


# Real-time traffic classes that get priority protection under high congestion
_REALTIME_CLASSES = {"Video Conferencing", "Gaming"}


def adaptive_qos_schedule(packets, bandwidth_mbps, simulation_time, congestion_level, **_):
    """
    Adaptive QoS scheduling – hybrid priority + weighted approach.

    Strategy:
    - Divides simulation into time windows. After each window, recomputes the
      congestion score from observed BW utilisation, drop rate and latency.
    - Under LOW congestion  → pure WFQ (same as the WFQ algorithm).
    - Under MODERATE        → boost real-time weights (6/5 vs 2/1).
    - Under HIGH / SEVERE   → strict priority for real-time classes:
        serve ALL queued VC and Gaming packets before touching Streaming/BG.
        This is what makes Adaptive QoS measurably better for real-time
        traffic when the network is congested.

    Buffer management:
    - Under high congestion, new Background packets are dropped first (tail-drop
      by class) to protect the buffer for real-time traffic.

    Returns: (processed_packets, adaptive_log)
    """
    import heapq
    from collections import deque
    from metrics import _compute_group_metrics

    # Window size: recalculate every N served packets
    WINDOW_PACKETS = max(30, len(packets) // 25)

    current_score = _INITIAL_SCORE.get(congestion_level, 0.60)
    weights = _get_adaptive_weights(current_score)

    current_time = 0.0
    virtual_time = {t: 0.0 for t in weights}

    # Separate queues for strict-priority mode
    rt_queue  = deque()   # real-time (VC + Gaming)
    be_queue  = deque()   # best-effort (Streaming + Background)

    # WFQ heap used in low-congestion mode
    heap = []

    results = []
    adaptive_log = []
    pkt_index = 0
    n = len(packets)
    buffer_count = 0
    window_results = []
    packets_since_update = 0

    def _use_strict_priority():
        """True when congestion is High or Severe."""
        return current_score >= 0.6

    def _push_wfq(pkt, w_map, v_time_map):
        """Push a packet onto the WFQ heap."""
        w = w_map.get(pkt["traffic_type"], 1)
        v_start = max(v_time_map[pkt["traffic_type"]], current_time)
        v_finish = v_start + (pkt["size"] / w)
        v_time_map[pkt["traffic_type"]] = v_finish
        heapq.heappush(heap, (v_finish, pkt["arrival_time"], pkt))

    def _enqueue(pkt):
        """Add arriving packet to appropriate queue; drop if buffer full."""
        nonlocal buffer_count
        is_rt = pkt["traffic_type"] in _REALTIME_CLASSES

        if buffer_count >= BUFFER_SIZE:
            # Under strict-priority mode, drop BE packets to protect RT
            if _use_strict_priority() and not is_rt and be_queue:
                # Drop the oldest background packet to make room
                dropped_old = be_queue.popleft()
                r_drop = _make_result(dropped_old, departure_time=current_time, dropped=True)
                results.append(r_drop)
                window_results.append(r_drop)
                buffer_count -= 1
            elif buffer_count >= BUFFER_SIZE:
                # Buffer truly full – drop incoming packet
                r = _make_result(pkt, departure_time=current_time, dropped=True)
                results.append(r)
                window_results.append(r)
                return

        if _use_strict_priority():
            if is_rt:
                rt_queue.append(pkt)
            else:
                be_queue.append(pkt)
        else:
            _push_wfq(pkt, weights, virtual_time)
        buffer_count += 1

    def _next_packet():
        """Pop the next packet to serve according to current mode."""
        if _use_strict_priority():
            if rt_queue:
                return rt_queue.popleft()
            if be_queue:
                return be_queue.popleft()
            return None
        else:
            if heap:
                _, _, pkt = heapq.heappop(heap)
                return pkt
            return None

    def _has_queued():
        if _use_strict_priority():
            return bool(rt_queue or be_queue)
        return bool(heap)

    while pkt_index < n or _has_queued():
        # Enqueue all arrivals up to current_time
        while pkt_index < n and packets[pkt_index]["arrival_time"] <= current_time:
            _enqueue(packets[pkt_index])
            pkt_index += 1

        served = _next_packet()
        if served is None:
            if pkt_index < n:
                current_time = packets[pkt_index]["arrival_time"]
            continue

        buffer_count -= 1
        tx = _tx_time(served["size"], bandwidth_mbps)
        current_time += tx

        r = _make_result(served, departure_time=current_time, dropped=False)
        results.append(r)
        window_results.append(r)
        packets_since_update += 1

        # ----- Adaptation step: recalculate after each window -----
        if packets_since_update >= WINDOW_PACKETS:
            window_metrics = _compute_group_metrics(
                window_results, simulation_time, bandwidth_mbps
            )

            # Normalise to [0, 1] for congestion score
            bw_norm   = min(window_metrics["bw_utilization"] / 100.0, 1.0)
            loss_norm = min(window_metrics["packet_loss"]    / 100.0, 1.0)
            lat_norm  = min(window_metrics["avg_latency"]    / 500.0, 1.0)  # 0-500ms range

            prev_score = current_score
            current_score = _compute_congestion_score(bw_norm, loss_norm, lat_norm)
            # Smooth score to avoid flapping
            current_score = 0.7 * current_score + 0.3 * prev_score
            weights = _get_adaptive_weights(current_score)

            # When transitioning out of strict-priority mode, flush queues to WFQ heap
            if not _use_strict_priority():
                while rt_queue:
                    _push_wfq(rt_queue.popleft(), weights, virtual_time)
                while be_queue:
                    _push_wfq(be_queue.popleft(), weights, virtual_time)

            # Ensure virtual_time has all classes
            for t in weights:
                if t not in virtual_time:
                    virtual_time[t] = 0.0

            adaptive_log.append({
                "time": round(current_time, 4),
                "congestion_score": round(current_score, 4),
                "weights": dict(weights),
                "strict_priority": _use_strict_priority(),
                "bw_utilization": round(window_metrics["bw_utilization"], 2),
                "packet_loss": round(window_metrics["packet_loss"], 2),
                "avg_latency": round(window_metrics["avg_latency"], 3),
            })

            window_results = []
            packets_since_update = 0

    return results, adaptive_log
