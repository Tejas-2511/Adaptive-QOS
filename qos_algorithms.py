"""
qos_algorithms.py
-----------------
Implements five packet-scheduling algorithms:

    1. FIFO          – First In, First Out
    2. PQ            – Priority Queuing
    3. WFQ           – Weighted Fair Queuing
    4. CBWFQ         – Class-Based WFQ
    5. Adaptive QoS  – Rule-based congestion-adaptive scheduling

Packets are stored in parallel numpy arrays for speed (index = packet id).

Each scheduler receives:
    pt  : np.array[int8]   – traffic-class index (0..3)
    sz  : np.array[int32]  – packet sizes in bytes
    arr : np.array[float]  – arrival times in seconds
    bw  : float            – link capacity in Mbps

Schedulers return departure-time arrays (NaN = dropped).
Adaptive QoS additionally returns a metadata dict with 'log', 'switches',
and 'evicted' fields.

High-level helpers:
    generate()   – numpy-based packet generator (supports workload profiles)
    run_all()    – run all (or selected) algorithms and return metrics dicts
"""

import heapq
import math
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# Traffic class names (index 0..3 → priority 4..1)
CLASSES = ["Video Conferencing", "Gaming", "Video Streaming", "Background"]

# Packet size ranges (bytes) per class
SIZE_RANGE = [(100, 300), (50, 200), (800, 1500), (500, 1400)]

# WFQ integer weights per class
WFQ_W = [4, 3, 2, 1]

# CBWFQ guaranteed bandwidth shares per class (must sum to 1.0)
CBWFQ_S = [0.35, 0.30, 0.20, 0.15]

# Traffic-mix profiles: fraction of bytes contributed by each class
MIXES = {
    "Default":    [0.20, 0.25, 0.30, 0.25],
    "RT-heavy":   [0.35, 0.30, 0.20, 0.15],
    "Bulk-heavy": [0.10, 0.15, 0.35, 0.40],
    "Bursty":     [0.20, 0.25, 0.30, 0.25],   # same mix, on/off arrival process
}

# Offered-load multiplier per congestion level
MULT = {"Low": 0.55, "Moderate": 1.10, "High": 1.65}

# Initial congestion score per level (seeds the adaptive algorithm immediately)
INIT_SCORE = {"Low": 0.15, "Moderate": 0.60, "High": 0.85}

# Shared buffer size (packets); excess are dropped.  Alias kept for legacy callers.
BUFFER = 200
BUFFER_SIZE = BUFFER   # backward-compatible alias

# Adaptive weight table indexed by congestion level (0=Low … 3=Severe)
ADAPT_W = {
    0: (4, 3, 2, 1),
    1: (6, 5, 2, 1),
    2: (8, 7, 2, 1),
    3: (10, 9, 3, 1),
}


# Transmission time helper (kept for legacy dict-based callers)
def _tx_time(size_bytes: float, bandwidth_mbps: float) -> float:
    """Return transmission time in seconds for a packet of given size."""
    return (size_bytes * 8) / (bandwidth_mbps * 1e6)


# ---------------------------------------------------------------------------
# Numpy-based helpers and generate
# ---------------------------------------------------------------------------

def level_of(score: float) -> int:
    """Map a congestion score [0, 1] to an integer level (0=Low … 3=Severe)."""
    return 0 if score < 0.3 else 1 if score < 0.6 else 2 if score < 0.8 else 3


def generate(
    bw: float,
    load: float,
    level: str,
    sim_time: float,
    seed: int,
    profile: str = "Default",
):
    """
    Generate a packet stream as three parallel numpy arrays.

    Parameters
    ----------
    bw        : link bandwidth in Mbps
    load      : traffic-load slider (0–1) from the UI
    level     : congestion level ("Low" | "Moderate" | "High")
    sim_time  : simulation duration in seconds
    seed      : RNG seed for reproducibility
    profile   : workload mix profile name ("Default" | "RT-heavy" | "Bulk-heavy" | "Bursty")

    Returns
    -------
    pt  : np.array[int8]   – class indices (0..3), sorted by arrival
    sz  : np.array[int32]  – packet sizes in bytes
    arr : np.array[float]  – arrival times in seconds
    """
    rng = np.random.default_rng(seed)
    mix = np.array(MIXES[profile])

    total_bytes = MULT[level] * load * bw * 1e6 / 8 * sim_time
    mean_sz = sum(m * (a + b) / 2 for m, (a, b) in zip(mix, SIZE_RANGE))
    n_est = int(total_bytes / mean_sz * 1.25) + 1000

    ptype = rng.choice(4, size=n_est, p=mix)
    lo = np.array([r[0] for r in SIZE_RANGE])[ptype]
    hi = np.array([r[1] for r in SIZE_RANGE])[ptype]
    size = rng.integers(lo, hi + 1)
    csum = np.cumsum(size)

    n = int(np.searchsorted(csum, total_bytes)) + 1
    ptype, size, csum = ptype[:n], size[:n], csum[:n]
    frac = (csum - size) / total_bytes

    if profile == "Bursty":
        # 70% of bytes arrive in the first 30% of each 5-second cycle
        cycles = sim_time / 5.0
        g = frac * cycles
        k = np.floor(g)
        gi = g - k
        tin = np.where(gi < 0.7, gi / 0.7 * 0.3, 0.3 + (gi - 0.7) / 0.3 * 0.7)
        arr = (k + tin) * 5.0
    else:
        # Slight early-burst skew (same formula as generate_packets in simulation.py)
        arr = (frac ** 0.85) * sim_time

    arr = arr + rng.uniform(0, 0.002, size=n)
    order = np.argsort(arr, kind="stable")
    return ptype[order].astype(np.int8), size[order].astype(np.int32), arr[order]


# ---------------------------------------------------------------------------
# Numpy-based schedulers (fast parallel-array API)
# ---------------------------------------------------------------------------

def fifo(pt, sz, arr, bw):
    """FIFO scheduler (numpy API). Returns departure-time array (NaN = dropped)."""
    n = len(pt)
    tx = (sz * 8.0 / (bw * 1e6)).tolist()
    a = arr.tolist()
    dep = [math.nan] * n
    q = deque()
    t = 0.0
    i = 0
    while i < n or q:
        if not q and t < a[i]:
            t = a[i]
        while i < n and a[i] <= t:
            if len(q) < BUFFER:
                q.append(i)
            i += 1
        if q:
            j = q.popleft()
            t += tx[j]
            dep[j] = t
    return np.array(dep)


def pq(pt, sz, arr, bw):
    """Strict Priority Queuing (numpy API). Returns departure-time array (NaN = dropped)."""
    n = len(pt)
    tx = (sz * 8.0 / (bw * 1e6)).tolist()
    a = arr.tolist()
    p = pt.tolist()
    dep = [math.nan] * n
    qs = [deque() for _ in range(4)]
    cnt = 0
    t = 0.0
    i = 0
    while i < n or cnt:
        if not cnt and t < a[i]:
            t = a[i]
        while i < n and a[i] <= t:
            if cnt < BUFFER:
                qs[p[i]].append(i)
                cnt += 1
            i += 1
        for c in range(4):
            if qs[c]:
                j = qs[c].popleft()
                cnt -= 1
                t += tx[j]
                dep[j] = t
                break
    return np.array(dep)


def _vtag(pt, sz, arr, bw, weights):
    """
    Virtual-time tag scheduler shared by wfq() and cbwfq() (numpy API).
    weights : sequence of per-class weights indexed 0..3.
    """
    n = len(pt)
    tx = (sz * 8.0 / (bw * 1e6)).tolist()
    a = arr.tolist()
    p = pt.tolist()
    s = sz.tolist()
    dep = [math.nan] * n
    h = []
    vt = [0.0] * 4
    V = 0.0
    t = 0.0
    i = 0
    while i < n or h:
        if not h and t < a[i]:
            t = a[i]
        while i < n and a[i] <= t:
            if len(h) < BUFFER:
                c = p[i]
                vs = vt[c] if vt[c] > V else V
                vf = vs + s[i] / weights[c]
                vt[c] = vf
                heapq.heappush(h, (vf, i))
            i += 1
        if h:
            V, j = heapq.heappop(h)
            t += tx[j]
            dep[j] = t
    return np.array(dep)


def wfq(pt, sz, arr, bw):
    """Weighted Fair Queuing (numpy API). Returns departure-time array (NaN = dropped)."""
    return _vtag(pt, sz, arr, bw, WFQ_W)


def cbwfq(pt, sz, arr, bw):
    """Class-Based WFQ (numpy API). Returns departure-time array (NaN = dropped)."""
    return _vtag(pt, sz, arr, bw, CBWFQ_S)


def adaptive(
    pt, sz, arr, bw, level,
    window=None,
    smooth=True,
    evict=True,
    evict_thr=0.6,
    seed_score=True,
    strict_mode=True,
):
    """
    Adaptive QoS scheduler (numpy API).

    Parameters
    ----------
    pt, sz, arr : parallel numpy arrays from generate()
    bw          : link bandwidth in Mbps
    level       : congestion level string ("Low" | "Moderate" | "High")
    window      : window size (int) or callable(n)->int; default max(30, n//25)
    smooth      : apply EMA score smoothing (default True)
    evict       : enable background-packet eviction under congestion (default True)
    evict_thr   : score threshold for triggering eviction (default 0.6)
    seed_score  : seed initial score from INIT_SCORE (default True)
    strict_mode : enable strict-priority mode above score 0.6 (default True)

    Returns
    -------
    dep  : np.array[float]  – departure times (NaN = dropped)
    meta : dict with keys:
               'log'      – list of per-window tuples
                            (time, score, level, strict, bw_pct, loss_pct, avg_lat_ms)
               'switches' – number of strict-priority ↔ WFQ mode switches
               'evicted'  – total background packets evicted to protect real-time
    """
    n = len(pt)
    tx = (sz * 8.0 / (bw * 1e6)).tolist()
    a = arr.tolist()
    p = pt.tolist()
    s = sz.tolist()

    W = window(n) if callable(window) else (window or max(30, n // 25))
    dep = [math.nan] * n

    score = INIT_SCORE[level] if seed_score else 0.0
    lev = level_of(score)
    strict = strict_mode and score >= 0.6

    heap = []
    rt = deque()    # real-time queue (classes 0, 1)
    be = deque()    # best-effort queue (classes 2, 3)
    vt = [0.0] * 4
    V = [0.0]       # wrapped in list so inner closure can mutate

    cnt = 0
    t = 0.0
    i = 0
    w_served = w_drop = 0
    w_bits = 0.0
    w_lat = 0.0
    w_start = 0.0
    log = []
    switches = 0
    evicted = 0

    def push_heap(j):
        c = p[j]
        wt = ADAPT_W[lev][c]
        vs = vt[c] if vt[c] > V[0] else V[0]
        vf = vs + s[j] / wt
        vt[c] = vf
        heapq.heappush(heap, (vf, j))

    while i < n or cnt:
        if cnt == 0 and t < a[i]:
            t = a[i]

        while i < n and a[i] <= t:
            rtp = p[i] < 2
            if cnt >= BUFFER:
                if evict and strict and score >= evict_thr and rtp and be:
                    be.popleft()
                    cnt -= 1
                    w_drop += 1
                    evicted += 1
                else:
                    w_drop += 1
                    i += 1
                    continue
            if strict:
                (rt if rtp else be).append(i)
            else:
                push_heap(i)
            cnt += 1
            i += 1

        if cnt:
            if strict:
                j = rt.popleft() if rt else be.popleft()
            else:
                V[0], j = heapq.heappop(heap)
            cnt -= 1
            t += tx[j]
            dep[j] = t
            w_served += 1
            w_bits += s[j] * 8.0
            w_lat += (t - a[j]) * 1000.0

            if w_served >= W:
                dur = max(t - w_start, 1e-9)
                bwn   = min(w_bits / (bw * 1e6 * dur), 1.0)
                lossn = min(w_drop / (w_served + w_drop), 1.0)
                latn  = min((w_lat / w_served) / 500.0, 1.0)
                new_score = 0.5 * bwn + 0.3 * lossn + 0.2 * latn
                score = (0.7 * new_score + 0.3 * score) if smooth else new_score
                lev = level_of(score)
                now_strict = strict_mode and score >= 0.6
                if now_strict != strict:
                    switches += 1
                    if now_strict:
                        for j2 in sorted(x[1] for x in heap):
                            (rt if p[j2] < 2 else be).append(j2)
                        heap = []
                    else:
                        for j2 in sorted(list(rt) + list(be)):
                            push_heap(j2)
                        rt.clear()
                        be.clear()
                    strict = now_strict
                log.append((t, score, lev, strict, bwn * 100, lossn * 100, w_lat / w_served))
                w_served = w_drop = 0
                w_bits = 0.0
                w_lat = 0.0
                w_start = t

    return np.array(dep), {"log": log, "switches": switches, "evicted": evicted}


def metrics(pt, sz, arr, dep, bw, sim_time):
    """
    Compute per-class and overall QoS metrics (numpy API).

    Returns
    -------
    dict keyed by class name and "Overall", each value containing:
        lat   – mean latency (ms)
        p99   – 99th-percentile latency (ms)
        jit   – mean jitter (ms)
        loss  – packet loss (%)
        thr   – throughput (Mbps)
        util  – BW utilization (%)
        n     – total packets in class
    """
    out = {}
    class_masks = [(c, pt == k) for k, c in enumerate(CLASSES)]
    overall_mask = np.ones(len(pt), dtype=bool)
    for name, mask in class_masks + [("Overall", overall_mask)]:
        tot = int(mask.sum())
        ok = mask & ~np.isnan(dep)
        lat = (dep[ok] - arr[ok]) * 1000.0
        jit = float(np.mean(np.abs(np.diff(lat)))) if len(lat) > 1 else 0.0
        inwin = ok & (dep <= sim_time)
        thr = sz[inwin].sum() * 8.0 / sim_time / 1e6
        out[name] = dict(
            lat=float(lat.mean()) if len(lat) else float("nan"),
            p99=float(np.percentile(lat, 99)) if len(lat) else float("nan"),
            jit=jit,
            loss=100.0 * (tot - int(ok.sum())) / max(tot, 1),
            thr=float(thr),
            util=float(thr / bw * 100),
            n=tot,
        )
    return out


def run_all(
    bw, load, level, sim_time, seed,
    profile="Default",
    algos=("FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"),
    **kw,
):
    """
    Generate packets once and run all selected algorithms (numpy API).

    Parameters
    ----------
    bw, load, level, sim_time, seed : passed to generate()
    profile : workload mix profile (Default | RT-heavy | Bulk-heavy | Bursty)
    algos   : tuple of algorithm names to run
    **kw    : extra keyword arguments forwarded to adaptive()

    Returns
    -------
    res   : dict[algo] -> metrics dict (from metrics())
    extra : dict[algo] -> adaptive metadata (log, switches, evicted)
    pkts  : (pt, sz, arr) tuple
    """
    pt, sz, arr = generate(bw, load, level, sim_time, seed, profile)
    res = {}
    extra = {}
    for al in algos:
        if al == "FIFO":
            dep = fifo(pt, sz, arr, bw)
        elif al == "PQ":
            dep = pq(pt, sz, arr, bw)
        elif al == "WFQ":
            dep = wfq(pt, sz, arr, bw)
        elif al == "CBWFQ":
            dep = cbwfq(pt, sz, arr, bw)
        else:
            dep, extra[al] = adaptive(pt, sz, arr, bw, level, **kw)
        res[al] = metrics(pt, sz, arr, dep, bw, sim_time)
    return res, extra, (pt, sz, arr)


