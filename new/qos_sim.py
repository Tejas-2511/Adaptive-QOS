"""
Adaptive QoS reference re-implementation (single-link, four traffic classes).
Follows the 'Adaptive QoS - Technical Reference': packet model, generator,
FIFO / PQ / WFQ / CBWFQ / Adaptive QoS schedulers and metrics.
Packets are stored in parallel arrays for speed (index = packet id).
"""
import heapq, math
from collections import deque
import numpy as np

CLASSES = ["Video Conferencing", "Gaming", "Video Streaming", "Background"]   # index 0..3, priority 4..1
SIZE_RANGE = [(100, 300), (50, 200), (800, 1500), (500, 1400)]
WFQ_W = [4, 3, 2, 1]
CBWFQ_S = [0.35, 0.30, 0.20, 0.15]
MIXES = {
    "Default":    [0.20, 0.25, 0.30, 0.25],
    "RT-heavy":   [0.35, 0.30, 0.20, 0.15],
    "Bulk-heavy": [0.10, 0.15, 0.35, 0.40],
    "Bursty":     [0.20, 0.25, 0.30, 0.25],   # same mix, on/off arrival process
}
MULT = {"Low": 0.55, "Moderate": 1.10, "High": 1.65}
INIT_SCORE = {"Low": 0.15, "Moderate": 0.60, "High": 0.85}
BUFFER = 200
ADAPT_W = {0: (4, 3, 2, 1), 1: (6, 5, 2, 1), 2: (8, 7, 2, 1), 3: (10, 9, 3, 1)}


def level_of(score):
    return 0 if score < 0.3 else 1 if score < 0.6 else 2 if score < 0.8 else 3


def generate(bw, load, level, sim_time, seed, profile="Default"):
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
    if profile == "Bursty":           # 70 % of the bytes arrive in the first 30 % of each 5 s cycle
        cycles = sim_time / 5.0
        g = frac * cycles
        k = np.floor(g); gi = g - k
        tin = np.where(gi < 0.7, gi / 0.7 * 0.3, 0.3 + (gi - 0.7) / 0.3 * 0.7)
        arr = (k + tin) * 5.0
    else:
        arr = (frac ** 0.85) * sim_time
    arr = arr + rng.uniform(0, 0.002, size=n)
    order = np.argsort(arr, kind="stable")
    return ptype[order].astype(np.int8), size[order].astype(np.int32), arr[order]


# ---------------------------------------------------------------- schedulers
def fifo(pt, sz, arr, bw):
    n = len(pt); tx = (sz * 8.0 / (bw * 1e6)).tolist(); a = arr.tolist()
    dep = [math.nan] * n; q = deque(); t = 0.0; i = 0
    while i < n or q:
        if not q and t < a[i]: t = a[i]
        while i < n and a[i] <= t:
            if len(q) >= BUFFER: pass
            else: q.append(i)
            i += 1
        if q:
            j = q.popleft(); t += tx[j]; dep[j] = t
    return np.array(dep)


def pq(pt, sz, arr, bw):
    n = len(pt); tx = (sz * 8.0 / (bw * 1e6)).tolist(); a = arr.tolist(); p = pt.tolist()
    dep = [math.nan] * n; qs = [deque() for _ in range(4)]; cnt = 0; t = 0.0; i = 0
    while i < n or cnt:
        if not cnt and t < a[i]: t = a[i]
        while i < n and a[i] <= t:
            if cnt < BUFFER: qs[p[i]].append(i); cnt += 1
            i += 1
        for c in range(4):
            if qs[c]:
                j = qs[c].popleft(); cnt -= 1; t += tx[j]; dep[j] = t; break
    return np.array(dep)


def _vtag(pt, sz, arr, bw, weights):
    n = len(pt); tx = (sz * 8.0 / (bw * 1e6)).tolist(); a = arr.tolist()
    p = pt.tolist(); s = sz.tolist()
    dep = [math.nan] * n; h = []; vt = [0.0] * 4; V = 0.0; t = 0.0; i = 0
    while i < n or h:
        if not h and t < a[i]: t = a[i]
        while i < n and a[i] <= t:
            if len(h) < BUFFER:
                c = p[i]; vs = vt[c] if vt[c] > V else V
                vf = vs + s[i] / weights[c]; vt[c] = vf
                heapq.heappush(h, (vf, i))
            i += 1
        if h:
            V, j = heapq.heappop(h); t += tx[j]; dep[j] = t
    return np.array(dep)


def wfq(pt, sz, arr, bw):   return _vtag(pt, sz, arr, bw, WFQ_W)
def cbwfq(pt, sz, arr, bw): return _vtag(pt, sz, arr, bw, CBWFQ_S)


def adaptive(pt, sz, arr, bw, level, window=None, smooth=True, evict=True,
             evict_thr=0.6, seed_score=True, strict_mode=True):
    """Adaptive QoS. Returns departure times (NaN = dropped) and a log list."""
    n = len(pt); tx = (sz * 8.0 / (bw * 1e6)).tolist(); a = arr.tolist()
    p = pt.tolist(); s = sz.tolist()
    W = window(n) if callable(window) else (window or max(30, n // 25))
    dep = [math.nan] * n
    score = INIT_SCORE[level] if seed_score else 0.0
    lev = level_of(score)
    strict = strict_mode and score >= 0.6
    heap = []; rt = deque(); be = deque(); vt = [0.0] * 4; V = [0.0]
    cnt = 0; t = 0.0; i = 0
    w_served = w_drop = 0; w_bits = 0.0; w_lat = 0.0; w_start = 0.0
    log = []; switches = 0; evicted = 0

    def push_heap(j, now):
        c = p[j]; wt = ADAPT_W[lev][c]
        vs = vt[c] if vt[c] > V[0] else V[0]
        vf = vs + s[j] / wt; vt[c] = vf
        heapq.heappush(heap, (vf, j))

    while i < n or cnt:
        if cnt == 0 and t < a[i]: t = a[i]
        while i < n and a[i] <= t:
            rtp = p[i] < 2
            if cnt >= BUFFER:
                if evict and strict and score >= evict_thr and rtp and be:
                    be.popleft(); cnt -= 1; w_drop += 1; evicted += 1
                else:
                    w_drop += 1; i += 1; continue
            if strict:
                (rt if rtp else be).append(i)
            else:
                push_heap(i, t)
            cnt += 1; i += 1
        if cnt:
            if strict:
                j = rt.popleft() if rt else be.popleft()
            else:
                V[0], j = heapq.heappop(heap)
            cnt -= 1; t += tx[j]; dep[j] = t
            w_served += 1; w_bits += s[j] * 8.0; w_lat += (t - a[j]) * 1000.0
            if w_served >= W:
                dur = max(t - w_start, 1e-9)
                bwn = min(w_bits / (bw * 1e6 * dur), 1.0)
                lossn = min(w_drop / (w_served + w_drop), 1.0)
                latn = min((w_lat / w_served) / 500.0, 1.0)
                new = 0.5 * bwn + 0.3 * lossn + 0.2 * latn
                score = (0.7 * new + 0.3 * score) if smooth else new
                lev = level_of(score)
                now_strict = strict_mode and score >= 0.6
                if now_strict != strict:
                    switches += 1
                    if now_strict:      # heap -> deques (arrival order)
                        for j2 in sorted(x[1] for x in heap):
                            (rt if p[j2] < 2 else be).append(j2)
                        heap = []
                    else:               # deques -> heap
                        for j2 in sorted(list(rt) + list(be)):
                            push_heap(j2, t)
                        rt.clear(); be.clear()
                    strict = now_strict
                log.append((t, score, lev, strict, bwn * 100, lossn * 100, w_lat / w_served))
                w_served = w_drop = 0; w_bits = 0.0; w_lat = 0.0; w_start = t
    return np.array(dep), {"log": log, "switches": switches, "evicted": evicted}


# ---------------------------------------------------------------- metrics
def metrics(pt, sz, arr, dep, bw, sim_time):
    out = {}
    for name, mask in [(c, pt == k) for k, c in enumerate(CLASSES)] + [("Overall", np.ones(len(pt), bool))]:
        tot = mask.sum()
        ok = mask & ~np.isnan(dep)
        lat = (dep[ok] - arr[ok]) * 1000.0
        jit = float(np.mean(np.abs(np.diff(lat)))) if len(lat) > 1 else 0.0
        inwin = ok & (dep <= sim_time)
        thr = sz[inwin].sum() * 8.0 / sim_time / 1e6
        out[name] = dict(lat=float(lat.mean()) if len(lat) else float("nan"),
                         p99=float(np.percentile(lat, 99)) if len(lat) else float("nan"),
                         jit=jit, loss=100.0 * (tot - ok.sum()) / max(tot, 1),
                         thr=float(thr), util=float(thr / bw * 100), n=int(tot))
    return out


def run_all(bw, load, level, sim_time, seed, profile="Default", algos=("FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"), **kw):
    pt, sz, arr = generate(bw, load, level, sim_time, seed, profile)
    res = {}; extra = {}
    for al in algos:
        if al == "FIFO": dep = fifo(pt, sz, arr, bw)
        elif al == "PQ": dep = pq(pt, sz, arr, bw)
        elif al == "WFQ": dep = wfq(pt, sz, arr, bw)
        elif al == "CBWFQ": dep = cbwfq(pt, sz, arr, bw)
        else: dep, extra[al] = adaptive(pt, sz, arr, bw, level, **kw)
        res[al] = metrics(pt, sz, arr, dep, bw, sim_time)
    return res, extra, (pt, sz, arr)
