import json, sys, time
import qos_sim as q

ALGOS = ("FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS")
rec = []
def add(exp, params, seed, res, extra, variant=None):
    for al, m in res.items():
        r = dict(exp=exp, params=params, seed=seed, algo=al if variant is None else variant, m=m)
        if al in extra:
            r["switches"] = extra[al]["switches"]; r["evicted"] = extra[al]["evicted"]
            if exp == "E1": r["log"] = extra[al]["log"]
        rec.append(r)

t0 = time.time()
# E1 reference run (seed 42, 10 Mbps, load 0.85, High, 60 s)
res, extra, _ = q.run_all(10, 0.85, "High", 60, 42)
add("E1", dict(bw=10, load=0.85, level="High", T=60, profile="Default"), 42, res, extra)
# E2 congestion scenarios (load 1.0), 30 seeds
for lv in ["Low", "Moderate", "High"]:
    for s in range(1, 31):
        res, extra, _ = q.run_all(10, 1.0, lv, 60, s)
        add("E2", dict(bw=10, load=1.0, level=lv, T=60, profile="Default"), s, res, extra)
print("E2", time.time() - t0, flush=True)
# E3 high-congestion reference configuration, 30 seeds
for s in range(1, 31):
    res, extra, _ = q.run_all(10, 0.85, "High", 60, s)
    add("E3", dict(bw=10, load=0.85, level="High", T=60, profile="Default"), s, res, extra)
print("E3", time.time() - t0, flush=True)
# E4 ablation
VARS = {
 "Full": {},
 "No smoothing": dict(smooth=False),
 "No eviction": dict(evict=False),
 "No initial seed": dict(seed_score=False),
 "Eviction at S>=0.8": dict(evict_thr=0.8),
 "No strict mode": dict(strict_mode=False),
 "Window = 30 pkts": dict(window=30),
 "Window = N/5": dict(window=lambda n: max(30, n // 5)),
 "No smoothing, window = 30": dict(smooth=False, window=30),
}
for s in range(1, 31):
    pt, sz, arr = q.generate(10, 0.85, "High", 60, s)
    for name, kw in VARS.items():
        dep, ex = q.adaptive(pt, sz, arr, 10, "High", **kw)
        m = q.metrics(pt, sz, arr, dep, 10, 60)
        rec.append(dict(exp="E4", params=dict(bw=10, load=0.85, level="High", T=60), seed=s, algo=name, m=m,
                        switches=ex["switches"], evicted=ex["evicted"]))
print("E4", time.time() - t0, flush=True)
# E5 workload profiles (High, load 0.85), 20 seeds
for prof in ["Default", "RT-heavy", "Bulk-heavy", "Bursty"]:
    for s in range(1, 21):
        res, extra, _ = q.run_all(10, 0.85, "High", 60, s, profile=prof)
        add("E5", dict(bw=10, load=0.85, level="High", T=60, profile=prof), s, res, extra)
print("E5", time.time() - t0, flush=True)
# E6 sweeps: bandwidth and offered-load, 10 seeds, 30 s
for bw in [5, 10, 20]:
    for s in range(1, 11):
        res, extra, _ = q.run_all(bw, 0.85, "High", 30, s)
        add("E6bw", dict(bw=bw, load=0.85, level="High", T=30, profile="Default"), s, res, extra)
for ld in [0.5, 0.7, 0.85, 1.0]:
    for s in range(1, 11):
        res, extra, _ = q.run_all(10, ld, "High", 30, s)
        add("E6load", dict(bw=10, load=ld, level="High", T=30, profile="Default"), s, res, extra)
print("E6", time.time() - t0, flush=True)
json.dump(rec, open("results.json", "w"))
print("done", time.time() - t0, flush=True)
