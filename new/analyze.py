import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
R = json.load(open("results.json"))
CL = ["Video Conferencing", "Gaming", "Video Streaming", "Background", "Overall"]
AL = ["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"]
COL = {"FIFO": "#7f7f7f", "PQ": "#1f77b4", "WFQ": "#2ca02c", "CBWFQ": "#9467bd", "Adaptive QoS": "#d62728"}
rows = []
for r in R:
    p = r["params"]
    for c, v in r["m"].items():
        rows.append(dict(exp=r["exp"], level=p.get("level"), bw=p.get("bw"), load=p.get("load"), T=p.get("T"), profile=p.get("profile", "Default"),
                         seed=r["seed"], algo=r["algo"], cls=c, sw=r.get("switches"), ev=r.get("evicted"), **v))
D = pd.DataFrame(rows); D.to_pickle("D.pkl")
out = {}
def ms(x): return f"{np.mean(x):.2f}"

# E1 reference
e1 = D[D.exp == "E1"]
out["E1"] = e1.pivot_table(index=["cls"], columns="algo", values=["lat", "jit", "loss", "thr"]).round(2).to_dict()
# E2 scenario means
e2 = D[(D.exp == "E2")]
g = e2.groupby(["level", "algo", "cls"])[["lat", "jit", "loss", "thr"]].agg(["mean", "std"])
g.to_pickle("E2.pkl")
# E3
e3 = D[D.exp == "E3"]
g3 = e3.groupby(["algo", "cls"])[["lat", "p99", "jit", "loss", "thr"]].agg(["mean", "std"]); g3.to_pickle("E3.pkl")
# tests: Adaptive vs each baseline, paired by seed
tests = []
for cls, met in [("Video Conferencing", "loss"), ("Gaming", "loss"), ("Video Conferencing", "lat"), ("Gaming", "lat"), ("Video Streaming", "lat"), ("Background", "loss"), ("Background", "lat"), ("Overall", "loss"), ("Overall", "lat"), ("Overall", "jit")]:
    a = e3[(e3.algo == "Adaptive QoS") & (e3.cls == cls)].sort_values("seed")[met].values
    for b in ["FIFO", "PQ", "WFQ", "CBWFQ"]:
        x = e3[(e3.algo == b) & (e3.cls == cls)].sort_values("seed")[met].values
        d = a - x
        t, p_t = stats.ttest_rel(a, x)
        try: w, p_w = stats.wilcoxon(a, x)
        except Exception: p_w = float("nan")
        se = d.std(ddof=1) / np.sqrt(len(d)); ci = (d.mean() - stats.t.ppf(.975, len(d) - 1) * se, d.mean() + stats.t.ppf(.975, len(d) - 1) * se)
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
        tests.append(dict(cls=cls, met=met, base=b, adapt=a.mean(), basev=x.mean(), diff=d.mean(), lo=ci[0], hi=ci[1], p_t=p_t, p_w=p_w, dz=dz, rel=(x.mean() - a.mean()) / x.mean() * 100))
T = pd.DataFrame(tests); T.to_pickle("tests.pkl")
# E4 ablation
e4 = D[D.exp == "E4"]
ab = e4.groupby(["algo", "cls"])[["lat", "jit", "loss"]].mean()
sw = e4[e4.cls == "Overall"].groupby("algo")[["sw", "ev"]].agg(["mean", "std"])
ab.to_pickle("E4.pkl"); sw.to_pickle("E4sw.pkl")
abt = []
for v in e4.algo.unique():
    if v == "Full": continue
    for cls, met in [("Video Conferencing", "loss"), ("Gaming", "loss"), ("Video Conferencing", "lat"), ("Gaming", "lat")]:
        a = e4[(e4.algo == "Full") & (e4.cls == cls)].sort_values("seed")[met].values
        x = e4[(e4.algo == v) & (e4.cls == cls)].sort_values("seed")[met].values
        d = x - a
        try: p_w = stats.wilcoxon(x, a).pvalue
        except Exception: p_w = float("nan")
        abt.append(dict(variant=v, cls=cls, met=met, full=a.mean(), var=x.mean(), p_t=stats.ttest_rel(x, a).pvalue, p_w=p_w))
pd.DataFrame(abt).to_pickle("E4tests.pkl")
# E5, E6
D[D.exp == "E5"].groupby(["profile", "algo", "cls"])[["lat", "jit", "loss", "thr"]].agg(["mean", "std"]).to_pickle("E5.pkl")
D[D.exp == "E6bw"].groupby(["bw", "algo", "cls"])[["lat", "jit", "loss", "thr"]].mean().to_pickle("E6bw.pkl")
D[D.exp == "E6load"].groupby(["load", "algo", "cls"])[["lat", "jit", "loss", "thr"]].mean().to_pickle("E6load.pkl")

# ---------------- figures
def bars(ax, df, cls, met, ylabel, groups, key):
    w = 0.16
    for k, a in enumerate(AL):
        m = [df[(df[key] == gv) & (df.algo == a) & (df.cls == cls)][met].mean() for gv in groups]
        s = [df[(df[key] == gv) & (df.algo == a) & (df.cls == cls)][met].std() for gv in groups]
        ax.bar(np.arange(len(groups)) + (k - 2) * w, m, w, yerr=s, color=COL[a], label=a, capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_ylabel(ylabel)
# fig: scenarios (VC loss, VC lat, BG lat)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
bars(axs[0], e2, "Video Conferencing", "loss", "VC packet loss (%)", ["Low", "Moderate", "High"], "level")
bars(axs[1], e2, "Video Conferencing", "lat", "VC mean latency (ms)", ["Low", "Moderate", "High"], "level")
bars(axs[2], e2, "Background", "lat", "BG mean latency (ms)", ["Low", "Moderate", "High"], "level")
axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig("figs/fig_scen.pdf"); plt.close()
# fig: per-class high
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
cl4 = CL[:4]
for ax, met, yl in zip(axs, ["loss", "lat", "jit"], ["Packet loss (%)", "Mean latency (ms)", "Jitter (ms)"]):
    w = 0.16
    for k, a in enumerate(AL):
        m = [e3[(e3.algo == a) & (e3.cls == c)][met].mean() for c in cl4]; s = [e3[(e3.algo == a) & (e3.cls == c)][met].std() for c in cl4]
        ax.bar(np.arange(4) + (k - 2) * w, m, w, yerr=s, color=COL[a], label=a, capsize=1.2, error_kw=dict(lw=0.5))
    ax.set_xticks(range(4)); ax.set_xticklabels(["VC", "Game", "Stream", "BG"]); ax.set_ylabel(yl)
axs[0].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig("figs/fig_class.pdf"); plt.close()
# fig: adaptive trace E1
tr = [r for r in R if r["exp"] == "E1" and r["algo"] == "Adaptive QoS"][0]["log"]
tt = [x[0] for x in tr]; sc = [x[1] for x in tr]; stt = [x[3] for x in tr]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
axs[0].plot(tt, sc, "o-", ms=3, color="#d62728"); axs[0].axhline(0.6, ls="--", c="k", lw=0.7); axs[0].axhline(0.3, ls=":", c="k", lw=0.7)
axs[0].text(tt[-1], 0.61, "strict-priority threshold", fontsize=6, ha="right", va="bottom"); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Smoothed congestion score $S_t$")
axs[0].set_ylim(0.3, 0.75)
for t_, s_ in zip(tt, stt): axs[0].axvspan(t_ - 3, t_, ymin=0, ymax=0.04, color="#d62728" if s_ else "#1f77b4")
bw_ = [x[4] for x in tr]; ls_ = [x[5] for x in tr]; la_ = [x[6] for x in tr]
axs[1].plot(tt, ls_, "s-", ms=3, label="window loss (%)", color="#ff7f0e"); axs[1].plot(tt, la_, "^-", ms=3, label="window latency (ms)", color="#1f77b4")
axs[1].set_xlabel("Time (s)"); axs[1].legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig("figs/fig_trace.pdf"); plt.close()
# fig: ablation
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
order = ["Full", "No smoothing", "No eviction", "No initial seed", "Eviction at S>=0.8", "No strict mode", "Window = 30 pkts", "Window = N/5", "No smoothing, window = 30"]
for ax, (cls, met, yl) in zip(axs, [("Video Conferencing", "loss", "VC loss (%)"), ("Video Conferencing", "lat", "VC latency (ms)"), ("Overall", "sw", "Mode switches per run")]):
    if met == "sw": vals = [sw.loc[o, ("sw", "mean")] for o in order]; err = [sw.loc[o, ("sw", "std")] for o in order]
    else:
        vals = [e4[(e4.algo == o) & (e4.cls == cls)][met].mean() for o in order]; err = [e4[(e4.algo == o) & (e4.cls == cls)][met].std() for o in order]
    ax.barh(range(len(order)), vals, xerr=err, color=["#d62728"] + ["#888"] * (len(order) - 1), error_kw=dict(lw=0.5), capsize=1.2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order if ax is axs[0] else [""] * len(order), fontsize=6.5); ax.invert_yaxis(); ax.set_xlabel(yl)
fig.tight_layout(); fig.savefig("figs/fig_ablation.pdf"); plt.close()
# fig: profiles & sweeps
e5 = D[D.exp == "E5"]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
bars(axs[0], e5, "Video Conferencing", "loss", "VC loss (%)", ["Default", "RT-heavy", "Bulk-heavy", "Bursty"], "profile")
axs[0].set_xticklabels(["Default", "RT-heavy", "Bulk", "Bursty"], fontsize=6.5)
e6 = D[D.exp == "E6load"]; e6b = D[D.exp == "E6bw"]
for a in AL:
    axs[1].plot([0.5, 0.7, 0.85, 1.0], [e6[(e6.load == l) & (e6.algo == a) & (e6.cls == "Video Conferencing")].loss.mean() for l in [0.5, 0.7, 0.85, 1.0]], "o-", ms=3, color=COL[a], label=a)
    axs[2].plot([5, 10, 20], [e6b[(e6b.bw == b) & (e6b.algo == a) & (e6b.cls == "Video Conferencing")].loss.mean() for b in [5, 10, 20]], "o-", ms=3, color=COL[a])
axs[1].set_xlabel("Traffic-load slider"); axs[1].set_ylabel("VC loss (%)"); axs[2].set_xlabel("Link bandwidth (Mbps)"); axs[2].set_ylabel("VC loss (%)"); axs[2].set_xticks([5, 10, 20])
fig.tight_layout(); fig.savefig("figs/fig_profiles.pdf"); plt.close()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(g3.round(2)); print(T.round(4).to_string()); print(ab.round(2)); print(sw.round(2))
