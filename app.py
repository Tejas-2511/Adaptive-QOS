"""
app.py
------
Streamlit dashboard for the QoS Network Traffic Simulation project.

Sections:
    1. Simulation Setup  – configure and run a simulation
    2. Results           – view metrics for the last run
    3. Algorithm Comparison – compare all five algorithms side-by-side
    4. Graphs            – visualise the results

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from simulation import run_simulation, TRAFFIC_TYPES
from database import (
    initialise_db,
    save_experiment,
    get_all_experiments,
    get_results_for_experiment,
    get_comparison_data,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QoS Network Traffic Simulator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialise DB on startup
initialise_db()

# ---------------------------------------------------------------------------
# Custom CSS for better aesthetics
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Dark theme accents */
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #3a3f55;
        border-radius: 12px;
        padding: 12px 18px;
    }

    /* Headers */
    h1 { color: #7eb8f7; }
    h2 { color: #a0c4f8; }
    h3 { color: #c3d8fa; }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #1a2744, #1e2e55);
        border-left: 4px solid #4a90d9;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    /* Traffic type badges */
    .badge-high    { color: #ff6b6b; font-weight: bold; }
    .badge-medium  { color: #ffa94d; font-weight: bold; }
    .badge-low     { color: #69db7c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar – global settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/network.png", width=72)
    st.title("QoS Simulator")
    st.markdown("---")
    st.markdown("**Computer Networks Mini Project**")
    st.markdown("*Intelligent QoS-Based Traffic Prioritization*")
    st.markdown("---")

    st.markdown("### Quick Reference")
    st.markdown("""
| Traffic Type | Priority |
|---|---|
| 🎥 Video Conf | Highest (4) |
| 🎮 Gaming | High (3) |
| 📺 Streaming | Medium (2) |
| 📁 Background | Low (1) |
""")
    st.markdown("---")
    st.caption("Algorithms: FIFO · PQ · WFQ · CBWFQ · Adaptive QoS")

# ---------------------------------------------------------------------------
# Helper colour maps
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    "FIFO":         "#778ca3",
    "PQ":           "#a29bfe",
    "WFQ":          "#74b9ff",
    "CBWFQ":        "#55efc4",
    "Adaptive QoS": "#fd79a8",
}

TRAFFIC_COLORS = {
    "Video Conferencing": "#fd79a8",
    "Gaming":             "#a29bfe",
    "Video Streaming":    "#74b9ff",
    "Background":         "#55efc4",
}

plt.style.use("dark_background")


def _style_fig(fig):
    """Apply consistent dark style to a matplotlib figure."""
    fig.patch.set_facecolor("#0f1117")
    for ax in fig.get_axes():
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="#c0c8d8")
        ax.xaxis.label.set_color("#c0c8d8")
        ax.yaxis.label.set_color("#c0c8d8")
        ax.title.set_color("#e0e8f8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3f55")


# ---------------------------------------------------------------------------
# Section 1 – Simulation Setup
# ---------------------------------------------------------------------------

st.title("🌐 Intelligent QoS-Based Network Traffic Simulation")
st.markdown(
    "Simulate how different QoS algorithms handle network congestion "
    "for latency-sensitive applications (Gaming, Video Conferencing)."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Simulation Setup",
    "📊 Results",
    "🔄 Algorithm Comparison",
    "📈 Graphs",
])

# ── Tab 1: Setup ──────────────────────────────────────────────────────────

with tab1:
    st.header("Simulation Setup")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Network Parameters")
        bandwidth = st.slider(
            "Link Bandwidth (Mbps)", min_value=10, max_value=1000,
            value=100, step=10,
            help="Total link capacity in Megabits per second."
        )
        traffic_load = st.slider(
            "Traffic Load (fraction of bandwidth)", min_value=0.1, max_value=1.5,
            value=0.8, step=0.05,
            help="Offered load relative to link capacity. >1.0 causes congestion."
        )
        simulation_time = st.slider(
            "Simulation Duration (seconds)", min_value=5, max_value=120,
            value=30, step=5,
            help="How many simulated seconds to run."
        )

    with col2:
        st.subheader("QoS Configuration")
        congestion_level = st.selectbox(
            "Congestion Scenario",
            options=["Low", "Moderate", "High"],
            index=1,
            help=(
                "Low ≈ 30% utilization · Moderate ≈ 60% · High ≈ 90%\n"
                "This controls how many packets are injected."
            ),
        )
        algorithm = st.selectbox(
            "QoS Algorithm",
            options=["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"],
            index=4,
            help=(
                "FIFO: first-in-first-out | PQ: strict priority | "
                "WFQ: weighted fair | CBWFQ: class-based WFQ | "
                "Adaptive QoS: congestion-aware (proposed method)"
            ),
        )

    # Algorithm description
    algo_descriptions = {
        "FIFO": (
            "**First In, First Out** – packets are processed in arrival order. "
            "No prioritization; all traffic types are treated equally. "
            "Simple but unfair to latency-sensitive flows during congestion."
        ),
        "PQ": (
            "**Priority Queuing** – packets are strictly prioritized by class. "
            "Higher-priority queues are always served first. "
            "Guarantees low latency for real-time traffic but may starve background flows."
        ),
        "WFQ": (
            "**Weighted Fair Queuing** – each class gets service proportional "
            "to its weight (VC=4, Gaming=3, Streaming=2, BG=1). "
            "Balances fairness with priority."
        ),
        "CBWFQ": (
            "**Class-Based WFQ** – each class has a guaranteed bandwidth share "
            "(VC=35%, Gaming=30%, Streaming=20%, BG=15%). "
            "Predictable and configurable."
        ),
        "Adaptive QoS": (
            "**Adaptive QoS (Proposed)** – dynamically adjusts weights based on "
            "real-time congestion score. As congestion increases, real-time traffic "
            "gets more resources while background traffic is throttled. "
            "Rule-based, no machine learning required."
        ),
    }

    st.markdown(
        f'<div class="info-box">ℹ️ {algo_descriptions[algorithm]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_run, col_all = st.columns([1, 2])

    with col_run:
        run_single = st.button("▶️ Run Simulation", use_container_width=True, type="primary")

    with col_all:
        run_all = st.button(
            "🔄 Run All Algorithms (for comparison)", use_container_width=True
        )
        st.caption("Runs all 5 algorithms with the same parameters and saves results.")

    # ── Run single algorithm ──
    if run_single:
        with st.spinner(f"Running {algorithm} simulation…"):
            result = run_simulation(
                algorithm=algorithm,
                simulation_time=simulation_time,
                bandwidth_mbps=bandwidth,
                traffic_load=traffic_load,
                congestion_level=congestion_level,
            )
        save_experiment(
            algorithm=algorithm,
            bandwidth=bandwidth,
            traffic_load=traffic_load,
            congestion_level=congestion_level,
            simulation_time=simulation_time,
            metrics_result=result,
        )
        st.session_state["last_result"] = result
        st.session_state["last_algo"] = algorithm
        st.session_state["last_params"] = {
            "bandwidth": bandwidth,
            "traffic_load": traffic_load,
            "congestion_level": congestion_level,
            "simulation_time": simulation_time,
        }
        st.success(f"✅ {algorithm} simulation complete! View results in the **Results** tab.")

    # ── Run all algorithms ──
    if run_all:
        all_algorithms = ["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"]
        progress = st.progress(0, text="Starting…")
        all_results = {}

        for i, algo in enumerate(all_algorithms):
            progress.progress((i) / len(all_algorithms), text=f"Running {algo}…")
            r = run_simulation(
                algorithm=algo,
                simulation_time=simulation_time,
                bandwidth_mbps=bandwidth,
                traffic_load=traffic_load,
                congestion_level=congestion_level,
            )
            save_experiment(
                algorithm=algo,
                bandwidth=bandwidth,
                traffic_load=traffic_load,
                congestion_level=congestion_level,
                simulation_time=simulation_time,
                metrics_result=r,
            )
            all_results[algo] = r

        progress.progress(1.0, text="Done!")
        st.session_state["all_results"] = all_results
        st.session_state["comparison_params"] = {
            "bandwidth": bandwidth,
            "traffic_load": traffic_load,
            "congestion_level": congestion_level,
            "simulation_time": simulation_time,
        }
        st.success("✅ All algorithms ran! Check the **Algorithm Comparison** and **Graphs** tabs.")


# ── Tab 2: Results ────────────────────────────────────────────────────────

with tab2:
    st.header("Simulation Results")

    if "last_result" not in st.session_state:
        st.info("Run a simulation in the **Setup** tab first.")
    else:
        result = st.session_state["last_result"]
        algo   = st.session_state["last_algo"]
        params = st.session_state["last_params"]

        st.markdown(
            f"**Algorithm:** `{algo}` · "
            f"**Bandwidth:** {params['bandwidth']} Mbps · "
            f"**Load:** {params['traffic_load']:.0%} · "
            f"**Congestion:** {params['congestion_level']} · "
            f"**Duration:** {params['simulation_time']} s"
        )

        # Overall metrics
        st.subheader("Overall Metrics")
        ov = result["overall"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Latency", f"{ov['avg_latency']:.2f} ms")
        c2.metric("Jitter",      f"{ov['jitter']:.2f} ms")
        c3.metric("Packet Loss", f"{ov['packet_loss']:.2f} %")
        c4.metric("Throughput",  f"{ov['throughput']:.3f} Mbps")
        c5.metric("BW Util.",    f"{ov['bw_utilization']:.1f} %")

        # Per-type breakdown
        st.subheader("Per-Traffic-Type Breakdown")

        rows = []
        for ttype, m in result["per_type"].items():
            rows.append({
                "Traffic Type": ttype,
                "Priority": TRAFFIC_TYPES[ttype]["priority"],
                "Avg Latency (ms)": m["avg_latency"],
                "Jitter (ms)": m["jitter"],
                "Packet Loss (%)": m["packet_loss"],
                "Throughput (Mbps)": m["throughput"],
                "BW Utilization (%)": m["bw_utilization"],
            })

        df = pd.DataFrame(rows).sort_values("Priority", ascending=False)
        st.dataframe(df.set_index("Traffic Type"), use_container_width=True)

        # Adaptive log (only for Adaptive QoS)
        if algo == "Adaptive QoS" and result.get("adaptive_log"):
            st.subheader("Adaptive QoS – Weight Evolution Log")
            log_df = pd.DataFrame(result["adaptive_log"])
            st.dataframe(log_df, use_container_width=True)

            # Mini chart: congestion score over time
            fig_log, ax_log = plt.subplots(figsize=(10, 3))
            ax_log.plot(
                log_df["time"], log_df["congestion_score"],
                color="#fd79a8", linewidth=2, marker="o", markersize=4
            )
            ax_log.axhline(0.3, color="#ffd93d", linestyle="--", linewidth=1, label="Moderate threshold")
            ax_log.axhline(0.6, color="#ff9f43", linestyle="--", linewidth=1, label="High threshold")
            ax_log.axhline(0.8, color="#ff6b6b", linestyle="--", linewidth=1, label="Severe threshold")
            ax_log.set_xlabel("Simulation Time (s)")
            ax_log.set_ylabel("Congestion Score")
            ax_log.set_title("Congestion Score Over Time (Adaptive QoS)")
            ax_log.legend(fontsize=8)
            _style_fig(fig_log)
            st.pyplot(fig_log)
            plt.close(fig_log)


# ── Tab 3: Algorithm Comparison ───────────────────────────────────────────

with tab3:
    st.header("Algorithm Comparison")

    if "all_results" not in st.session_state:
        st.info(
            "Click **Run All Algorithms** in the Setup tab to populate this section."
        )
    else:
        all_results = st.session_state["all_results"]
        cparams = st.session_state["comparison_params"]

        st.markdown(
            f"**Parameters:** Bandwidth={cparams['bandwidth']} Mbps · "
            f"Load={cparams['traffic_load']:.0%} · "
            f"Congestion={cparams['congestion_level']} · "
            f"Duration={cparams['simulation_time']} s"
        )

        # Build comparison table (overall metrics)
        comparison_rows = []
        for algo, res in all_results.items():
            ov = res["overall"]
            comparison_rows.append({
                "Algorithm": algo,
                "Avg Latency (ms)": ov["avg_latency"],
                "Jitter (ms)": ov["jitter"],
                "Packet Loss (%)": ov["packet_loss"],
                "Throughput (Mbps)": ov["throughput"],
                "BW Utilization (%)": ov["bw_utilization"],
            })

        df_cmp = pd.DataFrame(comparison_rows).set_index("Algorithm")

        # Highlight best values
        def highlight_best(col):
            # Lower is better for latency, jitter, packet_loss
            # Higher is better for throughput and bw_utilization
            lower_better = ["Avg Latency (ms)", "Jitter (ms)", "Packet Loss (%)"]
            styles = [""] * len(col)
            if col.name in lower_better:
                best_idx = col.idxmin()
            else:
                best_idx = col.idxmax()
            idx_pos = col.index.get_loc(best_idx)
            styles[idx_pos] = "background-color: #1a4a2a; color: #69db7c; font-weight: bold"
            return styles

        styled_df = df_cmp.style.apply(highlight_best, axis=0).format("{:.3f}")
        st.dataframe(styled_df, use_container_width=True)
        st.caption("🟢 Green cell = best value for that metric.")

        # Per-traffic-type comparison for a chosen metric
        st.subheader("Per-Traffic-Type Comparison")
        metric_choice = st.selectbox(
            "Select metric to compare per traffic type:",
            options=["avg_latency", "jitter", "packet_loss", "throughput"],
            format_func=lambda x: {
                "avg_latency": "Avg Latency (ms)",
                "jitter": "Jitter (ms)",
                "packet_loss": "Packet Loss (%)",
                "throughput": "Throughput (Mbps)",
            }[x],
        )

        traffic_types = list(TRAFFIC_TYPES.keys())
        per_type_rows = []
        for algo, res in all_results.items():
            row = {"Algorithm": algo}
            for ttype in traffic_types:
                val = res["per_type"].get(ttype, {}).get(metric_choice, 0)
                row[ttype] = val
            per_type_rows.append(row)

        df_pt = pd.DataFrame(per_type_rows).set_index("Algorithm")
        st.dataframe(df_pt.style.format("{:.3f}"), use_container_width=True)


# ── Tab 4: Graphs ─────────────────────────────────────────────────────────

with tab4:
    st.header("Graphs & Visualisations")

    if "all_results" not in st.session_state:
        st.info("Click **Run All Algorithms** in the Setup tab to generate graphs.")
    else:
        all_results = st.session_state["all_results"]
        algos = list(all_results.keys())
        colors = [ALGO_COLORS[a] for a in algos]

        # ── Graph 1: Average Latency vs Algorithm ──
        st.subheader("1. Average Latency vs Algorithm")
        latencies = [all_results[a]["overall"]["avg_latency"] for a in algos]

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        bars = ax1.bar(algos, latencies, color=colors, edgecolor="#3a3f55", linewidth=0.8)
        ax1.set_ylabel("Average Latency (ms)")
        ax1.set_title("Average Latency by QoS Algorithm (Overall)")
        for bar, val in zip(bars, latencies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(latencies) * 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#e0e8f8"
            )
        _style_fig(fig1)
        st.pyplot(fig1)
        plt.close(fig1)

        # ── Graph 2: Packet Loss vs Algorithm ──
        st.subheader("2. Packet Loss vs Algorithm")
        losses = [all_results[a]["overall"]["packet_loss"] for a in algos]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars2 = ax2.bar(algos, losses, color=colors, edgecolor="#3a3f55", linewidth=0.8)
        ax2.set_ylabel("Packet Loss (%)")
        ax2.set_title("Packet Loss by QoS Algorithm (Overall)")
        for bar, val in zip(bars2, losses):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(losses) * 0.01 if max(losses) > 0 else 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, color="#e0e8f8"
            )
        _style_fig(fig2)
        st.pyplot(fig2)
        plt.close(fig2)

        # ── Graph 3: Throughput vs Algorithm ──
        st.subheader("3. Throughput vs Algorithm")
        throughputs = [all_results[a]["overall"]["throughput"] for a in algos]

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        bars3 = ax3.bar(algos, throughputs, color=colors, edgecolor="#3a3f55", linewidth=0.8)
        ax3.set_ylabel("Throughput (Mbps)")
        ax3.set_title("Throughput by QoS Algorithm (Overall)")
        for bar, val in zip(bars3, throughputs):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(throughputs) * 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#e0e8f8"
            )
        _style_fig(fig3)
        st.pyplot(fig3)
        plt.close(fig3)

        # ── Graph 4: Latency across congestion levels ──
        st.subheader("4. Latency Under Different Congestion Levels")
        st.caption(
            "Runs each algorithm under Low / Moderate / High congestion using the "
            "same bandwidth and traffic load. Uses cached session results when available."
        )

        cparams = st.session_state["comparison_params"]

        @st.cache_data(show_spinner=False)
        def run_congestion_sweep(bandwidth, traffic_load, simulation_time):
            """Run all algorithms under all congestion levels."""
            sweep = {}
            for cong in ["Low", "Moderate", "High"]:
                sweep[cong] = {}
                for algo in ["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"]:
                    r = run_simulation(
                        algorithm=algo,
                        simulation_time=simulation_time,
                        bandwidth_mbps=bandwidth,
                        traffic_load=traffic_load,
                        congestion_level=cong,
                    )
                    sweep[cong][algo] = r["overall"]["avg_latency"]
            return sweep

        if st.button("▶️ Generate Congestion-Level Graph", key="congestion_graph_btn"):
            with st.spinner("Running congestion sweep (15 simulations)…"):
                sweep = run_congestion_sweep(
                    cparams["bandwidth"],
                    cparams["traffic_load"],
                    cparams["simulation_time"],
                )
            st.session_state["congestion_sweep"] = sweep

        if "congestion_sweep" in st.session_state:
            sweep = st.session_state["congestion_sweep"]
            cong_levels = ["Low", "Moderate", "High"]
            x = np.arange(len(cong_levels))
            width = 0.15

            fig4, ax4 = plt.subplots(figsize=(11, 5))
            for i, algo in enumerate(["FIFO", "PQ", "WFQ", "CBWFQ", "Adaptive QoS"]):
                vals = [sweep[c][algo] for c in cong_levels]
                offset = (i - 2) * width
                ax4.bar(x + offset, vals, width, label=algo,
                        color=ALGO_COLORS[algo], edgecolor="#3a3f55", linewidth=0.6)
            ax4.set_xticks(x)
            ax4.set_xticklabels(cong_levels)
            ax4.set_ylabel("Average Latency (ms)")
            ax4.set_title("Average Latency vs Congestion Level — All Algorithms")
            ax4.legend(fontsize=9)
            _style_fig(fig4)
            st.pyplot(fig4)
            plt.close(fig4)

        # ── Graph 5: Adaptive QoS weight evolution ──
        st.subheader("5. Adaptive QoS — Dynamic Weight Allocation")

        adaptive_result = all_results.get("Adaptive QoS")
        if adaptive_result and adaptive_result.get("adaptive_log"):
            log = adaptive_result["adaptive_log"]
            log_df = pd.DataFrame(log)

            traffic_types_in_log = ["Video Conferencing", "Gaming", "Video Streaming", "Background"]

            fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

            # Weight traces
            for ttype in traffic_types_in_log:
                weights_over_time = [
                    entry["weights"].get(ttype, 1)
                    for entry in log
                ]
                ax5a.plot(
                    log_df["time"], weights_over_time,
                    label=ttype, linewidth=2,
                    color=TRAFFIC_COLORS.get(ttype, "#aaa"),
                    marker="o", markersize=3,
                )
            ax5a.set_ylabel("Weight")
            ax5a.set_title("Adaptive QoS: Dynamic Weight Allocation Over Time")
            ax5a.legend(fontsize=8)

            # Congestion score trace
            ax5b.fill_between(
                log_df["time"], log_df["congestion_score"],
                color="#fd79a8", alpha=0.4
            )
            ax5b.plot(
                log_df["time"], log_df["congestion_score"],
                color="#fd79a8", linewidth=2
            )
            for thresh, label, col in [
                (0.3, "Moderate", "#ffd93d"),
                (0.6, "High",     "#ff9f43"),
                (0.8, "Severe",   "#ff6b6b"),
            ]:
                ax5b.axhline(thresh, color=col, linestyle="--", linewidth=1, label=label)
            ax5b.set_xlabel("Simulation Time (s)")
            ax5b.set_ylabel("Congestion Score")
            ax5b.set_title("Congestion Score (drives weight updates)")
            ax5b.set_ylim(0, 1)
            ax5b.legend(fontsize=8)

            _style_fig(fig5)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
        else:
            st.info(
                "Run the simulation with **Adaptive QoS** selected (or run all algorithms) "
                "to see the weight evolution graph."
            )

        # ── Graph 6: Per-type latency heatmap ──
        st.subheader("6. Per-Traffic-Type Latency Heatmap")

        heat_data = {}
        for algo, res in all_results.items():
            heat_data[algo] = {
                ttype: res["per_type"].get(ttype, {}).get("avg_latency", 0)
                for ttype in TRAFFIC_TYPES
            }

        df_heat = pd.DataFrame(heat_data).T   # rows=algos, cols=traffic types

        fig6, ax6 = plt.subplots(figsize=(10, 4))
        im = ax6.imshow(df_heat.values, aspect="auto", cmap="RdYlGn_r")
        ax6.set_xticks(range(len(df_heat.columns)))
        ax6.set_xticklabels(df_heat.columns, rotation=20, ha="right")
        ax6.set_yticks(range(len(df_heat.index)))
        ax6.set_yticklabels(df_heat.index)
        ax6.set_title("Average Latency Heatmap (ms) — Algorithms × Traffic Types")
        plt.colorbar(im, ax=ax6, label="Avg Latency (ms)")
        for i in range(len(df_heat.index)):
            for j in range(len(df_heat.columns)):
                val = df_heat.iloc[i, j]
                ax6.text(j, i, f"{val:.1f}", ha="center", va="center",
                         fontsize=8, color="white")
        _style_fig(fig6)
        st.pyplot(fig6)
        plt.close(fig6)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "QoS Network Traffic Simulation · Computer Networks Mini Project · "
    "FIFO | PQ | WFQ | CBWFQ | Adaptive QoS"
)
