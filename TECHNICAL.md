# Adaptive QoS – Technical Reference

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Descriptions](#2-module-descriptions)
3. [Packet Model](#3-packet-model)
4. [Traffic Generation (`simulation.py` & `qos_algorithms.py`)](#4-traffic-generation)
5. [QoS Algorithm Implementations (`qos_algorithms.py`)](#5-qos-algorithm-implementations)
   - 5.1 [Shared Constants & Parameters](#51-shared-constants--parameters)
   - 5.2 [FIFO – First In, First Out](#52-fifo--first-in-first-out)
   - 5.3 [PQ – Priority Queuing](#53-pq--priority-queuing)
   - 5.4 [WFQ – Weighted Fair Queuing](#54-wfq--weighted-fair-queuing)
   - 5.5 [CBWFQ – Class-Based WFQ](#55-cbwfq--class-based-wfq)
   - 5.6 [Adaptive QoS (Proposed Solution)](#56-adaptive-qos-proposed-solution)
6. [Metrics Computation (`qos_algorithms.metrics`)](#6-metrics-computation)
7. [Database Layer (`database.py`)](#7-database-layer)
8. [Congestion Scenarios & Overload Modeling](#8-congestion-scenarios--overload-modeling)
9. [Reference Results & Comparative Evaluation](#9-reference-results--comparative-evaluation)
10. [Standalone Experiment Suite (`new/`)](#10-standalone-experiment-suite)
11. [Viva / Interview Technical Q&A](#11-viva--interview-technical-qa)

---

## 1. Architecture Overview

```
+---------------------------------------------------------------+
|                     app.py (Streamlit UI)                     |
|  +--------------+  +-------------------+  +----------------+  |
|  |  Simulation  |  | Algorithm Compare |  | Graphs/History |  |
|  |  Setup Tab   |  |       Tab         |  |      Tabs      |  |
|  +-------+------+  +---------+---------+  +--------+-------+  |
+----------|-------------------|---------------------|----------+
           |                   |                     |
           v                   v                     v
    simulation.run_simulation()              database.get_*()
           |
     +-----+-----------------------------------------+
     |                  simulation.py                |
     |  - High-performance vectorized runner         |
     |  - Schema adapter for UI & database logging   |
     +-----+-----------------------------------------+
           | (pt, sz, arr) parallel arrays
           v
    qos_algorithms.{fifo, pq, wfq, cbwfq, adaptive}()
           | dep[] (departure timestamps, NaN = dropped)
           v
    qos_algorithms.metrics()
           | {per_type, overall}
           v
    database.save_experiment()
           |
    SQLite (qos_results.db)
```

### Data Flow per Simulation Run
1. **Parameter Acquisition:** The UI collects network bandwidth, traffic load, duration, and congestion scenario, calling `simulation.run_simulation()`.
2. **Vectorized Traffic Generation:** `qos_algorithms.generate()` creates aligned NumPy parallel arrays: class index `pt`, packet size `sz`, and arrival timestamp `arr`.
3. **Queue Scheduling:** The selected scheduler (`fifo`, `pq`, `wfq`, `cbwfq`, or `adaptive`) executes in vectorized memory, producing an array of departure timestamps `dep` (with `NaN` denoting dropped packets).
4. **Metrics Calculation:** `qos_algorithms.metrics()` applies vectorized boolean masks to calculate latency, P99 latency, jitter, packet loss, throughput, and link utilization.
5. **Persistence & Presentation:** `simulation.py` transforms the raw metric arrays into the schema expected by `app.py` and `database.py`, logs mode transitions, persists the run to SQLite, and updates the reactive Streamlit dashboard.

---

## 2. Module Descriptions

| File / Folder | Responsibility |
|---|---|
| `app.py` | Streamlit interactive web dashboard; user controls, live metric cards, algorithm comparisons, and dynamic Matplotlib visualisations |
| `simulation.py` | Top-level simulation orchestrator; generates vectorized packet traces, invokes schedulers, formats metrics, and exposes backwards-compatible APIs |
| `qos_algorithms.py` | Core mathematical engine containing all five vectorized schedulers (`fifo`, `pq`, `wfq`, `cbwfq`, `adaptive`), generator (`generate`), metrics evaluator (`metrics`), and batch runner (`run_all`) |
| `database.py` | SQLite persistence layer; initializes tables, records experiments, and serves comparison history |
| `requirements.txt` | Python runtime dependencies: `streamlit`, `numpy`, `pandas`, `matplotlib`, `scipy` |
| `qos_results.db` | Auto-generated SQLite database preserving experiment records and comparative metrics |
| `new/` | Standalone research & publication experiment harness containing `run_experiments.py` (E1–E6 benchmark suite), `analyze.py` (hypothesis testing & figure rendering), and `qos_sim.py` |

---

## 3. Packet Model

The simulator operates on aligned, parallel NumPy arrays for memory efficiency, cache locality, and vector speed:

| Array | Type | Description |
|---|---|---|
| `pt` | `np.int8` | Traffic class index: `0` = Video Conferencing, `1` = Gaming, `2` = Video Streaming, `3` = Background |
| `sz` | `np.int32` | Packet payload size in bytes |
| `arr` | `np.float64` | Packet arrival timestamp in seconds (monotonically non-decreasing) |
| `dep` | `np.float64` | Packet departure timestamp in seconds (`NaN` represents a dropped packet) |

For UI tables and backward compatibility, `simulation.py` can materialize individual packet records on demand:
- `packet_id`: Sequential integer ID
- `traffic_type`: Descriptive string (`"Video Conferencing"`, etc.)
- `size`: Size in bytes
- `arrival_time`: Ingress timestamp
- `priority`: Static priority integer (4 = highest, 1 = lowest)

---

## 4. Traffic Generation

**Implementation:** `qos_algorithms.generate()` and `simulation.generate_packets()`

### 4.1 Traffic Class Parameters

| Traffic Class | Priority | Packet Size Range | WFQ Weight | CBWFQ Share | Default Mix | Application Archetype |
|---|---|---|---|---|---|---|
| **Video Conferencing** | 4 (highest) | 100 – 300 B | 4 | 35% | 20% | UDP voice/video frames (Zoom, Teams) |
| **Gaming** | 3 | 50 – 200 B | 3 | 30% | 25% | UDP state updates (low payload, tight deadline) |
| **Video Streaming** | 2 | 800 – 1500 B | 2 | 20% | 30% | TCP segmented media (YouTube, Netflix) |
| **Background** | 1 (lowest) | 500 – 1400 B | 1 | 15% | 25% | Bulk TCP transfers (file downloads, backups) |

### 4.2 Workload Mix Profiles (`MIXES`)

The generator supports four traffic distribution profiles:

```python
MIXES = {
    "Default":    [0.20, 0.25, 0.30, 0.25],   # Balanced general workload
    "RT-heavy":   [0.35, 0.30, 0.20, 0.15],   # Real-time voice/gaming dominated
    "Bulk-heavy": [0.10, 0.15, 0.35, 0.40],   # Heavy downloads and streaming
    "Bursty":     [0.20, 0.25, 0.30, 0.25],   # Periodic on/off active bursts
}
```

### 4.3 Offered Load & Overload Multipliers (`MULT`)

The offered traffic volume intentional scales beyond raw bandwidth to simulate controlled queuing stress:

$$\text{Offered Bytes} = \text{Multiplier} \times \text{Load Slider} \times \frac{\text{Bandwidth (bps)}}{8} \times \text{Duration (s)}$$

| Congestion Level | Overload Multiplier (`MULT`) | Queuing Stress Description |
|---|---|---|
| **Low** | 0.55× | ~55% link capacity — minimal queuing delay, zero drop |
| **Moderate** | 1.10× | 10% capacity overload — moderate queue build-up and initial drops |
| **High** | 1.65× | 65% capacity overload — persistent buffer saturation, severe packet drop |

### 4.4 Arrival Time Dynamics

- **Standard Profiles (Default, RT-heavy, Bulk-heavy):**
  $$\text{arrival}_i = \left(\frac{\text{bytes\_before}_i}{\text{total\_bytes}}\right)^{0.85} \times T + \mathcal{U}(0, 0.002)$$
  The exponent $0.85$ introduces a slight front-load burst skew, testing algorithm responsiveness.
- **Bursty Profile:**
  Divides the timeline into 5-second cycles. Within each cycle, **70% of traffic arrives in the first 30% of the period**, accurately reflecting real-world intermittent voice activity.

---

## 5. QoS Algorithm Implementations

**Implementation:** `qos_algorithms.py`

### 5.1 Shared Constants & Parameters

```python
BUFFER = 200                                                  # Finite queue buffer capacity (packets)
CLASSES = ["Video Conferencing", "Gaming", "Video Streaming", "Background"]
SIZE_RANGE = [(100, 300), (50, 200), (800, 1500), (500, 1400)]
WFQ_W = [4, 3, 2, 1]                                         # WFQ static class weights
CBWFQ_S = [0.35, 0.30, 0.20, 0.15]                           # CBWFQ guaranteed bandwidth fractions
INIT_SCORE = {"Low": 0.15, "Moderate": 0.60, "High": 0.85}  # Initial seed scores
ADAPT_W = {
    0: (4, 3, 2, 1),   # Low congestion
    1: (6, 5, 2, 1),   # Moderate congestion
    2: (8, 7, 2, 1),   # High congestion
    3: (10, 9, 3, 1),  # Severe congestion
}
```

### 5.2 FIFO – First In, First Out

- **Mechanism:** Packets are transmitted strictly in arrival sequence. When queue occupancy hits `BUFFER` (200 packets), incoming arrivals suffer tail drop.
- **Complexity:** $\mathcal{O}(n)$.
- **Limitation:** Head-of-line (HOL) blocking. Large 1500-byte background frames hold the transmission link, inducing high latency and jitter on small, time-critical packets.

### 5.3 PQ – Priority Queuing

- **Mechanism:** Packets are sorted into strict priority levels (Class 0 > Class 1 > Class 2 > Class 3). The scheduler always serves the highest priority non-empty queue.
- **Complexity:** $\mathcal{O}(n)$.
- **Limitation:** Starvation of best-effort flows. During sustained overload, Background and Video Streaming queues are never granted transmission slots, causing extreme packet drop and near-zero throughput for bulk transfers.

### 5.4 WFQ – Weighted Fair Queuing

- **Mechanism:** Approximates Generalized Processor Sharing (GPS). Each packet receives a virtual finish tag:
  $$V_{\text{start}} = \max(V_{\text{last\_class}}, V_{\text{virtual\_time}}), \quad V_{\text{finish}} = V_{\text{start}} + \frac{\text{size}}{\text{weight}_c}$$
  Packets depart in ascending order of $V_{\text{finish}}$ using a min-heap.
- **Complexity:** $\mathcal{O}(n \log k)$, where $k$ is active queues.
- **Limitation:** Static allocation. In severe overload, assigning 4:3:2:1 weights cannot prevent real-time queues from overflowing, leading to unacceptably high drop rates for gaming and video conferences.

### 5.5 CBWFQ – Class-Based Weighted Fair Queuing

- **Mechanism:** Extends WFQ by assigning guaranteed bandwidth percentages (35%, 30%, 20%, 15%). Effective service rate corresponds to configured link allocation.
- **Complexity:** $\mathcal{O}(n \log k)$.
- **Limitation:** Lack of adaptability. Unused allocations cannot dynamically invert priority during transient burst emergencies.

### 5.6 Adaptive QoS (Proposed Solution)

Adaptive QoS dynamically optimizes scheduling rules across three synergistic mechanisms:

```
                      +-----------------------------+
                      |   Packet Arrives at Link    |
                      +--------------+--------------+
                                     |
                                     v
                        Is Buffer Occupancy >= 200?
                                /         \
                             Yes           No
                             /               \
         Is Strict Mode & S >= 0.6 & RT Pkt?  Enqueue to RT/BE or WFQ Heap
                   /           \
                 Yes            No
                 /                \
        Evict 1 Oldest BE Pkt;     Tail Drop Incoming Pkt
        Enqueue New RT Pkt
                                     |
                                     v
                    Transmit Packet (Advance Link Time)
                                     |
                        Window Served >= W Packets?
                                /         \
                             Yes           No
                             /               \
               Compute Window Metrics;         Continue Transmission
               Calculate New Score S_t;
               EMA Smoothing; Update Mode
```

#### 5.6.1 Live Congestion Scoring

Every sliding evaluation window ($W = \max(30, \lfloor n / 25 \rfloor)$ served packets), link state is evaluated:

$$S_{\text{raw}} = 0.50 \cdot U_{\text{bw}} + 0.30 \cdot P_{\text{loss}} + 0.20 \cdot \min\left(\frac{L_{\text{avg}}}{500\text{ ms}}, 1.0\right)$$

Where:
- $U_{\text{bw}} = \min(\text{bits\_served} / (\text{bandwidth} \times \Delta t), 1.0)$
- $P_{\text{loss}} = \text{dropped} / (\text{served} + \text{dropped})$
- $L_{\text{avg}}$ is the mean queuing delay across the window

#### 5.6.2 Exponential Moving Average (EMA) Smoothing

To eliminate high-frequency weight flapping, the congestion score is smoothed over time:

$$S_t = 0.70 \cdot S_{\text{raw}} + 0.30 \cdot S_{t-1}$$

#### 5.6.3 Dual-Mode Hybrid Scheduling

- **Normal / Moderate ($S_t < 0.60$):** Operates in **Dynamic WFQ Mode**. Packets are scheduled through a fair queue with weights mapped from $S_t$:
  - Level 0 ($S_t < 0.30$): Weights `(4, 3, 2, 1)`
  - Level 1 ($0.30 \le S_t < 0.60$): Weights `(6, 5, 2, 1)`
- **Congested / Severe ($S_t \ge 0.60$):** Transitions to **Strict Priority Protection Mode**.
  - Dual FIFO deques: Real-Time (`rt`: VC + Gaming) and Best-Effort (`be`: Streaming + Background).
  - High-priority frames are served immediately; best-effort queues are served whenever real-time queues empty, avoiding absolute starvation.

#### 5.6.4 Predictive Initial Seeding

Avoids warm-up latency by seeding initial scores directly from the congestion scenario:
`INIT_SCORE = {"Low": 0.15, "Moderate": 0.60, "High": 0.85}`.

#### 5.6.5 Active Buffer Eviction

When the 200-packet buffer is completely full, if an arriving frame belongs to real-time traffic and $S_t \ge 0.60$, the scheduler **drops the oldest background packet from `be`** to make immediate space for the real-time arrival.

---

## 6. Metrics Computation

**Implementation:** `qos_algorithms.metrics(pt, sz, arr, dep, bw, sim_time)`

All metrics are computed using vectorized NumPy boolean slicing over packet arrivals:

| Metric | Mathematical Definition | Units | Implementation Details |
|---|---|---|---|
| **Average Latency** | $\frac{1}{N_{\text{deliv}}} \sum (\text{dep}_i - \text{arr}_i) \times 1000$ | ms | Evaluated exclusively over delivered (non-NaN) frames |
| **P99 Latency** | $\text{Percentile}_{99}(\text{latency})$ | ms | Evaluates worst-case jitter buffer requirements |
| **Mean Jitter** | $\frac{1}{N-1} \sum \|\text{lat}_i - \text{lat}_{i-1}\|$ | ms | Absolute difference between consecutive packet latencies |
| **Packet Loss** | $\frac{N_{\text{total}} - N_{\text{deliv}}}{N_{\text{total}}} \times 100$ | % | Percentage of frames discarded due to buffer overflow |
| **Throughput** | $\frac{\sum \text{sz}_{\text{deliv}} \times 8}{\text{sim\_time} \times 10^6}$ | Mbps | Goodput achieved within the simulation window |
| **Link Utilization** | $\frac{\text{Throughput}}{\text{Bandwidth}} \times 100$ | % | Total network transmission capacity utilized |

---

## 7. Database Layer

**Implementation:** `database.py`  
**Storage Engine:** SQLite (`qos_results.db`)

### Schema Architecture

```sql
CREATE TABLE IF NOT EXISTS experiments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT,
    algorithm        TEXT,
    bandwidth        REAL,
    traffic_load     REAL,
    congestion_level TEXT,
    simulation_time  REAL
);

CREATE TABLE IF NOT EXISTS results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  INTEGER,
    traffic_type   TEXT,
    avg_latency    REAL,
    jitter         REAL,
    packet_loss    REAL,
    throughput     REAL,
    bw_utilization REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
```

---

## 8. Congestion Scenarios & Overload Modeling

| Scenario | Load Multiplier | Total Offered Load (at slider = 0.85, 10 Mbps) | Queue Dynamics Observed |
|---|---|---|---|
| **Low** | 0.55× | 4.68 Mbps (~47% capacity) | Queues drain immediately; zero drops, near-wire transmission latency |
| **Moderate** | 1.10× | 9.35 Mbps (~94% capacity) | Temporary buffer builds during packet size bursts; minor tail drop |
| **High** | 1.65× | 14.03 Mbps (140% capacity) | Continuous buffer pressure; standard algorithms drop 20–35% of packets |

---

## 9. Reference Results & Comparative Evaluation

Results from a 30-seed benchmark under High Congestion (10 Mbps, 0.85 load, 60 seconds):

### 9.1 Overall System Performance

| Algorithm | Mean Latency (ms) | P99 Latency (ms) | Jitter (ms) | Packet Loss (%) | Link Utilization (%) |
|---|---|---|---|---|---|
| **FIFO** | 100.68 | 499.12 | 0.45 | 32.04% | 99.8% |
| **PQ** | 100.92 | 498.74 | 148.22 | 32.38% | 99.7% |
| **WFQ** | 100.80 | 498.91 | 147.16 | 32.31% | 99.8% |
| **CBWFQ** | 100.50 | 497.80 | 132.21 | 32.25% | 99.8% |
| **Adaptive QoS** | **78.92** | **442.25** | **101.73** | **24.32%** | **99.1%** |

### 9.2 Real-Time Protection (Video Conferencing)

| Algorithm | VC Mean Latency (ms) | VC Packet Loss (%) | Statistical Significance vs Adaptive |
|---|---|---|---|
| **FIFO** | 99.91 ms | 36.42% | $p < 10^{-15}$, Cohen's $d_z = -483.6$ |
| **PQ** | 0.63 ms | 15.82% | Adaptive achieves equivalent latency ($p > 0.05$) |
| **WFQ** | 0.70 ms | 36.91% | $p < 10^{-15}$ |
| **CBWFQ** | 0.74 ms | 36.88% | $p < 10^{-15}$ |
| **Adaptive QoS** | **0.64 ms** | **15.81%** | Baseline reference |

### 9.3 Starvation Prevention (Background Traffic)

| Algorithm | BG Latency (ms) | BG Throughput (Mbps) | Starvation Status |
|---|---|---|---|
| **FIFO** | 101.15 ms | 2.41 Mbps | Unfair sharing |
| **PQ** | 384.43 ms | 0.81 Mbps | **Severe starvation** |
| **WFQ** | 381.36 ms | 1.82 Mbps | High queuing delay |
| **CBWFQ** | 14.68 ms | 1.85 Mbps | Fixed slice delay |
| **Adaptive QoS** | **262.03 ms** | **1.74 Mbps** | **Protected throughput, zero starvation** |

---

## 10. Standalone Experiment Suite (`new/`)

The [`new/`](file:///c:/Users/tejas/Documents/tejas/codin/Adaptive-QOS/new) folder contains an independent scientific evaluation pipeline:

- **`qos_sim.py`:** Standalone reference simulation module with vectorized generator, schedulers, and metrics.
- **`run_experiments.py`:** Automated experiment suite:
  - **E1:** Reference scenario (seed 42, 10 Mbps, load 0.85, 60s)
  - **E2:** Congestion sweep across Low, Moderate, High (30 seeds each)
  - **E3:** 30-seed statistical benchmark
  - **E4:** Ablation study testing smoothing, eviction, initial seeding, and window sizes
  - **E5:** Traffic profiles (Default, RT-heavy, Bulk-heavy, Bursty)
  - **E6:** Parameter sweeps across bandwidth (5, 10, 20 Mbps) and load (0.5 – 1.0)
- **`analyze.py`:** Reads `results.json`, performs paired t-tests and Wilcoxon signed-rank tests, calculates effect sizes, and outputs publication-ready vector PDFs and PNGs in `new/figs/`:
  - `fig_scen.pdf` / `.png`: Latency and loss across congestion levels
  - `fig_class.pdf` / `.png`: Per-class performance breakdown
  - `fig_trace.pdf` / `.png`: Real-time score & weight adaptation trace
  - `fig_ablation.pdf` / `.png`: Component contribution ablation
  - `fig_profiles.pdf` / `.png`: Workload profile and parameter sweeps

---

## 11. Viva / Interview Technical Q&A

**Q: Why was the legacy list-of-dicts simulation replaced with NumPy parallel arrays?**  
A: Generating and managing hundreds of thousands of Python dictionary objects introduces severe memory allocator overhead and cache misses. Storing packet attributes in aligned contiguous NumPy arrays (`pt`, `sz`, `arr`, `dep`) improves simulation speed by over 98%, enabling 30-seed statistical sweeps and instant UI rendering while preserving 100% computational fidelity.

**Q: What is the computational complexity of Adaptive QoS?**  
A: Per-packet scheduling is $\mathcal{O}(\log k)$ in dynamic WFQ mode (where $k \le 4$ classes, effectively constant time) and $\mathcal{O}(1)$ in strict-priority deque mode. Periodic congestion score updates cost $\mathcal{O}(W)$ every $W$ packets, amortizing to $\mathcal{O}(1)$ per transmitted packet.

**Q: Why does Priority Queuing (PQ) starve Background traffic, and how does Adaptive QoS solve it?**  
A: PQ always inspects queues in priority order. During high real-time traffic volume, the video conferencing and gaming queues are continuously replenished, so lower queues never get served. Adaptive QoS uses a dual-mode hybrid: in strict mode, best-effort traffic is served whenever real-time queues empty; in lower congestion, fair weights guarantee bandwidth to all classes.

**Q: What does the buffer limit of 200 packets represent?**  
A: Assuming an average packet size of ~600 bytes, 200 packets correspond to ~120 KB of queue storage. On a 10 Mbps link, this represents a buffer absorption window of ~96 ms, closely modeling realistic edge router buffer sizes for 100 ms Round Trip Time (RTT) paths.

**Q: What is the role of active packet eviction?**  
A: When a burst fills the router queue to capacity, traditional tail-drop discards incoming packets regardless of priority. Active eviction inspects incoming real-time frames during high congestion, discarding the oldest background frame already in the buffer to admit the critical packet, dropping video conferencing loss from 36% to under 16%.
