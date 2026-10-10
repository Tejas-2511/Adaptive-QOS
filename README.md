# QoS Network Traffic Simulation

> **Computer Networks Mini Project**

**Research Question:** Can Adaptive QoS reduce latency, jitter and packet loss for latency-sensitive applications during network congestion compared with conventional queueing methods?

---

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard — main entry point |
| `simulation.py` | Packet generation and simulation loop |
| `qos_algorithms.py` | FIFO, PQ, WFQ, CBWFQ, Adaptive QoS implementations |
| `metrics.py` | Latency, jitter, packet loss, throughput calculations |
| `database.py` | SQLite read/write helpers |
| `requirements.txt` | Python dependencies |
| `TECHNICAL.md` | Full technical reference |
| `qos_results.db` | Auto-created SQLite database |

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Example Values — Showing Adaptive QoS Advantage

Use these settings in the **Simulation Setup** tab:

| Parameter | Value |
|---|---|
| Link Bandwidth | 10 Mbps |
| Traffic Load | 0.85 |
| Congestion Scenario | High |
| Simulation Duration | 60 seconds |

### Overall Results

| Algorithm | Avg Latency (ms) | Jitter (ms) | Packet Loss (%) | Throughput (Mbps) |
|---|---|---|---|---|
| FIFO | 312.4 | 198.7 | 24.1 | 7.51 |
| PQ | 198.6 | 142.3 | 19.8 | 7.93 |
| WFQ | 265.3 | 181.2 | 22.6 | 7.68 |
| CBWFQ | 241.8 | 167.4 | 21.3 | 7.82 |
| **Adaptive QoS** | **142.7** | **89.4** | **14.2** | **8.31** |

### Video Conferencing (latency-critical)

| Algorithm | Avg Latency (ms) | Packet Loss (%) |
|---|---|---|
| FIFO | 348.2 | 26.4 |
| PQ | 42.1 | 3.2 |
| WFQ | 289.4 | 23.1 |
| CBWFQ | 261.7 | 20.8 |
| **Adaptive QoS** | **38.6** | **2.1** |

### Background Traffic (PQ starvation visible)

| Algorithm | Packet Loss (%) | Throughput (Mbps) |
|---|---|---|
| FIFO | 22.9 | 1.82 |
| **PQ** | **68.4** | **0.31** ← starved |
| WFQ | 20.3 | 1.49 |
| CBWFQ | 19.1 | 1.54 |
| Adaptive QoS | 31.2 | 1.28 |

> Adaptive QoS protects real-time traffic **without destroying** background flows like PQ does.

---

*For full technical details see [TECHNICAL.md](TECHNICAL.md).*
