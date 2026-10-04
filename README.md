# QoS Network Traffic Simulation

> **Computer Networks Mini Project**  
> *Intelligent QoS-Based Network Traffic Prioritization for Low-Latency Applications*

---

## What This Project Does

This project simulates network traffic and evaluates how five different **Quality of Service (QoS)** algorithms handle congestion, especially for latency-sensitive applications like gaming and video conferencing.

**Research Question:**  
> Can Adaptive QoS reduce latency, jitter and packet loss for latency-sensitive applications during network congestion compared with conventional queueing methods?

---

## Project Structure

```
qos_project/
├── app.py              # Streamlit dashboard (main entry point)
├── simulation.py       # Packet generation & simulation loop
├── qos_algorithms.py   # FIFO, PQ, WFQ, CBWFQ, Adaptive QoS
├── metrics.py          # Latency, jitter, packet loss, throughput
├── database.py         # SQLite read/write helpers
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── qos_results.db      # Auto-created SQLite database
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the dashboard

```bash
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`.

---

## Traffic Types Simulated

| Traffic Type      | Priority | Typical Use          |
|-------------------|----------|----------------------|
| Video Conferencing| Highest  | Zoom, Teams, Meet    |
| Gaming            | High     | Online multiplayer   |
| Video Streaming   | Medium   | Netflix, YouTube     |
| Background        | Low      | Downloads, backups   |

---

## QoS Algorithms

### FIFO (First In, First Out)
Processes packets in arrival order. Simple but unfair during congestion.

### PQ (Priority Queuing)
Strictly serves higher-priority queues first. Best for real-time traffic, but background traffic may starve.

### WFQ (Weighted Fair Queuing)
Weights: VC=4, Gaming=3, Streaming=2, Background=1.  
Proportional service based on weight.

### CBWFQ (Class-Based WFQ)
Guaranteed bandwidth shares: VC=35%, Gaming=30%, Streaming=20%, Background=15%.

### Adaptive QoS (Proposed Method)
Dynamically adjusts weights based on congestion score:

```
Congestion Score = 0.5 × BW_Util + 0.3 × Packet_Loss + 0.2 × Latency
```

| Score Range | Level    | Action                                      |
|-------------|----------|---------------------------------------------|
| 0.0 – 0.3   | Low      | Normal weights                              |
| 0.3 – 0.6   | Moderate | Slightly boost real-time traffic            |
| 0.6 – 0.8   | High     | Strongly prioritize real-time traffic       |
| 0.8 – 1.0   | Severe   | Maximum priority for real-time, min for BG  |

---

## Metrics Calculated

| Metric              | Formula                                              |
|---------------------|------------------------------------------------------|
| Average Latency     | Mean(Departure_Time − Arrival_Time) × 1000 ms        |
| Jitter              | Mean(|Latency_i − Latency_{i-1}|)                    |
| Packet Loss         | Dropped_Packets / Total_Packets × 100%               |
| Throughput          | Delivered_Bytes × 8 / Simulation_Time (Mbps)         |
| BW Utilization      | Throughput / Link_Bandwidth × 100%                   |

---

## Database Schema (SQLite)

**experiments** – one row per simulation run  
**results** – one row per traffic type per experiment (+ "Overall")

---

## Key Points for Viva

1. **Why simulate?** Real packet capture requires root access and hardware; simulation lets us test controlled, reproducible scenarios.
2. **Why Adaptive QoS?** Fixed-weight algorithms can't react to changing network conditions. Adaptive QoS adjusts priorities in real time.
3. **Is Adaptive QoS always better?** Not necessarily — the simulation produces actual results. Under low congestion, all algorithms behave similarly.
4. **Reproducibility** — using a fixed random seed ensures the same packet stream is used for all algorithms, making comparisons fair.
5. **No ML** — the adaptive logic is purely rule-based using simple thresholds on the congestion score.

---

## Congestion Scenarios

| Scenario | Network Utilization |
|----------|---------------------|
| Low      | ~30%                |
| Moderate | ~60%                |
| High     | ~90%                |

---

*Built with Python · Streamlit · NumPy · Pandas · Matplotlib · SQLite*
