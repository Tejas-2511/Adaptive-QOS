Adaptive QoS - simulator and experiments (re-implementation of the technical reference)

Requirements: Python 3.10+, numpy, pandas, scipy, matplotlib
  pip install numpy pandas scipy matplotlib

Run order (put all three .py files in one folder):
  1. mkdir figs
  2. python run_experiments.py   -> writes results.json (about 3 minutes)
  3. python analyze.py           -> writes tables (.pkl) and result figures into figs/, prints summaries

Files:
  qos_sim.py          packet generator, FIFO/PQ/WFQ/CBWFQ/Adaptive QoS schedulers, metrics
  run_experiments.py  experiments E1-E6 (reference run, congestion levels, 30-seed stats, ablation, workloads, sweeps)
  analyze.py          means, paired t-test, Wilcoxon, CIs, effect sizes, ablation tables, figures
Seeds are fixed (42 for the reference run, 1-30 for statistics), so results are reproducible.
