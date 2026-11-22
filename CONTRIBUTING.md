# Contributing

Issues and PRs are welcome.

```bash
git clone https://github.com/Aranya2801/AI-Powered-Navigation-for-Autonomous-Vehicles-Real-Time-Hadoop-Cluster-Fault-Detection.git
cd AI-Powered-Navigation-for-Autonomous-Vehicles-Real-Time-Hadoop-Cluster-Fault-Detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

A few things to keep in mind:

- If you change the reward, the observation or the sensor models, old checkpoints
  are no longer valid. Retrain (`scripts/train_all.sh`) and re-run
  `scripts/run_benchmarks.sh` before quoting numbers.
- Don't tune anything on the test seeds (navigation 100000+, cluster 1000+). Use the
  validation ranges (`--seed_base 300000` for navigation, clusters 100-103).
- Numbers in the docs should come from `scripts/results_tables.py`, not be typed in.

When reporting a bug, please include the command, the full traceback, and your
Python / PyTorch versions.
