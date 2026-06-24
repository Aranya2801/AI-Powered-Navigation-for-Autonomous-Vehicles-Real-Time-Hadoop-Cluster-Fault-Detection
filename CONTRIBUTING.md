# Contributing

This is currently a solo research/portfolio project, but contributions, issues, and
suggestions are welcome.

## Setup

```bash
git clone https://github.com/Aranya2801/<repo-name>.git
cd <repo-name>
python -m venv venv && source venv/bin/activate  # or your preferred env manager
pip install -r requirements.txt
```

## Running tests

```bash
python -m pytest tests/ -v
```

Tests are currently smoke/correctness tests for the core simulation and planning
modules (environment dynamics, sensor fusion, A*/RRT correctness) rather than a
full coverage suite — see [`docs/module1_results.md`](docs/module1_results.md) for
what's been verified and how.

## Code style

- Type hints on public function signatures where practical.
- Prefer clear, commented reward/algorithm logic over cleverness — this is a research
  benchmark meant to be read, not just run.
- If you change the reward function or observation encoding, retrain (BC warm-start +
  PPO fine-tune) before submitting — old checkpoints are not compatible across reward/
  observation changes, and this project's own development history is full of examples
  of exactly that kind of regression going unnoticed without retraining.

## Reporting issues

Please include:
- The command you ran
- Full traceback or unexpected output
- Whether you modified `simulation/env_core.py`, `navigation/observation.py`, or
  `navigation/policy_ppo.py` (these three are the most failure-sensitive to changes)
