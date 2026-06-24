# Changelog

All notable changes to this project are documented here. This project follows an
incremental, module-by-module build approach (see README "Project Philosophy"), so
entries are grouped by module rather than by date/version.

## Module 1 — Fleet Navigation + Sensor Fusion

### Added
- Continuous-dynamics navigation environment (`simulation/env_core.py`) with
  curriculum-scaled difficulty (obstacle count, episode length, start-goal distance)
- Simulated sensor suite: 36-beam LiDAR (vectorized, with range noise + dropout), GPS
  (noise + bias drift + degraded-fix events), IMU (heading + speed, with bias)
- Extended Kalman Filter sensor fusion (`simulation/sensors.py`)
- Classical planning baselines: A* (grid-based), RRT (sampling-based), pure-pursuit
  path tracking, and a periodic-replanning A* variant (`navigation/classical_planners.py`)
- PPO implemented from scratch in PyTorch — actor-critic network, GAE advantage
  estimation, clipped surrogate objective, running reward normalization
  (`navigation/policy_ppo.py`)
- Behavior-cloning warm start from a heuristic demonstration policy
  (`navigation/behavior_cloning.py`)
- Curriculum-scheduled, resumable PPO training loop (`navigation/train_ppo.py`)
- Held-out benchmark harness comparing PPO against all classical baselines
  (`navigation/benchmark.py`)
- 13 unit/regression tests covering environment dynamics, sensor fusion correctness,
  and planner behavior (`tests/`)

### Fixed (in order found during development — see `docs/module1_results.md` for full diagnosis)
- **A\* corner-cutting collisions**: grid obstacle inflation didn't account for the
  vehicle radius plus the pure-pursuit controller's corner-cutting between waypoints,
  causing collisions even on a successfully-planned path. Fixed by increasing
  inflation margin and tightening the controller's lookahead distance.
- **"Stand still" reward hack (v1)**: initial reward shaping made standing still a
  stable low-risk policy under sparse curriculum (no dynamic obstacles). Fixed by
  rebalancing per-step time cost vs. progress reward.
- **LiDAR performance**: the original per-beam, per-obstacle Python double loop took
  ~1.65ms/call, dominating training step time. Vectorized with numpy broadcasting for
  a 10x speedup, with output verified identical via a known-geometry test.
- **Curriculum distance-scaling bug**: rejection-sampling logic for start/goal
  placement resampled a fresh random angle on each retry, biasing accepted samples
  toward whichever angle happened to fit rather than the intended target distance —
  so mean distance didn't actually increase with difficulty. Fixed by placing the
  start position with margin equal to the target distance, guaranteeing the target
  distance is achievable for any sampled angle.
- **Over-strong obstacle proximity penalty**: an attempted fix for obstacle avoidance
  made the proximity penalty so strong it dominated goal-seeking almost everywhere in
  the obstacle field, causing the policy to commit to whichever heading was locally
  obstacle-free regardless of goal direction. Fixed by narrowing the activation radius
  and capping the penalty magnitude relative to the analytically-computed progress
  reward scale.
- **PPO log-prob / clamped-action mismatch**: action log-probability was computed on
  the raw unclamped Gaussian sample, but the clamped action was what got executed and
  stored — creating a biased PPO importance-sampling ratio whenever clamping
  triggered (~10% of actions). Fixed by computing log-prob on the same (clamped)
  tensor that gets executed and stored.
- **Reward-scale instability in the value function**: episode returns ranged from
  roughly -300 to +100 with high variance, causing large, unstable value-function
  gradients. Fixed with running reward normalization before GAE computation.
- **EKF angle-wraparound bug**: the Kalman filter's heading-measurement innovation
  wasn't wrapped to `(-pi, pi]`, so once the true heading crossed the +-pi boundary
  the filter's estimate diverged by ~2*pi within 2-3 steps — silently corrupting both
  the RL observation and the behavior-cloning demonstration data. Fixed by wrapping
  the innovation term and the IMU heading reading itself.
- **Unlearnable steering feature encoding**: encoding heading as raw `(sin, cos)` and
  goal position as a separate `(dx, dy)` vector required the network to implicitly
  learn an atan2-difference, verified to plateau at MSE≈0.15 (R²≈0.06) across multiple
  network configurations and an independent `sklearn.LinearRegression` baseline. Fixed
  by adding the relative bearing angle as an explicit `(sin, cos)` feature.
- **"Stand still" reward hack (v2)**: resurfaced specifically at full curriculum
  difficulty (4 dynamic obstacles) — a flat idle penalty was no longer sufficient,
  since parking in a momentarily-clear spot looked locally safer than navigating a
  denser field. Fixed with an escalating idle penalty that grows with consecutive
  low-speed steps, dropping the "stuck after 30 steps" rate from 13/20 to 1/20 in
  direct evaluation.
- **PPO cold-start exploration failure**: pure PPO exploration on the full-difficulty
  task essentially never sampled a successful episode (0/40 in direct testing), even
  though a trivial heading-chase heuristic solves 95% of easy-tier episodes — so there
  was no successful trajectory for policy-gradient updates to reinforce. Fixed with a
  behavior-cloning warm start from heuristic demonstrations before PPO fine-tuning.

## Planned

- **Module 2**: distributed fleet telemetry pipeline + cluster fault-detection model
  trained on a public cluster trace dataset (Google or Alibaba trace)
- **Module 3**: shared dashboard, MLflow experiment tracking, final repo polish
