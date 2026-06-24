# Module 1 Results: Multi-Agent Fleet Navigation with Sensor Fusion

## Summary

A PPO policy was trained from scratch (PyTorch, no RL library dependency) to navigate
a simulated vehicle through a field of static and dynamic obstacles toward a goal,
using only noisy fused sensor estimates (simulated LiDAR + GPS + IMU, fused via an
Extended Kalman Filter) — never ground-truth state. The learned policy is benchmarked
against three classical planning baselines (A* without replanning, A* with periodic
replanning, RRT) on held-out evaluation seeds never seen during training.

**Headline result (full difficulty, 50 held-out episodes):**

| Method                | Success | Collision | Timeout | Path Efficiency |
|------------------------|---------|-----------|---------|------------------|
| **PPO (learned, noisy obs)** | **80%** | 20%       | 0%      | 1.07             |
| A* (no replanning)     | 84%     | 8%        | 8%      | 1.00             |
| A* (with replanning)   | 92%     | 8%        | 0%      | 1.00             |
| RRT                    | 82%     | 14%       | 4%      | 1.11             |

PPO is competitive with classical planners despite a strictly harder information
environment (the classical planners are given the true obstacle field directly, as
is standard for planning baselines; PPO only ever sees noisy sensor-fusion output).
PPO never times out — once trained, it always commits to either reaching the goal or
colliding, unlike the search-based planners which occasionally fail to find a path
within their iteration/replanning budget.

**At medium difficulty (50 held-out episodes):**

| Method                | Success | Collision | Timeout | Path Efficiency |
|------------------------|---------|-----------|---------|------------------|
| PPO                    | 92%     | 8%        | 0%      | 1.07             |
| A* (no replanning)     | 98%     | 0%        | 2%      | 1.00             |
| A* (with replanning)   | 100%    | 0%        | 0%      | 1.00             |
| RRT                    | 86%     | 14%       | 0%      | 1.11             |

## Methodology

### Environment
- Continuous 2D world, bicycle-model vehicle dynamics (accel, steer-rate control)
- Curriculum-scaled difficulty: 3→14 static obstacles, 0→4 dynamic (moving) obstacles,
  150→400 max steps, 35→57 unit average start-goal distance
- Sensor suite: 36-beam LiDAR (range noise + 3% dropout), GPS (noise + slow bias drift
  + 2% "degraded fix" events), IMU (heading + speed, with bias)
- Fusion: Extended Kalman Filter combining GPS position + IMU heading/speed into a
  single state estimate, which is what the RL policy actually observes

### Training
1. **Behavior-cloning warm start**: a simple heading-chase heuristic (no obstacle
   awareness) generates demonstration trajectories; the PPO policy network is
   pretrained via supervised regression on these before any RL begins.
2. **Curriculum-scheduled PPO fine-tuning**: ~2000 episodes, difficulty linearly
   ramped from 0→1 over the first 60% of training, then held at full difficulty.
3. Policy: shared-trunk Gaussian actor-critic, GAE advantage estimation, clipped
   surrogate PPO objective, running reward normalization.

### Benchmark protocol
All four methods are evaluated on the same 50 episodes (seeds 100000-100049, disjoint
from the 0-2000 range used during training) at a fixed difficulty level, so comparisons
are apples-to-apples on identical start/goal/obstacle configurations.

## Notable findings from development

- **A* without replanning is structurally blind to dynamic obstacles.** Diagnosed
  directly: 100% of A*-no-replan collisions in an early test were attributable to
  dynamic obstacles (mean clearance ~1.2 units at collision) vs. static obstacles
  (mean clearance >9 units) — A* plans once against a t=0 snapshot and never updates.
  Adding periodic replanning (every 10 steps) raised success from 70%→85% in that test.

- **A subtle log_prob/action mismatch bug stalled PPO learning entirely.** The original
  implementation computed the action log-probability on the raw unclamped Gaussian
  sample, but stored/executed the `[-1,1]`-clamped action — creating a systematic bias
  in the PPO importance-sampling ratio whenever clamping triggered (~10% of actions).

- **An EKF angle-wraparound bug corrupted heading estimates.** The Kalman filter's
  innovation term for heading measurements wasn't wrapped to `(-π, π]`, so once the
  true heading crossed the ±π boundary, the filter's heading estimate diverged by
  ~2π radians within 2-3 steps — silently corrupting both the RL observation and the
  behavior-cloning demonstration data. Verified via direct true-vs-estimated heading
  comparison; fixed by wrapping the innovation before the Kalman update.

- **Raw trigonometric features made the steering target unlearnable.** Encoding
  heading as `(sin, cos)` and goal position as a separate `(dx, dy)` vector, while
  mathematically sufficient, required the network to implicitly learn an
  atan2-difference — verified to plateau at MSE≈0.15 (R²≈0.06) across multiple
  network configurations *and* an independent `sklearn.LinearRegression` baseline.
  Adding the relative bearing angle as an explicit `(sin, cos)` feature dropped
  achievable MSE to ≈0.005-0.008.

- **A "stand still" reward hack appeared twice, in two different forms.** First, with
  a flat per-step idle penalty too small relative to dense progress shaping (fixed by
  rebalancing coefficients). Second — more subtly — at full curriculum difficulty
  (4 dynamic obstacles), a flat idle penalty was no longer sufficient: standing in a
  momentarily-clear spot looked locally safer than navigating a denser field, even
  though parking is eventually unsafe once a dynamic obstacle wanders into the parked
  position. Fixed with an escalating idle penalty (grows with consecutive low-speed
  steps), which dropped the "stuck after 30 steps" rate from 13/20 to 1/20 in direct
  evaluation.

## Reproducing

```bash
# 1. Behavior-cloning warm start
python -m navigation.behavior_cloning --demo_episodes 60 --bc_epochs 20 \
    --out runs/ppo_v1/policy_bc_warmstart.pt

# 2. Curriculum PPO fine-tuning (run in chunks; supports --resume)
python -m navigation.train_ppo --episodes 2000 --out runs/ppo_v2 \
    --curriculum_total_episodes 2500 --warmstart_path runs/ppo_v1/policy_bc_warmstart.pt

# 3. Benchmark against classical baselines on held-out seeds
python -m navigation.benchmark --ppo_checkpoint runs/ppo_v2/policy_checkpoint.pt \
    --n_episodes 50 --difficulty 1.0 --out runs/benchmark_results.json
```

## Known limitations / honest caveats

- PPO's path efficiency (1.07) is slightly worse than the classical planners' (1.00),
  meaning it takes marginally longer routes — expected, since it's reacting to noisy
  partial observations in real time rather than planning against a known map.
- The curriculum and reward function required several rounds of empirical debugging
  (documented above) before producing stable training; the reward coefficients are
  tuned for this specific environment scale and would need re-tuning for materially
  different world sizes or obstacle densities.
- Training run is ~2000 episodes / a few minutes wall-clock on CPU — sufficient to
  demonstrate the methodology and get real comparative numbers, but a production
  system would benefit from longer training, parallelized rollouts, and likely a
  larger network.
