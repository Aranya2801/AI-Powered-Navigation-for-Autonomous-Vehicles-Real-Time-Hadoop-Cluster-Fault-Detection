#!/usr/bin/env bash
# Every navigation training run used for the results in docs/. Two lanes in
# parallel (each run is single-threaded). ~75 min total on a 2-core machine.
set -e
EP=8000
COMMON="--episodes $EP --curriculum_total_episodes $EP --ramp_frac 0.5 --lr_anneal"

python -m navigation.behavior_cloning --demo_episodes 1000 --bc_epochs 30 --out runs/bc/policy_bc.pt

lane1() {
  python -m navigation.train_ppo $COMMON --seed 0 --out runs/ppo_s0 > runs/ppo_s0.log 2>&1
  python -m navigation.train_ppo $COMMON --seed 2 --out runs/ppo_s2 > runs/ppo_s2.log 2>&1
  python -m navigation.train_ppo $COMMON --seed 0 --out runs/ppo_bc_s0 \
      --warmstart_path runs/bc/policy_bc.pt > runs/ppo_bc_s0.log 2>&1
}
lane2() {
  python -m navigation.train_ppo $COMMON --seed 1 --out runs/ppo_s1 > runs/ppo_s1.log 2>&1
  python -m navigation.train_ppo --episodes $EP --no_curriculum --lr_anneal --seed 0 \
      --curriculum_total_episodes $EP --out runs/ppo_nocurr_s0 > runs/ppo_nocurr_s0.log 2>&1
}
lane1 & lane2 & wait
