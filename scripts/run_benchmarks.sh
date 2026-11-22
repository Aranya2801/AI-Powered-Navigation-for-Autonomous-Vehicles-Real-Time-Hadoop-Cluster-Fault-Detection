#!/usr/bin/env bash
# Held-out evaluation for everything in docs/results.md. Needs the checkpoints
# from scripts/train_all.sh (or the ones committed in runs/).
set -e
CK=(--ppo_checkpoint "PPO s0=runs/ppo_s0/policy_checkpoint.pt"
    --ppo_checkpoint "PPO s1=runs/ppo_s1/policy_checkpoint.pt"
    --ppo_checkpoint "PPO s2=runs/ppo_s2/policy_checkpoint.pt")
ABL=(--ppo_checkpoint "PPO + BC init=runs/ppo_bc_s0/policy_checkpoint.pt"
     --ppo_checkpoint "PPO, no curriculum=runs/ppo_nocurr_s0/policy_checkpoint.pt"
     --ppo_checkpoint "BC only=runs/bc/policy_bc.pt")

for d in 1.0 0.5; do
  python -m navigation.benchmark "${CK[@]}" "${ABL[@]}" --n_episodes 200 \
      --difficulty $d --out runs/benchmark_d$d.json
done

mkdir -p runs/noise
for k in 0.5 1.0 2.0 3.0 4.0; do
  python -m navigation.benchmark "${CK[@]}" --n_episodes 200 --difficulty 1.0 \
      --noise_scale $k --out runs/noise/benchmark_noise$k.json
done
