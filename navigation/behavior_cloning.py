"""
Behaviour-cloning warm start for PPO.

Teacher: replanning A* + pure pursuit, which has the true obstacle map.
Student: the PPO actor, which only gets the sensor observation.

This is the usual privileged-teacher / sensor-only-student split. The teacher's
map access only shapes the starting point; at test time the policy runs on
sensor data alone.

To reduce covariate shift (the student drifting into states the teacher never
visits) we use DART-style noise injection: in a fraction of episodes the
*executed* action is the teacher action plus Gaussian noise, while the label
stays the clean teacher action. That way the data includes recoveries.

    python -m navigation.behavior_cloning --demo_episodes 300 --bc_epochs 30 \
        --out runs/bc/policy_bc.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from simulation.env_core import NavEnv
from navigation.classical_planners import PlannerAgent
from navigation.observation import ObservationBuilder
from navigation.policy_ppo import PPO, PPOConfig

DEMO_SEED_BASE = 200_000   # disjoint from training (0..) and eval (100_000..)


def collect_demos(n_episodes, noise_frac=0.5, noise_sigma=0.3, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    env = NavEnv()
    teacher = PlannerAgent("astar", replan_every=10)
    X, Y, outcomes = [], [], []
    for ep in range(n_episodes):
        env.set_difficulty(rng.uniform(0, 1))
        s = DEMO_SEED_BASE + seed * 10_000 + ep
        meas = env.reset(s)
        teacher.reset(env, meas, s)
        ob = ObservationBuilder(env)
        obs = ob(meas)
        noisy = rng.random() < noise_frac
        done = False
        while not done:
            a = np.clip(teacher.act(meas), -1, 1)
            X.append(obs)
            Y.append(a)
            ex = np.clip(a + rng.normal(0, noise_sigma, 2), -1, 1) if noisy else a
            meas, _, done, info = env.step(ex)
            obs = ob(meas, ex)
        outcomes.append(info.event)
        if verbose and (ep + 1) % 50 == 0:
            sr = np.mean([o == "goal" for o in outcomes[-50:]])
            print(f"  demos {ep + 1}/{n_episodes}  samples={len(X)}  teacher success(last 50)={sr:.2f}")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32), outcomes


def train_bc(ppo: PPO, X, Y, epochs=30, batch=512, lr=1e-3, val_frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    ppo.obs_rms.update(X)
    Xn = torch.as_tensor(ppo.norm_obs(X))
    Yt = torch.as_tensor(Y)
    n = len(Xn)
    # hold out the *last* episodes rather than random samples - neighbouring
    # steps of one episode are nearly identical, a random split leaks
    nv = int(n * val_frac)
    ti, vi = torch.arange(n - nv), torch.arange(n - nv, n)
    opt = torch.optim.Adam(ppo.net.actor.parameters(), lr=lr, weight_decay=1e-5)
    hist = []
    best, best_state = float("inf"), None
    for ep in range(epochs):
        p = ti[torch.randperm(len(ti), generator=g)]
        tl = []
        for i in range(0, len(p), batch):
            idx = p[i:i + batch]
            loss = ((ppo.net.actor(Xn[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        with torch.no_grad():
            vp = ppo.net.actor(Xn[vi])
            vl = ((vp - Yt[vi]) ** 2).mean().item()
            ss = ((Yt[vi] - Yt[vi].mean(0)) ** 2).mean().item()
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in ppo.net.actor.state_dict().items()}
        hist.append({"epoch": ep + 1, "train_mse": float(np.mean(tl)), "val_mse": vl,
                     "val_r2": 1 - vl / ss})
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  bc epoch {ep + 1:3d}  train_mse={np.mean(tl):.4f}  val_mse={vl:.4f}  R2={1 - vl / ss:.3f}")
    # keep the epoch with the best held-out MSE (it starts overfitting after ~15)
    ppo.net.actor.load_state_dict(best_state)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_episodes", type=int, default=300)
    ap.add_argument("--bc_epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bc/policy_bc.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    print("collecting teacher demonstrations...")
    X, Y, outcomes = collect_demos(args.demo_episodes, seed=args.seed)
    env = NavEnv()
    ppo = PPO(ObservationBuilder(env).dim, PPOConfig())
    print(f"training BC on {len(X)} samples...")
    hist = train_bc(ppo, X, Y, epochs=args.bc_epochs, seed=args.seed)
    torch.save(ppo.state_dict(), args.out)
    meta = {"samples": int(len(X)), "demo_episodes": args.demo_episodes,
            "teacher_success": float(np.mean([o == "goal" for o in outcomes])),
            "history": hist, "wall_time_s": time.time() - t0}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"saved {args.out}  ({meta['wall_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
