"""
Held-out evaluation: PPO vs A* / A* with replanning / RRT.

Every method runs on the same seeds (100000 + i), which never appear in
training (0..N) or in the BC demos (200000+). Same seed -> same start, goal,
obstacles *and* the same sensor noise sequence, so the comparison is paired.

    python -m navigation.benchmark --ppo_checkpoint runs/ppo_bc/policy_checkpoint.pt \
        --n_episodes 200 --difficulty 1.0 --out runs/benchmark_d1.0.json

    # sensor-noise stress test
    python -m navigation.benchmark ... --noise_scale 2.0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

from simulation.env_core import NavEnv
from simulation.sensors import SensorNoise
from navigation.classical_planners import PlannerAgent

EVAL_SEED_BASE = 100_000


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def run_episode(env: NavEnv, agent, seed: int):
    meas = env.reset(seed)
    straight = float(np.linalg.norm(env.goal - env.start))
    agent.reset(env, meas, seed)
    done = False
    t_act = 0.0
    min_clear = float("inf")
    while not done:
        t0 = time.perf_counter()
        a = agent.act(meas)
        t_act += time.perf_counter() - t0
        meas, _, done, info = env.step(a)
        min_clear = min(min_clear, info.min_clearance)
    return {"seed": seed, "event": info.event, "collided_with": info.collided_with,
            "steps": env.t, "path_length": info.path_length, "straight": straight,
            "efficiency": info.path_length / straight if info.event == "goal" else None,
            "min_clearance": min_clear, "ms_per_step": 1000 * t_act / max(1, env.t)}


def summarise(rows):
    n = len(rows)
    ev = [r["event"] for r in rows]
    k = sum(e == "goal" for e in ev)
    eff = [r["efficiency"] for r in rows if r["efficiency"] is not None]
    steps = [r["steps"] for r in rows if r["event"] == "goal"]
    coll = [r["collided_with"] for r in rows if r["event"] == "collision"]
    return {
        "n": n,
        "success": k / n, "success_ci95": wilson(k, n),
        "collision": ev.count("collision") / n,
        "timeout": ev.count("timeout") / n,
        "collision_static": coll.count("static") / n,
        "collision_dynamic": coll.count("dynamic") / n,
        "collision_wall": coll.count("wall") / n,
        "path_efficiency": float(np.mean(eff)) if eff else None,
        "steps_to_goal": float(np.mean(steps)) if steps else None,
        "ms_per_step": float(np.mean([r["ms_per_step"] for r in rows])),
    }


def mcnemar(a_rows, b_rows):
    """Exact two-sided McNemar test on paired success/failure."""
    from math import comb
    b = sum(x["event"] == "goal" and y["event"] != "goal" for x, y in zip(a_rows, b_rows))
    c = sum(x["event"] != "goal" and y["event"] == "goal" for x, y in zip(a_rows, b_rows))
    n = b + c
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}


def build_agents(ppo_checkpoints):
    agents = []
    for spec in ppo_checkpoints:
        from navigation.policy_ppo import PPO, PPOAgent
        name, _, path = spec.rpartition("=") if "=" in spec else ("PPO", "", spec)
        agents.append(PPOAgent(PPO.load(path), name=name or "PPO"))
    agents += [PlannerAgent("astar"), PlannerAgent("astar", replan_every=10), PlannerAgent("rrt")]
    return agents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo_checkpoint", action="append", default=[],
                    help="path or NAME=path, can be repeated")
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--only", default=None, help="comma separated agent names to run")
    ap.add_argument("--seed_base", type=int, default=EVAL_SEED_BASE,
                    help="use e.g. 300000 for a validation split while tuning")
    ap.add_argument("--out", default="runs/benchmark_results.json")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)   # batch-of-one inference; extra threads just spin

    env = NavEnv(args.difficulty, noise=SensorNoise().scaled(args.noise_scale))
    agents = build_agents(args.ppo_checkpoint)
    if args.only:
        keep = set(args.only.split(","))
        agents = [a for a in agents if a.name in keep]
    seeds = [args.seed_base + i for i in range(args.n_episodes)]

    results = {"config": vars(args), "methods": {}, "episodes": {}}
    for ag in agents:
        t0 = time.time()
        rows = [run_episode(env, ag, s) for s in seeds]
        summ = summarise(rows)
        results["methods"][ag.name] = summ
        results["episodes"][ag.name] = rows
        lo, hi = summ["success_ci95"]
        eff = summ["path_efficiency"]
        print(f"{ag.name:14s} success {summ['success']:.3f} [{lo:.2f},{hi:.2f}]  "
              f"coll {summ['collision']:.3f} (stat {summ['collision_static']:.2f} "
              f"dyn {summ['collision_dynamic']:.2f})  timeout {summ['timeout']:.3f}  "
              f"eff {eff if eff is None else round(eff, 3)}  "
              f"{summ['ms_per_step']:.2f} ms/step  ({time.time() - t0:.0f}s)", flush=True)

    names = list(results["episodes"])
    if names and args.ppo_checkpoint:
        ref = names[0]
        results["paired_tests"] = {
            f"{ref} vs {o}": mcnemar(results["episodes"][ref], results["episodes"][o])
            for o in names[1:]}
        for k, v in results["paired_tests"].items():
            print(f"  McNemar {k}: {ref}-only wins {v['a_only']}, other-only wins {v['b_only']}, p={v['p']:.3g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
