"""
Curriculum PPO training.

Difficulty goes linearly 0 -> 1 over the first `ramp_frac` of
`curriculum_total_episodes` and then stays at 1. Training is resumable: the
checkpoint stores the episode counter, normaliser stats and optimiser state.

    # fine-tune from the BC warm start
    python -m navigation.train_ppo --episodes 3000 --out runs/ppo_bc \
        --warmstart_path runs/bc/policy_bc.pt

    # from scratch (ablation)
    python -m navigation.train_ppo --episodes 3000 --out runs/ppo_scratch

    # continue an interrupted run
    python -m navigation.train_ppo --episodes 500 --out runs/ppo_bc --resume
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from simulation.env_core import NavEnv
from navigation.observation import ObservationBuilder
from navigation.policy_ppo import PPO, PPOConfig


def difficulty_at(ep, total, ramp_frac, start=0.0):
    if total <= 0:
        return 1.0
    return float(min(1.0, start + (1 - start) * ep / max(1.0, ramp_frac * total)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000, help="episodes to run in this call")
    ap.add_argument("--out", default="runs/ppo_bc")
    ap.add_argument("--curriculum_total_episodes", type=int, default=3000)
    ap.add_argument("--ramp_frac", type=float, default=0.6)
    ap.add_argument("--curriculum_start", type=float, default=0.0)
    ap.add_argument("--no_curriculum", action="store_true")
    ap.add_argument("--warmstart_path", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rollout_steps", type=int, default=4096)
    ap.add_argument("--critic_warmup_iters", type=int, default=3,
                    help="first N updates train only the critic (useful after BC)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_anneal", action="store_true",
                    help="decay lr linearly to 10%% over curriculum_total_episodes")
    ap.add_argument("--init_log_std", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "policy_checkpoint.pt")
    hist_path = os.path.join(args.out, "training_history.json")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)   # tiny nets, threads only add overhead

    env = NavEnv()
    ob = ObservationBuilder(env)
    cfg = PPOConfig(lr=args.lr)
    start_ep, history, update_i = 0, [], 0

    if args.resume and os.path.exists(ckpt_path):
        d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ppo = PPO(d["obs_dim"], PPOConfig(**d["cfg"]))
        ppo.load_state_dict(d)
        start_ep, update_i = d["episode"], d.get("update", 0)
        torch.set_rng_state(d["torch_rng"])
        with open(hist_path) as f:
            history = json.load(f)
        print(f"resumed at episode {start_ep}")
    else:
        if args.init_log_std is not None:
            cfg.init_log_std = args.init_log_std
        elif args.warmstart_path:
            cfg.init_log_std = -1.2   # BC policy is already decent, explore less
        ppo = PPO(ob.dim, cfg)
        if args.warmstart_path:
            d = torch.load(args.warmstart_path, map_location="cpu", weights_only=False)
            ppo.net.actor.load_state_dict({k[len("actor."):]: v for k, v in d["net"].items()
                                           if k.startswith("actor.")})
            ppo.obs_rms.load_state_dict(d["obs_rms"])
            print(f"warm-started actor from {args.warmstart_path}")

    end_ep = start_ep + args.episodes
    ep = start_ep
    t0 = time.time()

    def new_episode(ep):
        d = 1.0 if args.no_curriculum else difficulty_at(
            ep, args.curriculum_total_episodes, args.ramp_frac, args.curriculum_start)
        env.set_difficulty(d)
        ob.reset()
        return ob(env.reset(seed=args.seed * 1_000_000 + ep)), d

    obs, diff = new_episode(ep)
    ep_ret, ep_len = 0.0, 0
    recent = []

    while ep < end_ep:
        B = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(args.rollout_steps):
            ppo.obs_rms.update(obs)
            on = ppo.norm_obs(obs)
            raw, logp, v = ppo.act(obs)
            a = np.clip(raw, -1, 1)
            meas, r, done, info = env.step(a)
            nobs = ob(meas, a)
            ep_ret += r
            ep_len += 1
            rs = ppo.scale_reward(r, done)
            if info.event == "timeout":
                # truncation, not termination: bootstrap from the next state
                rs += ppo.cfg.gamma * ppo.value(nobs)
            B["obs"].append(on); B["act"].append(raw); B["logp"].append(logp)
            B["val"].append(v); B["rew"].append(rs); B["done"].append(float(done))
            obs = nobs
            if done:
                recent.append({"episode": ep, "difficulty": diff, "return": ep_ret,
                               "length": ep_len, "event": info.event,
                               "path_length": info.path_length})
                ep += 1
                ep_ret, ep_len = 0.0, 0
                obs, diff = new_episode(ep)
                if ep >= end_ep:
                    break

        last_v = 0.0 if B["done"][-1] else ppo.value(obs)
        val = np.array(B["val"], dtype=np.float32)
        adv, ret = ppo.compute_gae(np.array(B["rew"], dtype=np.float32), val,
                                   np.array(B["done"], dtype=np.float32), last_v)
        batch = {"obs": np.array(B["obs"]), "act": np.array(B["act"], dtype=np.float32),
                 "logp": np.array(B["logp"], dtype=np.float32), "val": val,
                 "adv": adv, "ret": ret.astype(np.float32)}
        if args.lr_anneal:
            frac = min(1.0, ep / max(1, args.curriculum_total_episodes))
            for g in ppo.opt.param_groups:
                g["lr"] = args.lr * (1 - 0.9 * frac)
        frozen = update_i < args.critic_warmup_iters and args.warmstart_path is not None
        stats = ppo.update(batch, actor_frozen=frozen)
        update_i += 1

        if recent:
            ev = [r["event"] for r in recent]
            row = {"update": update_i, "episode": ep, "steps": len(B["obs"]),
                   "difficulty": float(np.mean([r["difficulty"] for r in recent])),
                   "mean_return": float(np.mean([r["return"] for r in recent])),
                   "success": float(np.mean([e == "goal" for e in ev])),
                   "collision": float(np.mean([e == "collision" for e in ev])),
                   "timeout": float(np.mean([e == "timeout" for e in ev])),
                   "n_episodes": len(recent), "log_std": ppo.net.log_std.detach().tolist(),
                   "actor_frozen": frozen, "wall_s": time.time() - t0, **stats}
            history.append(row)
            print(f"upd {update_i:4d} ep {ep:5d} diff {row['difficulty']:.2f} "
                  f"ret {row['mean_return']:7.2f} succ {row['success']:.2f} "
                  f"coll {row['collision']:.2f} to {row['timeout']:.2f} "
                  f"kl {stats['kl']:.4f} std {np.exp(row['log_std']).round(2).tolist()} "
                  f"{'(critic only)' if frozen else ''}", flush=True)
            recent = []

        torch.save({**ppo.state_dict(), "episode": ep, "update": update_i,
                    "torch_rng": torch.get_rng_state()}, ckpt_path)
        with open(hist_path, "w") as f:
            json.dump(history, f)

    print(f"done: {ep} episodes, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
