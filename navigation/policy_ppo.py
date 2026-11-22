"""
PPO written directly in PyTorch (no SB3 / RLlib).

  * separate actor and critic MLPs (tanh), Gaussian policy with a learned,
    state-independent log std
  * GAE(lambda), clipped surrogate objective, clipped value loss
  * running observation normalisation and reward scaling by the running std of
    the discounted return
  * time-limit truncation is bootstrapped (a timeout is not a real terminal)

Action handling: the *raw* Gaussian sample is stored and used for log-probs;
it is only clipped to [-1, 1] when sent to the env. Computing the log-prob on
one tensor and storing another makes the PPO ratio wrong whenever clipping kicks
in, see tests/test_ppo.py::test_logprob_matches_stored_action.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    def __init__(self, shape=(), eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None]
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), float(d["count"])


def mlp(i, o, h=128, out_gain=1.0):
    layers = [nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o)]
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=out_gain)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden=128, init_log_std=-0.5):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim, hidden, out_gain=0.01)
        self.critic = mlp(obs_dim, 1, hidden, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def dist(self, obs):
        mu = self.actor(obs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vclip: float = 0.2
    epochs: int = 8
    minibatch: int = 512
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden: int = 128
    init_log_std: float = -0.5
    min_log_std: float = -2.5


class PPO:
    def __init__(self, obs_dim, cfg: PPOConfig | None = None, act_dim=2):
        self.cfg = cfg or PPOConfig()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.net = ActorCritic(obs_dim, act_dim, self.cfg.hidden, self.cfg.init_log_std)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self._ret = 0.0

    # ---------------------------------------------------------------- helpers
    def norm_obs(self, obs, update=False):
        if update:
            self.obs_rms.update(obs)
        o = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(o, -10, 10).astype(np.float32)

    def scale_reward(self, r, done):
        self._ret = self._ret * self.cfg.gamma + r
        self.ret_rms.update(np.array([self._ret]))
        if done:
            self._ret = 0.0
        return r / np.sqrt(self.ret_rms.var + 1e-8)

    @torch.no_grad()
    def act(self, obs, deterministic=False, update_norm=False):
        o = torch.as_tensor(self.norm_obs(obs, update_norm))
        d = self.net.dist(o)
        raw = d.mean if deterministic else d.sample()
        logp = d.log_prob(raw).sum(-1)
        v = self.net.value(o)
        return raw.numpy(), float(logp), float(v)

    @torch.no_grad()
    def value(self, obs):
        return float(self.net.value(torch.as_tensor(self.norm_obs(obs))))

    # ---------------------------------------------------------------- update
    def compute_gae(self, rew, val, done, last_val):
        T = len(rew)
        adv = np.zeros(T, dtype=np.float32)
        g, lam = self.cfg.gamma, self.cfg.lam
        last = 0.0
        for t in reversed(range(T)):
            nv = last_val if t == T - 1 else val[t + 1]
            nonterm = 1.0 - done[t]
            delta = rew[t] + g * nv * nonterm - val[t]
            last = delta + g * lam * nonterm * last
            adv[t] = last
        return adv, adv + val

    def update(self, batch, actor_frozen=False):
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"])
        act = torch.as_tensor(batch["act"])
        old_logp = torch.as_tensor(batch["logp"])
        old_v = torch.as_tensor(batch["val"])
        adv = torch.as_tensor(batch["adv"])
        ret = torch.as_tensor(batch["ret"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        stats = {"pi_loss": [], "v_loss": [], "kl": [], "clipfrac": [], "entropy": []}
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                d = self.net.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - old_logp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]
                pi_loss = -torch.min(s1, s2).mean()
                ent = d.entropy().sum(-1).mean()

                v = self.net.value(obs[idx])
                v_clip = old_v[idx] + (v - old_v[idx]).clamp(-cfg.vclip, cfg.vclip)
                v_loss = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()

                if actor_frozen:
                    loss = cfg.vf_coef * v_loss
                else:
                    loss = pi_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                self.opt.zero_grad()
                loss.backward()
                if actor_frozen:
                    for p in list(self.net.actor.parameters()) + [self.net.log_std]:
                        p.grad = None
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(min=cfg.min_log_std)
                    kl = ((ratio - 1) - (logp - old_logp[idx])).mean().item()
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["kl"].append(kl)
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats["entropy"].append(ent.item())
            if not actor_frozen and np.mean(stats["kl"][-max(1, n // cfg.minibatch):]) > cfg.target_kl:
                break
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ---------------------------------------------------------------- io
    def state_dict(self):
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict(),
                "obs_rms": self.obs_rms.state_dict(), "ret_rms": self.ret_rms.state_dict(),
                "cfg": asdict(self.cfg), "obs_dim": self.obs_dim}

    def load_state_dict(self, d, load_opt=True):
        self.net.load_state_dict(d["net"])
        if load_opt and "opt" in d:
            self.opt.load_state_dict(d["opt"])
        self.obs_rms.load_state_dict(d["obs_rms"])
        if "ret_rms" in d:
            self.ret_rms.load_state_dict(d["ret_rms"])

    @classmethod
    def load(cls, path, load_opt=False):
        d = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(d["obs_dim"], PPOConfig(**d["cfg"]))
        agent.load_state_dict(d, load_opt=load_opt)
        return agent


class PPOAgent:
    """Benchmark wrapper so PPO has the same reset/act interface as the planners."""

    def __init__(self, ppo: PPO, name="PPO", deterministic=True):
        self.ppo, self.name, self.det = ppo, name, deterministic

    def reset(self, env, meas, episode_seed=0):
        from navigation.observation import ObservationBuilder
        self.ob = ObservationBuilder(env)
        self._last = None
        self.plan_time = 0.0

    def act(self, meas):
        # same call pattern as training: first obs has no previous action
        obs = self.ob(meas, self._last)
        a, _, _ = self.ppo.act(obs, deterministic=self.det)
        a = np.clip(a, -1, 1)
        self._last = a
        return a
