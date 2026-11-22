import numpy as np
import pytest
import torch

from simulation.env_core import NavEnv
from navigation.observation import ObservationBuilder
from navigation.policy_ppo import PPO, PPOConfig, RunningMeanStd
from navigation.benchmark import wilson, mcnemar


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, (1000, 5))
    rms = RunningMeanStd((5,))
    for chunk in np.array_split(x, 7):
        rms.update(chunk)
    assert np.allclose(rms.mean, x.mean(0), atol=1e-3)
    assert np.allclose(rms.var, x.var(0), atol=1e-2)


def test_gae_matches_hand_computation():
    ppo = PPO(4, PPOConfig(gamma=0.9, lam=0.5))
    rew = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    val = np.array([0.5, 0.2, 0.1], dtype=np.float32)
    done = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = ppo.compute_gae(rew, val, done, last_val=99.0)  # last_val ignored: done
    d2 = 2.0 - 0.1
    d1 = 0.0 + 0.9 * 0.1 - 0.2
    d0 = 1.0 + 0.9 * 0.2 - 0.5
    a2 = d2
    a1 = d1 + 0.45 * a2
    a0 = d0 + 0.45 * a1
    assert adv == pytest.approx([a0, a1, a2], rel=1e-5)
    assert ret == pytest.approx(adv + val, rel=1e-5)


def test_logprob_matches_stored_action():
    """The stored action must be the tensor the log-prob was computed on,
    otherwise the first PPO ratio is not 1 for clipped samples."""
    torch.manual_seed(0)
    ppo = PPO(6, PPOConfig(init_log_std=0.5))   # wide policy -> lots of clipping
    obs = np.random.default_rng(0).normal(size=(256, 6)).astype(np.float32)
    acts, logps = [], []
    for o in obs:
        a, lp, _ = ppo.act(o)
        acts.append(a)
        logps.append(lp)
    acts = np.array(acts)
    assert (np.abs(acts) > 1).any(), "test needs some out-of-range samples"
    with torch.no_grad():
        d = ppo.net.dist(torch.as_tensor(ppo.norm_obs(obs)))
        lp = d.log_prob(torch.as_tensor(acts)).sum(-1).numpy()
    assert np.allclose(lp, logps, atol=1e-5)


def test_observation_does_not_use_true_state():
    env = NavEnv(1.0)
    m = env.reset(3)
    ob = ObservationBuilder(env)
    o1 = ob(m)
    env.state = env.state + np.array([5.0, -5.0, 1.0, 1.0])
    ob.reset()
    o2 = ob(m)
    assert o1.shape == (ob.dim,) and np.allclose(o1, o2)


def test_update_runs_and_changes_policy():
    torch.manual_seed(0)
    ppo = PPO(8, PPOConfig(epochs=2, minibatch=64))
    rng = np.random.default_rng(0)
    n = 256
    batch = {"obs": rng.normal(size=(n, 8)).astype(np.float32),
             "act": rng.normal(size=(n, 2)).astype(np.float32),
             "logp": np.full(n, -2.0, dtype=np.float32),
             "val": np.zeros(n, dtype=np.float32),
             "adv": rng.normal(size=n).astype(np.float32),
             "ret": rng.normal(size=n).astype(np.float32)}
    before = [p.clone() for p in ppo.net.actor.parameters()]
    stats = ppo.update(batch)
    assert np.isfinite(stats["pi_loss"]) and np.isfinite(stats["v_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, ppo.net.actor.parameters()))


def test_critic_only_update_keeps_actor_fixed():
    torch.manual_seed(0)
    ppo = PPO(8, PPOConfig(epochs=1, minibatch=64))
    rng = np.random.default_rng(1)
    n = 128
    batch = {"obs": rng.normal(size=(n, 8)).astype(np.float32),
             "act": rng.normal(size=(n, 2)).astype(np.float32),
             "logp": np.full(n, -2.0, dtype=np.float32),
             "val": np.zeros(n, dtype=np.float32),
             "adv": rng.normal(size=n).astype(np.float32),
             "ret": rng.normal(size=n).astype(np.float32)}
    before = [p.clone() for p in ppo.net.actor.parameters()]
    ppo.update(batch, actor_frozen=True)
    assert all(torch.equal(a, b) for a, b in zip(before, ppo.net.actor.parameters()))


def test_stats_helpers():
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and hi - lo < 0.2
    a = [{"event": "goal"}] * 10 + [{"event": "collision"}] * 10
    b = [{"event": "goal"}] * 20
    r = mcnemar(a, b)
    assert r["b_only"] == 10 and r["p"] < 0.01
