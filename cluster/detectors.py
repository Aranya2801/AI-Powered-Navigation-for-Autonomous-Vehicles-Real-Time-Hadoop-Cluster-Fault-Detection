"""
Online (one tick at a time) per-node anomaly detectors.

All detectors share the same interface:

    det.reset(n_nodes)
    scores = det.update(x_t)       # x_t: (n_nodes, n_metrics), NaN rows = no report

and return one anomaly score per node per tick. Turning scores into alarms
(threshold + debounce) is done by `AlarmPolicy`, so every detector is judged with
the same alarm logic and a threshold calibrated the same way.

Liveness is handled the same way everywhere: a node that misses
`missing_ticks` consecutive reports gets score +inf. That's what a heartbeat
timeout does in practice, and it means the interesting differences between
detectors are in the non-crash faults.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cluster.simulator import METRICS, M

# metrics where only an increase is bad (one-sided scoring)
UP_BAD = ["cpu_pct", "mem_pct", "gc_pause_ms", "io_wait_pct", "tcp_retrans_pct",
          "heartbeat_ms", "failed_block_ops", "task_failures"]
UP_IDX = np.array([M[m] for m in UP_BAD])


class _Liveness:
    def __init__(self, missing_ticks=2):
        self.k = missing_ticks

    def reset(self, n):
        self.miss = np.zeros(n, dtype=int)

    def __call__(self, x):
        dead = np.isnan(x).all(axis=1)
        self.miss = np.where(dead, self.miss + 1, 0)
        return self.miss >= self.k, dead


class StaticThreshold:
    """What a stock Ambari/Grafana alert set looks like: fixed limits per metric.
    Score = number of metrics over their limit."""

    name = "static-threshold"
    LIMITS = {"cpu_pct": 95, "io_wait_pct": 30, "gc_pause_ms": 1000, "heartbeat_ms": 100,
              "tcp_retrans_pct": 2.0, "failed_block_ops": 3, "task_failures": 4, "mem_pct": 92}

    def __init__(self):
        self.idx = np.array([M[k] for k in self.LIMITS])
        self.lim = np.array(list(self.LIMITS.values()), dtype=float)
        self.live = _Liveness()

    def reset(self, n):
        self.live.reset(n)

    def update(self, x):
        gone, dead = self.live(x)
        s = (np.nan_to_num(x[:, self.idx], nan=0.0) > self.lim).sum(1).astype(float)
        s[dead] = 0.0
        s[gone] = np.inf
        return s


class EWMAZScore:
    """Each node against its own recent history. Cheap and common, but a
    cluster-wide load change (nightly ETL) looks like an anomaly on every node."""

    name = "ewma-zscore"

    def __init__(self, alpha=0.01, warmup=60, smooth=0.3):
        self.alpha, self.warmup, self.smooth = alpha, warmup, smooth
        self.live = _Liveness()

    def reset(self, n):
        F = len(METRICS)
        self.mu = np.zeros((n, F))
        self.var = np.ones((n, F))
        self.t = np.zeros(n, dtype=int)
        self.s = np.zeros(n)
        self.live.reset(n)

    def update(self, x):
        gone, dead = self.live(x)
        ok = ~dead
        xo = x[ok]
        mu, var = self.mu[ok], self.var[ok]
        z = (xo - mu) / np.sqrt(var + 1e-6)
        z = np.clip(z, -20, 20)
        raw = np.zeros(len(x))
        raw[ok] = np.where(self.t[ok] < self.warmup, 0.0, np.max(z[:, UP_IDX], axis=1))
        # don't let anomalies drag the baseline: freeze updates when clearly anomalous
        a = np.where(np.abs(z) > 6, 0.0, self.alpha)
        a = np.where(self.t[ok, None] < self.warmup, 1.0 / (self.t[ok, None] + 1), a)
        d = xo - mu
        self.mu[ok] = mu + a * d
        self.var[ok] = (1 - a) * (var + a * d * d)
        self.t[ok] += 1
        self.s = (1 - self.smooth) * self.s + self.smooth * raw
        out = self.s.copy()
        out[gone] = np.inf
        return out


def peer_z(x, eps_floor=None):
    """Robust z-score of each node against the cluster at the same tick."""
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    if eps_floor is not None:
        mad = np.maximum(mad, eps_floor)
    return (x - med) / (mad + 1e-6)


# small absolute floors so near-constant metrics (failed ops ~ 0) don't blow up
FLOORS = np.ones(len(METRICS))
FLOORS[M["failed_block_ops"]] = 0.5
FLOORS[M["task_failures"]] = 0.5
FLOORS[M["tcp_retrans_pct"]] = 0.05
FLOORS[M["containers"]] = 1.0


class PeerMAD:
    """Each node against its peers right now. Hadoop worker nodes run the same
    software on (mostly) the same hardware and share the workload, so a
    cluster-wide change moves the median along with every node."""

    name = "peer-mad"

    def __init__(self, smooth=0.25):
        self.smooth = smooth
        self.live = _Liveness()

    def reset(self, n):
        self.s = np.zeros(n)
        self.live.reset(n)

    def update(self, x):
        gone, dead = self.live(x)
        z = np.clip(peer_z(x, FLOORS), -30, 30)
        raw = np.nan_to_num(np.max(z[:, UP_IDX], axis=1), nan=0.0)
        # throughput collapse is also a symptom (disk / network faults)
        low = -np.nan_to_num(np.min(z[:, [M["disk_read_mbps"], M["net_in_mbps"]]], axis=1), nan=0.0)
        raw = np.maximum(raw, 0.6 * low)
        raw[dead] = 0.0
        self.s = (1 - self.smooth) * self.s + self.smooth * raw
        out = self.s.copy()
        out[gone] = np.inf
        return out


# ---------------------------------------------------------------- learned
class _AE(nn.Module):
    def __init__(self, d, h=64, z=12):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


def feature_frames(X):
    """(T, N, F) raw metrics -> (T, N, 2F) features: peer z-score and
    log-ratio to the node's own slow EWMA. Computed causally."""
    T, N, F = X.shape
    out = np.zeros((T, N, 2 * F), dtype=np.float32)
    mu = None
    for t in range(T):
        x = X[t]
        pz = np.clip(np.nan_to_num(peer_z(x, FLOORS)), -10, 10)
        xl = np.log1p(np.nan_to_num(x, nan=0.0))
        if mu is None:
            mu = xl.copy()
        own = np.clip(xl - mu, -5, 5)
        alive = ~np.isnan(x).all(1)
        mu[alive] = 0.99 * mu[alive] + 0.01 * xl[alive]
        out[t, :, :F] = pz
        out[t, :, F:] = own
    return out


class WindowAutoencoder:
    """MLP autoencoder over a sliding window of features, trained only on
    normal windows. Score = reconstruction error (EWMA-smoothed)."""

    name = "window-ae"

    def __init__(self, window=10, smooth=0.3, epochs=15, seed=0):
        self.w, self.smooth, self.epochs, self.seed = window, smooth, epochs, seed
        self.live = _Liveness()
        self.model = None

    # --------------------------------------------------------- training
    def _windows(self, Fr, mask=None, stride=1):
        T, N, D = Fr.shape
        idx = np.arange(self.w - 1, T, stride)
        W = np.stack([Fr[i - self.w + 1:i + 1] for i in idx], axis=0)     # (T', w, N, D)
        W = W.transpose(0, 2, 1, 3).reshape(len(idx), N, self.w * D)
        if mask is None:
            return W.reshape(-1, self.w * D)
        m = np.stack([~mask[i - self.w + 1:i + 1].any(0) for i in idx], 0)  # clean windows
        return W[m]

    def fit(self, traces, max_windows=150_000, verbose=True):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        data = []
        for tr in traces:
            Fr = feature_frames(tr.X)
            bad = tr.labels > 0
            bad |= np.isnan(tr.X).all(2)
            # also drop the hour after a fault (recovery is not "normal" either)
            for f in tr.faults:
                bad[f.end:f.end + tr.cfg.ticks_per_hour, f.node] = True
            data.append(self._windows(Fr, bad, stride=3))
        D = np.concatenate(data)
        if len(D) > max_windows:
            D = D[rng.choice(len(D), max_windows, replace=False)]
        self.mu, self.sd = D.mean(0), D.std(0) + 1e-3
        Xt = torch.as_tensor((D - self.mu) / self.sd, dtype=torch.float32)
        self.model = _AE(Xt.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for ep in range(self.epochs):
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), 1024):
                b = Xt[perm[i:i + 1024]]
                loss = ((self.model(b) - b) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if verbose and (ep == 0 or (ep + 1) % 5 == 0):
                print(f"  ae epoch {ep + 1:2d}  recon {tot / len(Xt):.4f}")
        return self

    # --------------------------------------------------------- streaming
    def reset(self, n):
        self.live.reset(n)
        self.buf = None
        self.ewma_mu = None
        self.s = np.zeros(n)

    @torch.no_grad()
    def update(self, x):
        gone, dead = self.live(x)
        F = x.shape[1]
        pz = np.clip(np.nan_to_num(peer_z(x, FLOORS)), -10, 10)
        xl = np.log1p(np.nan_to_num(x, nan=0.0))
        if self.ewma_mu is None:
            self.ewma_mu = xl.copy()
        own = np.clip(xl - self.ewma_mu, -5, 5)
        self.ewma_mu[~dead] = 0.99 * self.ewma_mu[~dead] + 0.01 * xl[~dead]
        f = np.concatenate([pz, own], 1).astype(np.float32)
        if self.buf is None:
            self.buf = np.repeat(f[None], self.w, 0)
        else:
            self.buf = np.concatenate([self.buf[1:], f[None]], 0)
        win = self.buf.transpose(1, 0, 2).reshape(len(x), -1)
        z = torch.as_tensor((win - self.mu) / self.sd, dtype=torch.float32)
        raw = ((self.model(z) - z) ** 2).mean(1).numpy()
        raw[dead] = 0.0
        self.s = (1 - self.smooth) * self.s + self.smooth * raw
        out = self.s.copy()
        out[gone] = np.inf
        return out

    def state_dict(self):
        return {"model": self.model.state_dict(), "mu": self.mu, "sd": self.sd,
                "window": self.w, "d": int(self.mu.shape[0])}

    def load_state_dict(self, d):
        self.w = d["window"]
        self.mu, self.sd = d["mu"], d["sd"]
        self.model = _AE(d["d"])
        self.model.load_state_dict(d["model"])
        return self


class AlarmPolicy:
    """score > threshold for `k` consecutive ticks raises an alarm on a node; it
    clears after `clear` consecutive ticks below threshold. Returns the set of
    nodes whose alarm *started* on this tick."""

    def __init__(self, threshold, k=3, clear=10):
        self.thr, self.k, self.clear = threshold, k, clear

    def reset(self, n):
        self.above = np.zeros(n, dtype=int)
        self.below = np.zeros(n, dtype=int)
        self.active = np.zeros(n, dtype=bool)

    def update(self, s):
        hi = s > self.thr
        self.above = np.where(hi, self.above + 1, 0)
        self.below = np.where(hi, 0, self.below + 1)
        fire = (~self.active) & (self.above >= self.k)
        self.active |= fire
        self.active &= ~(self.below >= self.clear)
        return fire


class Hybrid:
    """Max of several detectors, each divided by its own calibrated threshold.
    Peer comparison is good at "this node is not like the others"; the
    autoencoder also looks at the node's own recent history, which helps when
    a node is legitimately hotter than its peers (data skew)."""

    name = "hybrid"

    def __init__(self, parts, scales):
        self.parts, self.scales = parts, np.asarray(scales, dtype=float)
        self.name = "hybrid(" + "+".join(p.name for p in parts) + ")"

    def reset(self, n):
        for p in self.parts:
            p.reset(n)

    def update(self, x):
        return np.max([p.update(x) / s for p, s in zip(self.parts, self.scales)], axis=0)
