"""
Synthetic Hadoop cluster telemetry with injected faults.

Why synthetic: public traces (Google 2019, Alibaba 2018) have resource usage but
no ground-truth hardware/JVM fault labels at the node level, which is exactly
what is needed to measure detection delay. So the cluster is simulated, with the
metric set modelled on what DataNode/NodeManager JMX + node exporters expose,
and faults are injected with known onset times. The simulator is the weakest
part of this module and the results should be read with that in mind.

Each node reports one row every `tick_s` seconds (default 30 s):

    cpu_pct, mem_pct, jvm_heap_pct, gc_pause_ms, io_wait_pct, disk_read_mbps,
    disk_write_mbps, net_in_mbps, net_out_mbps, tcp_retrans_pct,
    heartbeat_ms, containers, failed_block_ops, task_failures

A node that is down reports nothing (row of NaN).

Workload is shared across the cluster: a daily cycle (fleet telemetry ingest
follows vehicle activity), a nightly ETL batch, and Poisson-arriving MapReduce/
Spark jobs that land on random subsets of nodes and end with a shuffle phase.
There are also benign events that look scary to a naive threshold: full GC
spikes, disk compaction bursts, and NameNode GC pauses that slow every node's
heartbeat at the same time.

Fault types (per node, ramped in so the onset is subtle):
    disk        failing disk: io_wait up, throughput down, block op failures
    memleak     JVM leak: heap floor creeps up, GC pauses grow, ends in OOM restart
    network     NIC/switch-port problem: heartbeat latency, retransmits, task failures
    cpu         runaway process / noisy neighbour pinning the CPU
    crash       node goes silent (no heartbeat) and comes back later
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METRICS = [
    "cpu_pct", "mem_pct", "jvm_heap_pct", "gc_pause_ms", "io_wait_pct",
    "disk_read_mbps", "disk_write_mbps", "net_in_mbps", "net_out_mbps",
    "tcp_retrans_pct", "heartbeat_ms", "containers", "failed_block_ops", "task_failures",
]
M = {m: i for i, m in enumerate(METRICS)}
FAULT_TYPES = ["disk", "memleak", "network", "cpu", "crash"]


@dataclass
class Fault:
    node: int
    kind: str
    onset: int          # first tick the fault is active
    end: int            # last tick (inclusive)
    ramp: int           # ticks until full severity
    peak: float = 1.0   # how bad it gets; many real faults stay mild

    def severity(self, t):
        if t < self.onset or t > self.end:
            return 0.0
        return self.peak * min(1.0, (t - self.onset + 1) / max(1, self.ramp))


@dataclass
class ClusterConfig:
    n_nodes: int = 40
    tick_s: int = 30
    days: float = 2.0
    faults_per_node_day: float = 0.15
    job_rate: float = 0.02            # new jobs per tick
    fault_mix: dict = field(default_factory=lambda: {
        "disk": 0.25, "memleak": 0.2, "network": 0.2, "cpu": 0.2, "crash": 0.15})

    @property
    def ticks(self):
        return int(self.days * 86400 / self.tick_s)

    @property
    def ticks_per_hour(self):
        return 3600 // self.tick_s


@dataclass
class ClusterTrace:
    X: np.ndarray                 # (T, N, F) metrics, NaN when a node is down
    labels: np.ndarray            # (T, N) int: 0 normal, 1.. index into FAULT_TYPES + 1
    faults: list
    cfg: ClusterConfig
    seed: int
    hotspots: list = field(default_factory=list)   # (node, start, end), benign

    def fault_mask(self):
        return self.labels > 0


class HadoopClusterSim:
    def __init__(self, cfg: ClusterConfig | None = None, seed: int = 0):
        self.cfg = cfg or ClusterConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ faults
    def _schedule_faults(self):
        cfg, rng = self.cfg, self.rng
        T, N, H = cfg.ticks, cfg.n_nodes, cfg.ticks_per_hour
        kinds = list(cfg.fault_mix)
        p = np.array([cfg.fault_mix[k] for k in kinds])
        p = p / p.sum()
        n_faults = rng.poisson(cfg.faults_per_node_day * N * cfg.days)
        faults, busy = [], {i: [] for i in range(N)}
        dur_h = {"disk": (1.0, 3.0), "memleak": (1.5, 4.0), "network": (0.5, 2.0),
                 "cpu": (0.5, 2.0), "crash": (0.3, 2.0)}
        ramp_h = {"disk": (0.3, 1.0), "memleak": (1.0, 3.0), "network": (0.1, 0.5),
                  "cpu": (0.05, 0.3), "crash": (0.0, 0.0)}
        for _ in range(n_faults):
            for _ in range(50):
                kind = kinds[rng.choice(len(kinds), p=p)]
                node = int(rng.integers(N))
                dur = int(rng.uniform(*dur_h[kind]) * H)
                onset = int(rng.integers(H, T - dur - 1))   # nothing in the first hour
                end = onset + dur
                # one fault at a time per node, with 2h of quiet in between
                if all(end + 2 * H < s or onset > e + 2 * H for s, e in busy[node]):
                    ramp = max(1, int(rng.uniform(*ramp_h[kind]) * H))
                    if kind == "memleak":
                        ramp = dur                      # leaks get worse until OOM
                    peak = 1.0 if kind == "crash" else float(rng.uniform(0.25, 1.0))
                    faults.append(Fault(node, kind, onset, end, ramp, peak))
                    busy[node].append((onset, end))
                    break
        return sorted(faults, key=lambda f: f.onset)

    # ------------------------------------------------------------ workload
    def _workload(self):
        cfg, rng = self.cfg, self.rng
        T, N, H = cfg.ticks, cfg.n_nodes, cfg.ticks_per_hour
        t = np.arange(T)
        day = 24 * H
        L = 0.42 + 0.22 * np.sin(2 * np.pi * (t / day) - np.pi / 2 - 0.6)  # peak in the evening
        etl = np.zeros(T)
        for d in range(int(np.ceil(cfg.days)) + 1):
            s = int(d * day + 1 * H + rng.integers(-10, 10))
            etl[max(0, s):max(0, s + int(1.5 * H))] = 0.3        # nightly batch on every node
        node_load = np.repeat((L + etl)[:, None], N, axis=1)
        shuffle = np.zeros((T, N))
        n_jobs = rng.poisson(cfg.job_rate * T)
        for _ in range(n_jobs):
            s = int(rng.integers(T))
            dur = int(rng.uniform(20, 240))
            nodes = rng.choice(N, size=int(rng.integers(N // 5, N)), replace=False)
            amp = rng.uniform(0.05, 0.25)
            e = min(T, s + dur)
            node_load[s:e, nodes] += amp
            sh = s + int(0.7 * dur)
            shuffle[sh:e, nodes] += amp * 2.5
        # benign hot spots: data skew sends one node a lot more work for a while.
        # Looks like a CPU problem to a detector that only compares against peers.
        self.hotspots = []
        for _ in range(rng.poisson(0.15 * N * cfg.days)):
            n, s = int(rng.integers(N)), int(rng.integers(T))
            dur, amp = int(rng.uniform(0.5, 3) * H), rng.uniform(0.25, 0.55)
            node_load[s:s + dur, n] += amp
            self.hotspots.append((n, s, min(T - 1, s + dur)))
        # per-node AR(1) jitter
        ar = np.zeros((T, N))
        eps = rng.normal(0, 0.03, (T, N))
        for i in range(1, T):
            ar[i] = 0.9 * ar[i - 1] + eps[i]
        return np.clip(node_load + ar, 0.02, 1.3), np.clip(shuffle, 0, 2)

    # ------------------------------------------------------------ main
    def generate(self) -> ClusterTrace:
        cfg, rng = self.cfg, self.rng
        T, N = cfg.ticks, cfg.n_nodes
        load, shuffle = self._workload()
        faults = self._schedule_faults()
        X = np.zeros((T, N, len(METRICS)))
        labels = np.zeros((T, N), dtype=np.int64)

        # node heterogeneity (different hardware generations / disk ages)
        off_cpu = rng.normal(0, 4, N)
        off_io = rng.uniform(0.8, 1.3, N)
        off_hb = rng.uniform(0.8, 1.2, N)
        heap = rng.uniform(35, 60, N)

        # benign cluster-wide NameNode GC pauses: every node's heartbeat slows
        nn_gc = np.zeros(T)
        for s in rng.integers(0, T, rng.poisson(T / 600)):
            nn_gc[s:s + int(rng.integers(1, 4))] = rng.uniform(40, 150)

        by_node = {i: [f for f in faults if f.node == i] for i in range(N)}
        oom_down = np.zeros(N, dtype=int)

        for t in range(T):
            l = load[t]
            noise = rng.normal(0, 1, (N, len(METRICS)))
            cpu = 8 + 78 * l + off_cpu + 3 * noise[:, 0]
            mem = 28 + 52 * l + 2.5 * noise[:, 1]
            # JVM heap: allocation proportional to load, collected when it gets full
            leak_floor = np.zeros(N)
            for i in range(N):
                for f in by_node[i]:
                    if f.kind == "memleak":
                        leak_floor[i] = max(leak_floor[i], 60 * f.severity(t))
            heap += 2.0 * l + 0.5 * np.abs(noise[:, 2])
            gc = 15 + 25 * l + 6 * np.abs(noise[:, 3])
            full = heap > 80
            gc = gc + full * rng.normal(260, 60, N)
            heap = np.where(full, 30 + leak_floor + rng.normal(0, 3, N), heap)
            heap = np.maximum(heap, 30 + leak_floor)
            io = (2 + 9 * l) * off_io + 1.2 * (1 + l) * np.abs(noise[:, 4])
            comp = rng.random(N) < 0.004                       # compaction burst
            io = io + comp * rng.uniform(10, 30, N)
            rd = 30 + 170 * l + 10 * noise[:, 5]
            wr = 20 + 140 * l + 10 * noise[:, 6]
            nin = 40 + 260 * l + 350 * shuffle[t] + 15 * noise[:, 7]
            nout = 35 + 240 * l + 350 * shuffle[t] + 15 * noise[:, 8]
            retr = 0.08 + 0.04 * np.abs(noise[:, 9])
            hb = (3 + 4 * l) * off_hb * np.exp(0.25 * noise[:, 10]) + nn_gc[t]
            cont = np.round(np.clip(16 * l + 1.2 * noise[:, 11], 0, 24))
            fbo = rng.poisson(0.02, N).astype(float)
            tf = rng.poisson(0.04 + 0.03 * cont / 16).astype(float)

            row = np.stack([cpu, mem, heap.copy(), gc, io, rd, wr, nin, nout, retr,
                            hb, cont, fbo, tf], axis=1)

            for i in range(N):
                for f in by_node[i]:
                    s = f.severity(t)
                    if s <= 0:
                        continue
                    labels[t, i] = FAULT_TYPES.index(f.kind) + 1
                    r = row[i]
                    if f.kind == "disk":
                        r[M["io_wait_pct"]] += 35 * s
                        r[M["disk_read_mbps"]] *= 1 - 0.6 * s
                        r[M["disk_write_mbps"]] *= 1 - 0.6 * s
                        r[M["failed_block_ops"]] += rng.poisson(2.5 * s)
                        r[M["heartbeat_ms"]] += 15 * s * rng.random()
                    elif f.kind == "memleak":
                        r[M["gc_pause_ms"]] += 500 * s ** 2 * rng.uniform(0.5, 1.5)
                        r[M["mem_pct"]] += 12 * s
                        r[M["task_failures"]] += rng.poisson(1.5 * s ** 2)
                    elif f.kind == "network":
                        r[M["heartbeat_ms"]] += 120 * s * rng.uniform(0.3, 1.5)
                        r[M["tcp_retrans_pct"]] += 4 * s * rng.uniform(0.5, 1.5)
                        r[M["net_in_mbps"]] *= 1 - 0.6 * s
                        r[M["net_out_mbps"]] *= 1 - 0.6 * s
                        r[M["task_failures"]] += rng.poisson(1.5 * s)
                    elif f.kind == "cpu":
                        r[M["cpu_pct"]] = max(r[M["cpu_pct"]], 75 + 23 * s + rng.normal(0, 1.5))
                        r[M["gc_pause_ms"]] += 60 * s
                        r[M["task_failures"]] += rng.poisson(0.5 * s)
                    elif f.kind == "crash":
                        r[:] = np.nan
                    if f.kind == "memleak" and t == f.end and f.peak > 0.7:
                        oom_down[i] = int(rng.integers(4, 12))   # OOM kill + restart
                        heap[i] = 35.0
            down = oom_down > 0
            row[down] = np.nan
            oom_down = np.maximum(0, oom_down - 1)

            row[:, M["cpu_pct"]] = np.clip(row[:, M["cpu_pct"]], 0, 100)
            row[:, M["mem_pct"]] = np.clip(row[:, M["mem_pct"]], 0, 100)
            row[:, M["jvm_heap_pct"]] = np.clip(row[:, M["jvm_heap_pct"]], 0, 100)
            row[:, M["io_wait_pct"]] = np.clip(row[:, M["io_wait_pct"]], 0, 100)
            X[t] = np.where(np.isnan(row), np.nan, np.maximum(row, 0))
        return ClusterTrace(X, labels, faults, cfg, self.seed, self.hotspots)
