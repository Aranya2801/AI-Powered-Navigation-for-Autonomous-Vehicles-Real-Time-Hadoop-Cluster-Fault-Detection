import numpy as np
import pytest

from cluster.simulator import HadoopClusterSim, ClusterConfig, METRICS, M, FAULT_TYPES, Fault
from cluster.detectors import (PeerMAD, EWMAZScore, StaticThreshold, WindowAutoencoder,
                               AlarmPolicy, Hybrid, feature_frames, peer_z)
from cluster.evaluate import score_events, alarms_from_scores


@pytest.fixture(scope="module")
def small_trace():
    cfg = ClusterConfig(n_nodes=12, days=0.5, faults_per_node_day=0.6)
    return HadoopClusterSim(cfg, seed=3).generate()


def test_trace_shapes_and_labels(small_trace):
    tr = small_trace
    T, N = tr.cfg.ticks, tr.cfg.n_nodes
    assert tr.X.shape == (T, N, len(METRICS))
    assert tr.labels.shape == (T, N)
    assert len(tr.faults) > 0
    for f in tr.faults:
        assert (tr.labels[f.onset:f.end + 1, f.node] == FAULT_TYPES.index(f.kind) + 1).all()


def test_simulator_is_deterministic():
    cfg = ClusterConfig(n_nodes=6, days=0.1)
    a = HadoopClusterSim(cfg, seed=11).generate()
    b = HadoopClusterSim(cfg, seed=11).generate()
    assert np.array_equal(np.nan_to_num(a.X, nan=-1), np.nan_to_num(b.X, nan=-1))


def test_faults_do_not_overlap_on_a_node(small_trace):
    by = {}
    for f in small_trace.faults:
        by.setdefault(f.node, []).append((f.onset, f.end))
    for spans in by.values():
        spans.sort()
        for (s1, e1), (s2, _) in zip(spans, spans[1:]):
            assert s2 > e1


def test_crash_rows_are_missing(small_trace):
    for f in small_trace.faults:
        if f.kind == "crash":
            assert np.isnan(small_trace.X[f.onset:f.end + 1, f.node]).all()


def test_fault_severity_ramps():
    f = Fault(0, "disk", onset=10, end=50, ramp=10, peak=0.8)
    assert f.severity(9) == 0 and f.severity(51) == 0
    assert f.severity(14) == pytest.approx(0.4)
    assert f.severity(30) == pytest.approx(0.8)


def test_peer_z_ignores_cluster_wide_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(50, 2, (30, len(METRICS)))
    z1 = peer_z(x)
    z2 = peer_z(x + 40)        # whole cluster gets busier
    assert np.allclose(z1, z2)


def test_peer_mad_flags_the_odd_node():
    rng = np.random.default_rng(0)
    det = PeerMAD()
    det.reset(20)
    for _ in range(20):
        x = rng.normal(10, 1, (20, len(METRICS)))
        x[7, M["io_wait_pct"]] += 30
        s = det.update(x)
    assert int(np.argmax(s)) == 7


@pytest.mark.parametrize("cls", [PeerMAD, EWMAZScore, StaticThreshold])
def test_missing_heartbeats_give_infinite_score(cls):
    det = cls()
    det.reset(4)
    x = np.ones((4, len(METRICS)))
    det.update(x)
    x2 = x.copy()
    x2[2] = np.nan
    det.update(x2)
    s = det.update(x2)
    assert np.isinf(s[2]) and np.isfinite(s[[0, 1, 3]]).all()


def test_alarm_policy_debounce():
    pol = AlarmPolicy(1.0, k=3, clear=2)
    pol.reset(1)
    seq = [2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 2]
    fired = [bool(pol.update(np.array([v]))[0]) for v in seq]
    # fires once after 3 highs, a single dip doesn't clear it, two lows do
    assert fired == [False, False, True, False, False, False, False, False, False, False, True]


def test_streaming_ae_features_match_batch(small_trace):
    """The AE's streaming feature computation must equal feature_frames()."""
    X = small_trace.X[:60]
    Fr = feature_frames(X)
    ae = WindowAutoencoder(window=4)
    ae.mu, ae.sd = np.zeros(4 * Fr.shape[2]), np.ones(4 * Fr.shape[2])
    from cluster.detectors import _AE
    ae.model = _AE(4 * Fr.shape[2])
    ae.reset(X.shape[1])
    for t in range(len(X)):
        ae.update(X[t])
    expect = Fr[-4:].transpose(1, 0, 2).reshape(X.shape[1], -1)
    assert np.allclose(ae.buf.transpose(1, 0, 2).reshape(X.shape[1], -1), expect, atol=1e-5)


def test_event_scoring(small_trace):
    tr = small_trace
    f = tr.faults[0]
    alarms = [(f.onset + 4, f.node), (5, (f.node + 1) % tr.cfg.n_nodes)]
    per, fa, n, node_days, _ = score_events(alarms, tr)
    assert per[0]["detected"] and per[0]["delay_min"] == pytest.approx(4 * tr.cfg.tick_s / 60)
    assert n == 2 and fa >= 1


def test_hybrid_scales_components():
    class Const:
        name = "c"
        def __init__(self, v): self.v = v
        def reset(self, n): pass
        def update(self, x): return np.full(len(x), self.v)
    h = Hybrid([Const(2.0), Const(9.0)], [1.0, 10.0])
    h.reset(3)
    assert np.allclose(h.update(np.zeros((3, 2))), 2.0)
