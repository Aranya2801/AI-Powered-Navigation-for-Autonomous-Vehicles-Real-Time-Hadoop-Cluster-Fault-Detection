"""
End-to-end evaluation of the cluster fault detectors.

Splits (separate simulated clusters, 40 nodes x 2 days each):
    train  seeds 0-7      AE training + diagnosis classifier
    val    seeds 100-103  threshold calibration
    test   seeds 1000-1007  everything reported

Each detector's alarm threshold is set on *val* so that it produces at most
`--fa_budget` false alarms per node-day, then frozen and applied to test. The
static-threshold baseline is run with its own fixed limits (that's the point of
it), alarm on any breached limit.

Metrics (event level, which is what an on-call engineer feels):
    recall        fraction of faults with an alarm on the right node between
                  onset and end + grace
    delay         minutes from fault onset to that first alarm
    FA/node-day   alarms that don't overlap any fault window
    precision     true alarms / all alarms

    python -m cluster.evaluate --out runs/cluster
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch

from cluster.simulator import HadoopClusterSim, ClusterConfig, FAULT_TYPES
from cluster.detectors import (StaticThreshold, EWMAZScore, PeerMAD, WindowAutoencoder,
                               Hybrid, AlarmPolicy, feature_frames)
from cluster.diagnosis import FaultDiagnoser

TRAIN, VAL, TEST = range(0, 8), range(100, 104), range(1000, 1008)
GRACE = 10   # ticks after a fault ends that still count as "on time"


def gen(seeds, cfg):
    return [HadoopClusterSim(cfg, seed=s).generate() for s in seeds]


def stream_scores(det, trace):
    """Replay the trace tick by tick, the way a consumer on the metrics topic
    would see it. Returns (T, N) scores and mean ms per tick."""
    T, N, _ = trace.X.shape
    det.reset(N)
    S = np.zeros((T, N))
    t0 = time.perf_counter()
    for t in range(T):
        S[t] = det.update(trace.X[t])
    return S, 1000 * (time.perf_counter() - t0) / T


def alarms_from_scores(S, thr, k=3, clear=10):
    pol = AlarmPolicy(thr, k, clear)
    T, N = S.shape
    pol.reset(N)
    out = []
    for t in range(T):
        for n in np.flatnonzero(pol.update(S[t])):
            out.append((t, int(n)))
    return out


def score_events(alarms, trace):
    tph = trace.cfg.ticks_per_hour
    windows = {}
    for i, f in enumerate(trace.faults):
        windows.setdefault(f.node, []).append((f.onset, f.end + GRACE, i))
    first = {}
    false_alarms = 0
    fa_hot = 0
    for t, n in alarms:
        hit = None
        for s, e, i in windows.get(n, []):
            if s <= t <= e:
                hit = i
                break
        if hit is None:
            false_alarms += 1
            fa_hot += any(hn == n and hs <= t <= he + GRACE for hn, hs, he in trace.hotspots)
        elif hit not in first:
            first[hit] = t
    per = []
    for i, f in enumerate(trace.faults):
        det = i in first
        per.append({"kind": f.kind, "node": f.node, "onset": f.onset,
                    "detected": det, "alarm_tick": first.get(i),
                    "delay_min": (first[i] - f.onset) * trace.cfg.tick_s / 60 if det else None})
    node_days = trace.cfg.n_nodes * trace.cfg.days
    return per, false_alarms, len(alarms), node_days, fa_hot


def calibrate(S_list, traces, budget, k=3, clear=10):
    """Smallest threshold whose val false-alarm rate is within budget."""
    finite = np.concatenate([S[np.isfinite(S)] for S in S_list])
    cands = np.unique(np.quantile(finite, np.linspace(0.90, 0.99999, 120)))
    best = cands[-1]
    for thr in cands:          # ascending: lowest threshold that meets budget
        fa = nd = 0
        for S, tr in zip(S_list, traces):
            _, f, _, d, _ = score_events(alarms_from_scores(S, thr, k, clear), tr)
            fa, nd = fa + f, nd + d
        if fa / nd <= budget:
            best = thr
            break
    return float(best)


def threshold_sweep(S_list, traces, n=40):
    """Recall vs false-alarm rate on test over a range of thresholds - the
    operating curve, so detectors can be compared at equal FA rates too."""
    finite = np.concatenate([S[np.isfinite(S)] for S in S_list])
    out = []
    for thr in np.unique(np.quantile(finite, 1 - np.logspace(-1.3, -4.7, n))):
        det = tot = fa = nd = 0
        for S, tr in zip(S_list, traces):
            per, f, _, d, _ = score_events(alarms_from_scores(S, thr), tr)
            det += sum(p["detected"] for p in per)
            tot += len(per)
            fa, nd = fa + f, nd + d
        out.append({"threshold": float(thr), "recall": det / tot, "fa_per_node_day": fa / nd})
    return out


def summarise(per, fa, n_alarms, node_days, ms):
    det = [p for p in per if p["detected"]]
    delays = [p["delay_min"] for p in det]
    out = {
        "faults": len(per), "detected": len(det),
        "recall": len(det) / max(1, len(per)),
        "fa_per_node_day": fa / node_days,
        "precision": (n_alarms - fa) / n_alarms if n_alarms else None,
        "delay_median_min": float(np.median(delays)) if delays else None,
        "delay_mean_min": float(np.mean(delays)) if delays else None,
        "delay_p90_min": float(np.percentile(delays, 90)) if delays else None,
        "ms_per_tick": ms,
        "recall_by_kind": {},
        "delay_median_by_kind": {},
    }
    for k in FAULT_TYPES:
        pk = [p for p in per if p["kind"] == k]
        dk = [p["delay_min"] for p in pk if p["detected"]]
        out["recall_by_kind"][k] = len(dk) / len(pk) if pk else None
        out["delay_median_by_kind"][k] = float(np.median(dk)) if dk else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cluster")
    ap.add_argument("--fa_budget", type=float, default=0.25, help="false alarms per node-day on val")
    ap.add_argument("--n_nodes", type=int, default=40)
    ap.add_argument("--days", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = ClusterConfig(n_nodes=args.n_nodes, days=args.days)

    t0 = time.time()
    print("simulating clusters...")
    train, val, test = gen(TRAIN, cfg), gen(VAL, cfg), gen(TEST, cfg)
    print(f"  {len(train)}/{len(val)}/{len(test)} clusters, "
          f"{sum(len(t.faults) for t in test)} test faults ({time.time() - t0:.0f}s)")

    print("training window autoencoder on normal data...")
    ae = WindowAutoencoder().fit(train)
    torch.save(ae.state_dict(), os.path.join(args.out, "window_ae.pt"))

    print("training fault-type classifier...")
    diag = FaultDiagnoser().fit(train)

    peer = PeerMAD()
    detectors = [StaticThreshold(), EWMAZScore(), peer, ae, None]
    results, all_events, scores_test, sweep = {}, {}, {}, {}
    for j, det in enumerate(detectors):
        if det is None:
            # built last, from the thresholds the two components got on val
            det = Hybrid([PeerMAD(), ae], [results[peer.name]["threshold"],
                                          results[ae.name]["threshold"]])
            detectors[j] = det
        if det.name == "static-threshold":
            thr = 0.5
        else:
            Sv = [stream_scores(det, tr)[0] for tr in val]
            thr = calibrate(Sv, val, args.fa_budget)
        per_all, fa_all, na_all, nd_all, ms_all, hot_all = [], 0, 0, 0, [], 0
        for tr in test:
            S, ms = stream_scores(det, tr)
            scores_test.setdefault(det.name, []).append(S)
            per, fa, na, nd, hot = score_events(alarms_from_scores(S, thr), tr)
            hot_all += hot
            for p in per:
                p["cluster"] = tr.seed
            per_all += per
            fa_all, na_all, nd_all = fa_all + fa, na_all + na, nd_all + nd
            ms_all.append(ms)
        summ = summarise(per_all, fa_all, na_all, nd_all, float(np.mean(ms_all)))
        if det.name != "static-threshold":
            sweep[det.name] = threshold_sweep(scores_test[det.name], test)
        summ["threshold"] = thr
        summ["false_alarms"] = fa_all
        summ["false_alarms_on_hotspots"] = hot_all
        results[det.name] = summ
        all_events[det.name] = per_all
        print(f"{det.name:17s} recall {summ['recall']:.2f}  FA/node-day {summ['fa_per_node_day']:.3f}  "
              f"precision {summ['precision'] if summ['precision'] is None else round(summ['precision'], 2)}  "
              f"median delay {summ['delay_median_min']} min  {summ['ms_per_tick']:.2f} ms/tick")

    # diagnosis: classify every fault the best detector caught, at alarm time
    best = max((d for d in results if d != "static-threshold"),
               key=lambda d: (results[d]["recall"], -results[d]["delay_median_min"]))
    frames = {tr.seed: feature_frames(tr.X) for tr in test}
    raw = {tr.seed: tr.X for tr in test}
    y_true, y_pred = [], []
    for p in all_events[best]:
        if not p["detected"]:
            continue
        # classify a few ticks after the alarm: that's when the on-call page goes out
        t = min(p["alarm_tick"] + 5, len(raw[p["cluster"]]) - 1)
        pred = diag.predict_one(frames[p["cluster"]], raw[p["cluster"]], t, p["node"])
        y_true.append(p["kind"])
        y_pred.append(pred)
    cm = np.zeros((len(FAULT_TYPES), len(FAULT_TYPES)), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[FAULT_TYPES.index(a), FAULT_TYPES.index(b)] += 1
    diag_res = {"detector": best, "n": len(y_true),
                "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else None,
                "labels": FAULT_TYPES, "confusion": cm.tolist()}
    print(f"diagnosis on faults caught by {best}: accuracy {diag_res['accuracy']:.2f} (n={len(y_true)})")

    out = {"config": {**vars(args), "train_seeds": list(TRAIN), "val_seeds": list(VAL),
                      "test_seeds": list(TEST), "grace_ticks": GRACE, "tick_s": cfg.tick_s},
           "detectors": results, "diagnosis": diag_res, "sweep": sweep, "events": all_events,
           "wall_time_s": time.time() - t0}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "example_scores.npz"),
                        X=test[0].X, labels=test[0].labels,
                        **{re.sub(r"\W", "_", k): v[0] for k, v in scores_test.items()},
                        thresholds=np.array([results[d.name]["threshold"] for d in detectors]),
                        names=np.array([d.name for d in detectors]))
    print(f"wrote {args.out}/results.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
