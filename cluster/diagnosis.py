"""
Fault-type classifier ("what kind of problem is this node having?").

Runs only after a detector has raised an alarm. Input is a summary of the last
`window` ticks for that node: mean and max of the peer z-scores and of the
own-history features, plus the fraction of missing reports. Trained on labelled
faults from the training clusters (in a real deployment, past incident tickets).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from cluster.detectors import feature_frames
from cluster.simulator import FAULT_TYPES


def summary_features(frames, X, t, node, window=10):
    lo = max(0, t - window + 1)
    f = frames[lo:t + 1, node]
    miss = np.isnan(X[lo:t + 1, node]).all(1).mean()
    return np.concatenate([f.mean(0), f.max(0), f.min(0), [miss]])


class FaultDiagnoser:
    def __init__(self, window=10, seed=0):
        self.window = window
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                  random_state=seed)

    def fit(self, traces, per_fault=25):
        rng = np.random.default_rng(0)
        Xs, ys = [], []
        for tr in traces:
            frames = feature_frames(tr.X)
            for f in tr.faults:
                # sample from the early part of each fault, where it matters
                hi = min(f.end, f.onset + max(f.ramp, 30))
                for t in rng.integers(f.onset + 2, hi + 1, per_fault):
                    Xs.append(summary_features(frames, tr.X, int(t), f.node, self.window))
                    ys.append(f.kind)
        self.clf.fit(np.array(Xs), np.array(ys))
        return self

    def predict_one(self, frames, X, t, node):
        return str(self.clf.predict(summary_features(frames, X, t, node, self.window)[None])[0])
