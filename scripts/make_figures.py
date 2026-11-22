"""
Regenerate every figure in docs/figures from the JSON files in runs/.

    python scripts/make_figures.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cluster.simulator import FAULT_TYPES, M  # noqa: E402

OUT = "docs/figures"
# fixed categorical order, so a method keeps its colour in every figure
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METHOD_COLOR = {"PPO": C["blue"], "A*": C["orange"], "A* (replan)": C["aqua"], "RRT": C["violet"],
                "PPO + BC init": C["magenta"], "BC only": C["yellow"], "PPO, no curriculum": C["red"]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
})


def load(p):
    with open(p) as f:
        return json.load(f)


def smooth(x, k=8):
    x = np.asarray(x, float)
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ------------------------------------------------------------------ training
def training_curves():
    runs = {"PPO": sorted(glob.glob("runs/ppo_s[0-9]/training_history.json")),
            "PPO + BC init": glob.glob("runs/ppo_bc_s0/training_history.json"),
            "PPO, no curriculum": glob.glob("runs/ppo_nocurr_s0/training_history.json")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, paths in runs.items():
        if not paths:
            continue
        curves = []
        for p in paths:
            h = load(p)
            ep = np.array([r["episode"] for r in h])
            sr = np.array([r["success"] for r in h])
            grid = np.linspace(ep[4], ep.max(), 200)
            # trailing 5-update mean, so the ends aren't dragged down by padding
            sm = np.convolve(sr, np.ones(5) / 5, mode="valid")
            curves.append(np.interp(grid, ep[4:], sm))
        curves = np.array(curves)
        col = METHOD_COLOR[name]
        lab = f"{name} ({len(paths)} seeds)" if len(paths) > 1 else name
        a1.plot(grid, curves.mean(0), color=col, lw=2, label=lab)
        if len(paths) > 1:
            a1.fill_between(grid, curves.min(0), curves.max(0), color=col, alpha=0.15, lw=0)
    h = load(runs["PPO"][0])
    a2.plot([r["episode"] for r in h], [r["difficulty"] for r in h], color=INK2, lw=2)
    a1.set(xlabel="training episode", ylabel="success rate (training episodes)",
           title="Training progress", ylim=(0, 1))
    a1.legend(loc="lower right")
    a2.set(xlabel="training episode", ylabel="difficulty", title="Curriculum schedule", ylim=(0, 1.05))
    a2.text(0.98, 0.08, "3 to 14 static, 0 to 4 moving obstacles", transform=a2.transAxes,
            ha="right", color=INK2, fontsize=9)
    fig.savefig(f"{OUT}/training_curves.png")
    plt.close(fig)


# ------------------------------------------------------------------ benchmark
def benchmark_bars(path, fname, title):
    if not os.path.exists(path):
        return
    r = load(path)["methods"]
    order = [m for m in ["PPO", "PPO + BC init", "PPO, no curriculum", "BC only",
                         "A*", "A* (replan)", "RRT"] if m in r or f"{m} s0" in r]
    rows = []
    for m in order:
        seeds = [k for k in r if k == m or k.startswith(m + " s")]
        rows.append((m, [r[k] for k in seeds]))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (m, res) in zip(y, rows):
        succ = np.array([x["success"] for x in res])
        col = METHOD_COLOR.get(m, INK2)
        ax.barh(yi, succ.mean(), height=0.6, color=col)
        if len(res) == 1:
            lo, hi = res[0]["success_ci95"]
            ax.plot([lo, hi], [yi, yi], color=INK, lw=1.2)
            txt = f"{succ.mean():.0%}"
        else:
            ax.scatter(succ, [yi] * len(succ), color=INK, s=14, zorder=3)
            txt = f"{succ.mean():.0%} (seeds {', '.join(f'{s:.0%}' for s in succ)})"
        ax.text(1.01, yi, txt, va="center", color=INK, fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [f"{m} ({len(res)} seeds)" if len(res) > 1 else m for m, res in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("success rate on held-out seeds (bars: 95% Wilson CI, dots: training seeds)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/{fname}")
    plt.close(fig)


def noise_robustness():
    files = sorted(glob.glob("runs/noise/benchmark_noise*.json"))
    if not files:
        return
    data = {}
    for f in files:
        d = load(f)
        k = d["config"]["noise_scale"]
        for m, s in d["methods"].items():
            base = m.split(" s")[0] if m.startswith("PPO") else m
            data.setdefault(base, {}).setdefault(k, []).append(s["success"])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m, pts in data.items():
        ks = sorted(pts)
        mu = [np.mean(pts[k]) for k in ks]
        ax.plot(ks, mu, marker="o", ms=6, lw=2, color=METHOD_COLOR.get(m, INK2), label=m)
        if len(pts[ks[0]]) > 1:
            ax.fill_between(ks, [min(pts[k]) for k in ks], [max(pts[k]) for k in ks],
                            color=METHOD_COLOR.get(m, INK2), alpha=0.15, lw=0)
    ax.set(xlabel="sensor noise multiplier (1 = training noise)", ylabel="success rate",
           title="Robustness to sensor degradation", ylim=(0, 1))
    ax.set_xticks(sorted({k for p in data.values() for k in p}))
    ax.legend(loc="lower left")
    fig.savefig(f"{OUT}/noise_robustness.png")
    plt.close(fig)


# ------------------------------------------------------------------ cluster
DET_COLOR = {"static-threshold": C["yellow"], "ewma-zscore": C["orange"], "peer-mad": C["aqua"],
             "window-ae": C["blue"], "hybrid(peer-mad+window-ae)": C["violet"]}
DET_LABEL = {"hybrid(peer-mad+window-ae)": "hybrid (peer + AE)"}


def cluster_figs():
    p = "runs/cluster/results.json"
    if not os.path.exists(p):
        return
    r = load(p)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, pts in r["sweep"].items():
        fa = [max(x["fa_per_node_day"], 0.004) for x in pts]   # 0 can't sit on a log axis
        rc = [x["recall"] for x in pts]
        o = np.argsort(fa)
        ax.plot(np.array(fa)[o], np.array(rc)[o], lw=2, color=DET_COLOR[name],
                label=DET_LABEL.get(name, name))
        d = r["detectors"][name]
        ax.scatter([d["fa_per_node_day"]], [d["recall"]], color=DET_COLOR[name], s=40,
                   edgecolor="white", linewidth=1.5, zorder=3)
    s = r["detectors"]["static-threshold"]
    ax.scatter([s["fa_per_node_day"]], [s["recall"]], marker="s", s=40,
               color=DET_COLOR["static-threshold"], label="static thresholds", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.004, 12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0.5, 1.01)
    ax.set(xlabel="false alarms per node-day (test clusters)", ylabel="faults detected (recall)",
           title="Recall vs false alarms (dots: thresholds calibrated on val)")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(f"{OUT}/cluster_operating_curves.png")
    plt.close(fig)

    # detection delay per fault type
    names = [n for n in r["detectors"] if n != "static-threshold"]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [r["detectors"][n]["delay_median_by_kind"][k] or np.nan for k in FAULT_TYPES]
        ax.bar(np.arange(len(FAULT_TYPES)) + (i - (len(names) - 1) / 2) * w, vals, width=w * 0.92,
               color=DET_COLOR[n], label=DET_LABEL.get(n, n))
    ax.set_xticks(range(len(FAULT_TYPES)), FAULT_TYPES)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(ylabel="median detection delay (min, log scale)", title="How fast each fault type is caught")
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{OUT}/cluster_delay_by_type.png")
    plt.close(fig)

    # one fault, raw metric + scores
    z = np.load("runs/cluster/example_scores.npz", allow_pickle=True)
    ev = [e for e in r["events"]["peer-mad"] if e["cluster"] == r["config"]["test_seeds"][0]
          and e["kind"] == "disk"]
    if ev:
        e = ev[0]
        n, t0 = e["node"], e["onset"]
        lo, hi = max(0, t0 - 240), min(len(z["X"]), t0 + 300)
        t = (np.arange(lo, hi) - t0) * r["config"]["tick_s"] / 60
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
        io = z["X"][lo:hi, :, M["io_wait_pct"]]
        a1.plot(t, np.nanmedian(io, 1), color=INK2, lw=1.2, label="cluster median")
        a1.plot(t, io[:, n], color=C["orange"], lw=1.5, label=f"node {n}")
        a1.set(ylabel="io_wait %", title="A disk fault as the detectors see it (fault starts at 0)")
        a1.legend(loc="upper left", fontsize=9)
        names_ = list(z["names"])
        thr = dict(zip(names_, z["thresholds"]))
        for key, nm in [("peer_mad", "peer-mad"), ("window_ae", "window-ae")]:
            s = z[key][lo:hi, n] / thr[nm]
            a2.plot(t, np.minimum(s, 6), color=DET_COLOR[nm], lw=1.5, label=DET_LABEL.get(nm, nm))
        a2.axhline(1, color=INK, lw=1, ls="--")
        a2.text(t[0], 1.1, "alarm threshold", color=INK2, fontsize=8)
        a2.set(xlabel="minutes from fault onset", ylabel="score / threshold")
        a2.legend(loc="upper left", fontsize=9)
        for a in (a1, a2):
            a.axvspan(0, t[-1], color=C["red"], alpha=0.06, lw=0)
        fig.savefig(f"{OUT}/cluster_example.png")
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    training_curves()
    benchmark_bars("runs/benchmark_d1.0.json", "benchmark_full.png",
                   "Full difficulty (14 static + 4 moving obstacles), 200 held-out episodes")
    benchmark_bars("runs/benchmark_d0.5.json", "benchmark_medium.png",
                   "Medium difficulty, 200 held-out episodes")
    noise_robustness()
    cluster_figs()
    print("figures written to", OUT)
