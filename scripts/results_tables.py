"""Print the markdown tables used in README / docs/results.md from runs/*.json,
so the numbers in the docs are copied from the outputs, not typed by hand."""
import json

import numpy as np


def nav_table(path):
    r = json.load(open(path))["methods"]
    groups = {}
    for k, v in r.items():
        groups.setdefault(k.split(" s")[0] if k.startswith("PPO s") else k, []).append(v)
    print("| Method | Success | Collision (static / moving) | Timeout | Path efficiency | ms / step |")
    print("|---|---|---|---|---|---|")
    for m, vs in groups.items():
        f = lambda key: np.mean([v[key] for v in vs])
        if len(vs) > 1:
            s = [v["success"] for v in vs]
            succ = f"**{np.mean(s):.1%}** ± {np.std(s, ddof=1):.1%} (3 seeds)"
            name = "PPO (ours)"
        else:
            lo, hi = vs[0]["success_ci95"]
            succ = f"{vs[0]['success']:.1%} [{lo:.0%}, {hi:.0%}]"
            name = m
        print(f"| {name} | {succ} | {f('collision'):.1%} ({f('collision_static'):.1%} / "
              f"{f('collision_dynamic'):.1%}) | {f('timeout'):.1%} | {f('path_efficiency'):.2f} | "
              f"{f('ms_per_step'):.2f} |")
    print()


def noise_table():
    rows = {}
    for k in ["0.5", "1.0", "2.0", "3.0", "4.0"]:
        r = json.load(open(f"runs/noise/benchmark_noise{k}.json"))["methods"]
        g = {}
        for m, v in r.items():
            g.setdefault(m.split(" s")[0] if m.startswith("PPO s") else m, []).append(v["success"])
        for m, v in g.items():
            rows.setdefault(m, []).append(np.mean(v))
    print("| Method | 0.5x | 1x | 2x | 3x | 4x |")
    print("|---|---|---|---|---|---|")
    for m, v in rows.items():
        print(f"| {m} | " + " | ".join(f"{x:.0%}" for x in v) + " |")
    print()


def cluster_table():
    r = json.load(open("runs/cluster/results.json"))
    print("| Detector | Recall | FA / node-day | Precision | Median delay | p90 delay | ms / tick (40 nodes) |")
    print("|---|---|---|---|---|---|---|")
    for n, d in r["detectors"].items():
        print(f"| {n} | {d['recall']:.1%} | {d['fa_per_node_day']:.2f} | {d['precision']:.0%} | "
              f"{d['delay_median_min']:.1f} min | {d['delay_p90_min']:.0f} min | {d['ms_per_tick']:.2f} |")
    print()
    print("| Detector | " + " | ".join(r["diagnosis"]["labels"]) + " |")
    print("|---|" + "---|" * len(r["diagnosis"]["labels"]))
    for n, d in r["detectors"].items():
        print(f"| {n} | " + " | ".join(f"{d['recall_by_kind'][k]:.0%} ({d['delay_median_by_kind'][k] or 0:.0f} min)"
                                      for k in r["diagnosis"]["labels"]) + " |")
    print()
    for n, d in r["detectors"].items():
        print(n, "false alarms", d["false_alarms"], "on hot spots", d["false_alarms_on_hotspots"])
    print(r["diagnosis"])
    print("test faults", sum(1 for _ in r["events"]["peer-mad"]))


if __name__ == "__main__":
    print("## full difficulty"); nav_table("runs/benchmark_d1.0.json")
    print("## medium"); nav_table("runs/benchmark_d0.5.json")
    print("## noise"); noise_table()
    print("## cluster"); cluster_table()
