#!/usr/bin/env python3
"""Compute paired intervention effect from master_results.csv using only stdlib."""
import csv
import math
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "multiview", "master_results.csv")

ctrl = {}   # (seed, setting) -> {MinSim, collapse}
intv = {}

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        key = (row["seed"], row["setting"])
        entry = {"MinSim": float(row["MinSim"]), "collapse": int(row["collapse"])}
        if row["method"] == "control":
            ctrl[key] = entry
        elif row["method"] == "intervention":
            intv[key] = entry

# Group by setting
settings = sorted(set(k[1] for k in ctrl))
print(f"{'setting':10s} | {'avg_delta':>10s} | {'improved_ratio':>15s} | {'collapse_reduction':>20s}")
print("-" * 65)

for s in settings:
    deltas = []
    n_improved = 0
    n_ctrl_collapsed = 0
    n_intv_collapsed = 0

    for key in sorted(ctrl):
        if key[1] != s or key not in intv:
            continue
        d = intv[key]["MinSim"] - ctrl[key]["MinSim"]
        deltas.append(d)
        if d > 0:
            n_improved += 1
        n_ctrl_collapsed += ctrl[key]["collapse"]
        n_intv_collapsed += intv[key]["collapse"]

    n = len(deltas)
    avg_d = sum(deltas) / n
    imp_ratio = n_improved / n * 100
    if n_ctrl_collapsed > 0:
        col_red = (n_ctrl_collapsed - n_intv_collapsed) / n_ctrl_collapsed * 100
        col_str = f"{n_ctrl_collapsed}->{n_intv_collapsed} ({col_red:+.1f}%)"
    else:
        col_str = "0->0 (N/A)"

    print(f"{s:10s} | {avg_d:+10.4f} | {imp_ratio:14.1f}% | {col_str:>20s}")
