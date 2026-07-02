"""Rebuild results/sweeps_summary.csv from per-point JSONs (robust to a missing
point, e.g. width=2048 which OOMs at N=25). Skips files that don't exist."""
import csv
import json
import os

from sweep import SWEEPS, CONDITIONS

rows = []
for name, _field, values in SWEEPS:
    for v in values:
        fp = f"results/sweep_{name}_{v}.json"
        if not os.path.exists(fp):
            print(f"skip missing: {fp}")
            continue
        s = json.load(open(fp))["summary"]
        for c in CONDITIONS:
            rows.append({"sweep": name, "value": v, "condition": c,
                         "mean": s[c]["mean"], "std": s[c]["std"], "n_seeds": s[c]["n_seeds"]})

with open("results/sweeps_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sweep", "value", "condition", "mean", "std", "n_seeds"])
    w.writeheader(); w.writerows(rows)
print(f"wrote results/sweeps_summary.csv ({len(rows)} rows)")
