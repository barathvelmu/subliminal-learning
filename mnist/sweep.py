"""
Run all MNIST sweeps in one process (loads MNIST once).

Each sweep varies ONE knob, rest at baseline defaults. All on the VALIDATION
split (test stays clean for final reporting). 3 seeds per point.

Writes:
  results/sweep_<name>_<value>.json   (full per-seed detail)
  results/sweeps_summary.csv          (tidy: sweep,value,condition,mean,std,n_seeds)
"""

import csv
import json
import os
from dataclasses import asdict, replace

import numpy as np

from experiment import Config, load_splits, run_seed

SEEDS = (0, 1, 2)
EVAL = "val"
CONDITIONS = [
    "reference",
    "teacher",
    "student_aux",
    "student_all",
    "xmodel_aux",
    "xmodel_all",
    "cos_aux_to_teacher",
]

# (sweep_name, field, [values])
SWEEPS = [
    ("loss", "loss", ["kl", "mse", "kl_temp"]),
    ("noise", "noise", ["uniform", "gaussian", "train_images"]),
    ("distill_epochs", "epochs_distill", [0, 1, 2, 3, 5, 10]),
    ("init_scale", "init_scale", [0.5, 1.0, 2.0, 4.0]),
    ("width", "width", [64, 128, 256, 512, 1024, 2048]),
]


def aggregate(per_seed, seeds):
    rows = {}
    for c in CONDITIONS:
        sm = np.array([np.mean(per_seed[s][c]) for s in seeds])
        rows[c] = {
            "mean": float(sm.mean()),
            "std": float(sm.std(ddof=1)) if len(sm) > 1 else 0.0,
            "n_seeds": len(sm),
        }
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    splits = load_splits()
    base = Config(seeds=SEEDS, eval_split=EVAL)
    csv_rows = []

    for sweep_name, field, values in SWEEPS:
        for val in values:
            cfg = replace(base, **{field: val}, tag=f"sweep_{sweep_name}_{val}")
            per_seed = {s: run_seed(cfg, s, splits) for s in cfg.seeds}
            rows = aggregate(per_seed, cfg.seeds)
            with open(f"results/sweep_{sweep_name}_{val}.json", "w") as f:
                json.dump(
                    {"config": asdict(cfg), "summary": rows, "per_seed": per_seed}, f
                )
            sa = rows["student_aux"]
            print(
                f"[{sweep_name}={val}] student_aux={sa['mean']:.3f}±{sa['std']:.3f} "
                f"xmodel_aux={rows['xmodel_aux']['mean']:.3f} "
                f"cos={rows['cos_aux_to_teacher']['mean']:.3f}"
            )
            for c in CONDITIONS:
                csv_rows.append(
                    {
                        "sweep": sweep_name,
                        "value": val,
                        "condition": c,
                        "mean": rows[c]["mean"],
                        "std": rows[c]["std"],
                        "n_seeds": rows[c]["n_seeds"],
                    }
                )

    with open("results/sweeps_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sweep", "value", "condition", "mean", "std", "n_seeds"]
        )
        w.writeheader()
        w.writerows(csv_rows)
    print("Saved results/sweeps_summary.csv")


if __name__ == "__main__":
    main()
