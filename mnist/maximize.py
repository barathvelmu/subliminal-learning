"""Select a higher-accuracy student configuration with a clean test split.

Candidate settings combine the strongest validation-sweep choices while fixing
``n_digits=10`` and ``n_aux=3``. Selection uses validation data; the chosen
configuration is then evaluated once on the test set.
"""

import json
import numpy as np
from dataclasses import asdict, replace
from experiment import Config, load_splits, run_seed

VAL_SEEDS = (0, 1, 2)
TEST_SEEDS = (0, 1, 2, 3, 4)

# Candidates combine higher-validation settings from the individual sweeps.
CANDIDATES = [
    dict(loss="mse", width=128, epochs_distill=20),
    dict(loss="mse", width=128, epochs_distill=40),
    dict(loss="mse", width=256, epochs_distill=20),
    dict(loss="mse", width=128, epochs_distill=20, epochs_teacher=10),
    dict(loss="mse", width=128, epochs_distill=40, lr=1e-3),
    dict(loss="mse", width=64, epochs_distill=40),
]


def mean_student_aux(cfg, seeds, splits):
    per = {s: run_seed(cfg, s, splits) for s in seeds}
    vals = np.array([np.mean(per[s]["student_aux"]) for s in seeds])
    return float(vals.mean()), float(vals.std(ddof=1)), per


def main():
    splits = load_splits()
    base = Config()
    print("=== validation search ===")
    results = []
    for i, ov in enumerate(CANDIDATES):
        cfg = replace(base, seeds=VAL_SEEDS, eval_split="val", tag=f"max_cand{i}", **ov)
        m, s, _ = mean_student_aux(cfg, VAL_SEEDS, splits)
        results.append((m, s, ov))
        print(f"cand{i}: student_aux(val)={m:.3f}±{s:.3f}  {ov}")

    best_m, best_s, best_ov = max(results, key=lambda r: r[0])
    print(f"\nBest on val: {best_ov}  (val student_aux={best_m:.3f})")

    print("=== final TEST run (chosen config, 5 seeds) ===")
    cfg = replace(
        base, seeds=TEST_SEEDS, eval_split="test", tag="max_best_test", **best_ov
    )
    m, s, per = mean_student_aux(cfg, TEST_SEEDS, splits)
    print(f"TEST student_aux = {m:.3f} ± {s:.3f}  (baseline was 0.463 ± 0.022)")

    out = {
        "chosen_config": asdict(cfg),
        "val_search": [(r[0], r[1], r[2]) for r in results],
        "test_student_aux_mean": m,
        "test_student_aux_std": s,
    }
    json.dump(out, open("results/maximize.json", "w"), indent=2)
    print("saved results/maximize.json")


if __name__ == "__main__":
    main()
