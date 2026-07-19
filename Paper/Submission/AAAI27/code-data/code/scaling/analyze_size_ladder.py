"""Summarize the existing Llama size-ladder entanglement artifacts.

The raw probe intentionally stays unchanged for comparability. This script adds
the pre-registered across-animal bootstrap interval and Benjamini-Hochberg
multiplicity control, then reports paired bootstrap differences between adjacent
model sizes as an explicitly exploratory scaling check.
"""

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = {
    "1B": "entanglement_1b.json",
    "3B": "entanglement_llama32_3b.json",
    "8B": "entanglement_8b.json",
}


def parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def bh_adjust(p_values):
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def percentile_interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("prompting/results"))
    parser.add_argument("--output", type=Path, default=Path("prompting/results/size_ladder_summary.json"))
    parser.add_argument(
        "--artifact",
        action="append",
        type=parse_artifact,
        help="override defaults with repeatable LABEL=PATH inputs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    args = parser.parse_args()

    loaded = {}
    animal_order = None
    sources = (
        dict(args.artifact)
        if args.artifact
        else {label: args.results_dir / filename for label, filename in MODELS.items()}
    )
    for label, path in sources.items():
        artifact = json.loads(path.read_text())
        results = artifact["results"]
        animals = [row["animal"] for row in results]
        if animal_order is None:
            animal_order = animals
        elif animals != animal_order:
            raise ValueError(f"Animal order differs for {label}; paired comparison is invalid")
        loaded[label] = artifact

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(animal_order), size=(args.bootstrap_resamples, len(animal_order)))

    summary = {}
    r_by_model = {}
    for label, artifact in loaded.items():
        results = artifact["results"]
        r = np.asarray([row["pearson_r"] for row in results])
        p = np.asarray([row["pearson_p"] for row in results])
        q = bh_adjust(p)
        medians = np.median(r[indices], axis=1)
        r_by_model[label] = r
        summary[label] = {
            "model": artifact["model"],
            "n_numbers": artifact["n_numbers"],
            "n_animals": len(results),
            "positive_p_lt_0.05": int(np.sum((r > 0) & (p < 0.05))),
            "positive_bh_fdr_q_lt_0.05": int(np.sum((r > 0) & (q < 0.05))),
            "median_pearson_r": float(np.median(r)),
            "median_pearson_r_bootstrap_95_ci": percentile_interval(medians),
            "pearson_r_range": [float(np.min(r)), float(np.max(r))],
            "animals": [
                {
                    "animal": row["animal"],
                    "pearson_r": float(row["pearson_r"]),
                    "pearson_p": float(row["pearson_p"]),
                    "pearson_bh_q": float(q[i]),
                }
                for i, row in enumerate(results)
            ],
        }

    adjacent_differences = {}
    model_order = list(r_by_model)
    for smaller, larger in zip(model_order, model_order[1:]):
        paired_delta = r_by_model[larger] - r_by_model[smaller]
        boot_mean_delta = np.mean(paired_delta[indices], axis=1)
        adjacent_differences[f"{larger}-{smaller}"] = {
            "estimand": "mean within-animal Pearson-r difference",
            "estimate": float(np.mean(paired_delta)),
            "paired_bootstrap_95_ci": percentile_interval(boot_mean_delta),
            "animals_increasing": int(np.sum(paired_delta > 0)),
            "n_animals": len(animal_order),
            "note": "Exploratory; added after the per-model summary was pre-registered.",
        }

    output = {
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "models": summary,
        "adjacent_differences": adjacent_differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    for label, row in summary.items():
        ci = row["median_pearson_r_bootstrap_95_ci"]
        print(
            f"{label}: {row['positive_p_lt_0.05']}/{row['n_animals']} raw, "
            f"{row['positive_bh_fdr_q_lt_0.05']}/{row['n_animals']} BH-FDR; "
            f"median r={row['median_pearson_r']:.3f} "
            f"(bootstrap 95% CI {ci[0]:.3f} to {ci[1]:.3f})"
        )
    for label, row in adjacent_differences.items():
        ci = row["paired_bootstrap_95_ci"]
        print(
            f"{label}: mean paired delta={row['estimate']:+.3f} "
            f"(bootstrap 95% CI {ci[0]:+.3f} to {ci[1]:+.3f}); "
            f"{row['animals_increasing']}/{row['n_animals']} animals increase"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
