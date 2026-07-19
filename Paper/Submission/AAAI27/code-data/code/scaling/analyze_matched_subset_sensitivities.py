"""Run saved-array sensitivities for number-set and animal-contrast concerns."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import analyze_causal_patch as causal
import analyze_geometry_scaling as geometry
import analyze_layerwise_probe as layerwise


def interval(values):
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def subset_indices(data, numbers):
    lookup = {
        value: index
        for index, value in enumerate(data["number_strs"].astype(str).tolist())
    }
    missing = [value for value in numbers if value not in lookup]
    if missing:
        raise ValueError(f"subset numbers absent from artifact: {missing[:5]}")
    return np.asarray([lookup[value] for value in numbers], dtype=np.int64)


def geometry_subset(path, numbers):
    data = geometry.load_npz(path)
    indices = subset_indices(data, numbers)
    matrix = geometry.geometry_matrices(data, "mean_subtokens")["raw_cos"][:, indices]
    reverse = data["reverse_logp"][indices].T
    matched = geometry.correlations(matrix, reverse)[0]
    mismatch = np.asarray(
        [
            np.mean(
                [
                    stats.pearsonr(matrix[other], reverse[animal]).statistic
                    for other in range(len(reverse))
                    if other != animal
                ]
            )
            for animal in range(len(reverse))
        ]
    )
    return data["animals"].tolist(), matched, matched - mismatch


def readout_subset(path, numbers):
    data, _ = layerwise.load_npz(path)
    indices = subset_indices(data, numbers)
    curves, _ = layerwise.correlation_curves(
        data["assistant_scores"][indices].astype(np.float64),
        data["reverse_logp"][indices].astype(np.float64),
    )
    auc = np.trapezoid(
        curves, data["relative_depth"].astype(np.float64), axis=1
    )
    return data["animals"].tolist(), auc


def paired_summary(smaller, larger, bootstrap_indices):
    delta = larger - smaller
    sampled = delta[bootstrap_indices].mean(axis=1)
    return {
        "8B_mean": float(smaller.mean()),
        "70B_mean": float(larger.mean()),
        "paired_70B_minus_8B": float(delta.mean()),
        "paired_animal_bootstrap_95_ci": interval(sampled),
        "animals_increasing": int(np.sum(delta > 0)),
        "animals_decreasing": int(np.sum(delta < 0)),
        "n_animals": int(len(delta)),
    }


def raw_logit_crossed_sensitivity(path8, path70, resamples, seed, chunk_size):
    artifacts = []
    signature = None
    for path in (path8, path70):
        arrays, info = causal.load_artifact(path)
        current = (
            arrays["animals"].tolist(),
            arrays["unique_numbers"].tolist(),
            arrays["donor_indices"].tolist(),
            arrays["recipient_indices"].tolist(),
        )
        if signature is None:
            signature = current
        elif current != signature:
            raise ValueError("causal artifacts do not share the paired design")
        artifacts.append((arrays, causal.point_analysis(arrays, info)))

    point_delta = (
        artifacts[1][1]["corrected_auc_raw_logits"]
        - artifacts[0][1]["corrected_auc_raw_logits"]
    )
    pair_count = len(artifacts[0][0]["donor_indices"]) // 2
    n_animals = len(artifacts[0][0]["animals"])
    rng = np.random.default_rng(seed)
    bootstrap_means = [np.empty(resamples), np.empty(resamples)]

    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        sampled_clusters = rng.integers(
            0, pair_count, size=(size, pair_count)
        )
        observations = np.concatenate(
            [sampled_clusters, sampled_clusters + pair_count], axis=1
        )
        sampled_animals = rng.integers(
            0, n_animals, size=(size, n_animals)
        )
        for model_index, (arrays, analysis) in enumerate(artifacts):
            clean = arrays["clean_logits"].astype(np.float64)
            donor = clean[arrays["donor_indices"]]
            recipient = clean[arrays["recipient_indices"]]
            outcome = arrays["patched_logits"].astype(np.float64)
            beta, _, _ = causal.selected_donor_coefficients(
                donor, recipient, outcome, observations
            )
            zeros = np.zeros((size, 1, n_animals), dtype=np.float64)
            auc = np.trapezoid(
                np.concatenate([zeros, beta], axis=1),
                np.concatenate([[0.0], analysis["depths"]]),
                axis=1,
            ) / analysis["depths"][-1]
            bootstrap_means[model_index][start : start + size] = np.take_along_axis(
                auc, sampled_animals, axis=1
            ).mean(axis=1)

    bootstrap_delta = bootstrap_means[1] - bootstrap_means[0]
    return {
        "paired_70B_minus_8B": float(point_delta.mean()),
        "crossed_bootstrap_95_ci": interval(bootstrap_delta),
        "animals_increasing": int(np.sum(point_delta > 0)),
        "n_animals": int(len(point_delta)),
        "bootstrap_draws_above_zero": int(np.sum(bootstrap_delta > 0)),
        "bootstrap_resamples": int(resamples),
        "bootstrap_seed": int(seed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-8b", type=Path, required=True)
    parser.add_argument("--geometry-70b", type=Path, required=True)
    parser.add_argument("--layerwise-8b", type=Path, required=True)
    parser.add_argument("--layerwise-70b", type=Path, required=True)
    parser.add_argument("--causal-8b", type=Path, required=True)
    parser.add_argument("--causal-70b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--animal-bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--crossed-bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    with np.load(args.causal_8b, allow_pickle=False) as data:
        numbers = data["unique_numbers"].astype(str).tolist()

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, 18, size=(args.animal_bootstrap_resamples, 18)
    )
    animals8, geometry8, specificity8 = geometry_subset(
        args.geometry_8b, numbers
    )
    animals70, geometry70, specificity70 = geometry_subset(
        args.geometry_70b, numbers
    )
    readout_animals8, readout8 = readout_subset(args.layerwise_8b, numbers)
    readout_animals70, readout70 = readout_subset(args.layerwise_70b, numbers)
    if not (animals8 == animals70 == readout_animals8 == readout_animals70):
        raise ValueError("animal order differs across artifacts")

    output = {
        "status": "post_hoc_saved_array_sensitivity",
        "purpose": (
            "Restrict full-universe geometry and observational readout to the "
            "exact 256 outcome-blind numbers used by the causal intervention."
        ),
        "subset_number_count": len(numbers),
        # Keep provenance stable across machines and extraction directories.
        "subset_source": f"data/raw/{args.causal_8b.name}",
        "animal_bootstrap_seed": args.seed,
        "animal_bootstrap_resamples": args.animal_bootstrap_resamples,
        "exact_256_number_subset": {
            "static_geometry_vs_behavior": paired_summary(
                geometry8, geometry70, bootstrap_indices
            ),
            "geometry_specificity": paired_summary(
                specificity8, specificity70, bootstrap_indices
            ),
            "observational_assistant_readout_auc": paired_summary(
                readout8, readout70, bootstrap_indices
            ),
        },
        "raw_target_logit_causal_sensitivity": raw_logit_crossed_sensitivity(
            args.causal_8b,
            args.causal_70b,
            args.crossed_bootstrap_resamples,
            args.seed,
            args.chunk_size,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
