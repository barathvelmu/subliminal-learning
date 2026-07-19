"""Analyze pre-registered tuned-logit-lens trajectories across model depth."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


POSITIONS = ("assistant", "number_mean", "number_last")


def parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    info = json.loads(str(result["metadata_json"].item()))
    if info.get("stage") != "complete":
        raise ValueError(f"incomplete artifact stage in {path}: {info.get('stage')}")
    for key in (
        "assistant_scores",
        "number_mean_scores",
        "number_last_scores",
        "reverse_logp",
    ):
        if np.isnan(result[key]).any():
            raise ValueError(f"incomplete {key} in {path}")
    return result, info


def interval(values):
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def safe_pearson(x, y):
    """Return zero readability for a constant predictor, plus a degeneracy flag."""
    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return 0.0, True
    return float(stats.pearsonr(x, y).statistic), False


def correlation_curves(scores, behavior):
    n_numbers, n_layers, n_animals = scores.shape
    if behavior.shape != (n_numbers, n_animals):
        raise ValueError("behavior and score dimensions disagree")
    curves = np.empty((n_animals, n_layers), dtype=np.float64)
    degenerate = np.zeros((n_animals, n_layers), dtype=bool)
    for animal_index in range(n_animals):
        for layer_index in range(n_layers):
            curves[animal_index, layer_index], degenerate[animal_index, layer_index] = (
                safe_pearson(
                    scores[:, layer_index, animal_index],
                    behavior[:, animal_index],
                )
            )
    return curves, degenerate


def specificity_inputs(scores, behavior):
    n_animals = scores.shape[2]
    score_contrast = (
        n_animals * scores - scores.sum(axis=2, keepdims=True)
    ) / (n_animals - 1)
    behavior_contrast = (
        n_animals * behavior - behavior.sum(axis=1, keepdims=True)
    ) / (n_animals - 1)
    return score_contrast, behavior_contrast


def curve_summary(curves, degenerate, relative_depth, animals, bootstrap_indices):
    mean_curve = curves.mean(axis=0)
    curve_ci = []
    for layer_index in range(curves.shape[1]):
        sampled = curves[:, layer_index][bootstrap_indices].mean(axis=1)
        curve_ci.append(interval(sampled))

    per_animal_auc = np.trapezoid(curves, relative_depth, axis=1)
    auc_bootstrap = per_animal_auc[bootstrap_indices].mean(axis=1)
    final_mean = float(mean_curve[-1])
    threshold = 0.5 * final_mean
    half_index = None
    if final_mean > 0:
        candidates = np.flatnonzero(mean_curve >= threshold)
        if len(candidates):
            half_index = int(candidates[0])
    leave_one_out_auc = [
        float(np.delete(per_animal_auc, index).mean())
        for index in range(len(animals))
    ]
    return {
        "mean_r_by_layer": mean_curve.tolist(),
        "bootstrap_95_ci_by_layer": curve_ci,
        "absolute_layer_index": list(range(curves.shape[1])),
        "relative_depth": relative_depth.tolist(),
        "final_layer_mean_r": final_mean,
        "auc_mean": float(per_animal_auc.mean()),
        "auc_bootstrap_95_ci": interval(auc_bootstrap),
        "half_final_threshold_r": float(threshold),
        "earliest_half_final_layer": half_index,
        "earliest_half_final_relative_depth": (
            None if half_index is None else float(relative_depth[half_index])
        ),
        "degenerate_cell_count": int(degenerate.sum()),
        "degenerate_cells_by_layer": degenerate.sum(axis=0).astype(int).tolist(),
        "leave_one_animal_out_auc_range": [
            float(min(leave_one_out_auc)), float(max(leave_one_out_auc))
        ],
        "per_animal": {
            animal: {
                "r_by_layer": curves[index].tolist(),
                "auc": float(per_animal_auc[index]),
                "leave_this_animal_out_auc": leave_one_out_auc[index],
            }
            for index, animal in enumerate(animals)
        },
        "_per_animal_auc": per_animal_auc,
    }


def analyze_model(label, path, bootstrap_indices):
    data, info = load_npz(path)
    animals = data["animals"].tolist()
    relative_depth = data["relative_depth"].astype(np.float64)
    behavior = data["reverse_logp"].astype(np.float64)
    results = {}
    raw_auc = {}
    raw_specific_auc = {}
    for position in POSITIONS:
        scores = data[f"{position}_scores"].astype(np.float64)
        curves, degenerate = correlation_curves(scores, behavior)
        summary = curve_summary(
            curves, degenerate, relative_depth, animals, bootstrap_indices
        )
        raw_auc[position] = summary.pop("_per_animal_auc")

        contrast_scores, contrast_behavior = specificity_inputs(scores, behavior)
        contrast_curves, contrast_degenerate = correlation_curves(
            contrast_scores, contrast_behavior
        )
        contrast_summary = curve_summary(
            contrast_curves,
            contrast_degenerate,
            relative_depth,
            animals,
            bootstrap_indices,
        )
        raw_specific_auc[position] = contrast_summary.pop("_per_animal_auc")
        summary["animal_specific_contrast"] = contrast_summary
        results[position] = summary

    return {
        "label": label,
        "path": str(path),
        "model": info["model"],
        "n_numbers": int(info["n_numbers"]),
        "n_animals": len(animals),
        "transformer_block_count": int(info["transformer_block_count"]),
        "layer_count_including_embedding": int(
            info["layer_count_including_embedding"]
        ),
        "final_logit_max_abs_delta": float(info["final_logit_max_abs_delta"]),
        "positions": results,
        "_animals": animals,
        "_raw_auc": raw_auc,
        "_raw_specific_auc": raw_specific_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", type=parse_artifact, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompting/results/layerwise_probe_summary.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    args = parser.parse_args()

    first, _ = load_npz(args.artifact[0][1])
    n_animals = len(first["animals"])
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, n_animals, size=(args.bootstrap_resamples, n_animals)
    )

    models = {}
    raw_auc = {}
    raw_specific_auc = {}
    animal_order = None
    for label, path in args.artifact:
        result = analyze_model(label, path, bootstrap_indices)
        animals = result.pop("_animals")
        if animal_order is None:
            animal_order = animals
        elif animals != animal_order:
            raise ValueError("animal order differs across artifacts")
        raw_auc[label] = result.pop("_raw_auc")
        raw_specific_auc[label] = result.pop("_raw_specific_auc")
        models[label] = result

    contrasts = {}
    labels = list(models)
    for smaller, larger in zip(labels, labels[1:]):
        position_results = {}
        for position in POSITIONS:
            delta = raw_auc[larger][position] - raw_auc[smaller][position]
            sampled = delta[bootstrap_indices].mean(axis=1)
            position_results[position] = {
                "mean_paired_delta_auc": float(delta.mean()),
                "paired_bootstrap_95_ci": interval(sampled),
                "animals_increasing": int(np.sum(delta > 0)),
                "n_animals": len(delta),
            }
            specific_delta = (
                raw_specific_auc[larger][position]
                - raw_specific_auc[smaller][position]
            )
            specific_sampled = specific_delta[bootstrap_indices].mean(axis=1)
            position_results[position]["animal_specific_contrast"] = {
                "mean_paired_delta_auc": float(specific_delta.mean()),
                "paired_bootstrap_95_ci": interval(specific_sampled),
                "animals_increasing": int(np.sum(specific_delta > 0)),
                "n_animals": len(specific_delta),
            }
        contrasts[f"{larger}-{smaller}"] = position_results

    output = {
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "constant_predictor_rule": (
            "A constant layer/animal readout has zero linear readability and is "
            "assigned r=0; degenerate cells are counted explicitly."
        ),
        "models": models,
        "paired_auc_contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    for label, result in models.items():
        print(label)
        for position in POSITIONS:
            summary = result["positions"][position]
            print(
                f"  {position}: final r={summary['final_layer_mean_r']:+.3f}; "
                f"AUC={summary['auc_mean']:+.3f} "
                f"[{summary['auc_bootstrap_95_ci'][0]:+.3f}, "
                f"{summary['auc_bootstrap_95_ci'][1]:+.3f}]; "
                f"half-depth={summary['earliest_half_final_relative_depth']}"
            )
    for label, positions in contrasts.items():
        assistant = positions["assistant"]
        print(
            f"{label} assistant AUC delta={assistant['mean_paired_delta_auc']:+.3f} "
            f"[{assistant['paired_bootstrap_95_ci'][0]:+.3f}, "
            f"{assistant['paired_bootstrap_95_ci'][1]:+.3f}]"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
