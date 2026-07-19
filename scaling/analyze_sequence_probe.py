"""Analyze pre-registered multi-token sequence-entanglement artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    for key in ("reverse_logp", "forward_sequence_logp", "forward_first_token_logp"):
        if np.isnan(result[key]).any():
            raise ValueError(f"incomplete {key} in {path}")
    return result


def bh_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def interval(values):
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def per_animal_correlations(forward, reverse, mask):
    coefficients, p_values = [], []
    for animal_index in range(forward.shape[0]):
        result = stats.pearsonr(
            forward[animal_index, mask], reverse[mask, animal_index]
        )
        coefficients.append(result.statistic)
        p_values.append(result.pvalue)
    return np.asarray(coefficients), np.asarray(p_values)


def summarize_correlations(coefficients, p_values, animals, bootstrap_indices):
    q_values = bh_adjust(p_values)
    bootstrap = coefficients[bootstrap_indices].mean(axis=1)
    return {
        "mean_r": float(coefficients.mean()),
        "bootstrap_95_ci": interval(bootstrap),
        "positive_raw_p_lt_0.05": int(np.sum((coefficients > 0) & (p_values < 0.05))),
        "positive_bh_fdr_q_lt_0.05": int(
            np.sum((coefficients > 0) & (q_values < 0.05))
        ),
        "per_animal": {
            animal: {
                "r": float(coefficients[index]),
                "p": float(p_values[index]),
                "bh_q": float(q_values[index]),
            }
            for index, animal in enumerate(animals)
        },
    }


def within_width_standardize(values, widths):
    result = np.empty_like(values, dtype=np.float64)
    for width in sorted(set(widths.tolist())):
        mask = widths == width
        subset = values[..., mask]
        means = subset.mean(axis=-1, keepdims=True)
        scales = subset.std(axis=-1, keepdims=True)
        scales[scales == 0] = 1.0
        result[..., mask] = (subset - means) / scales
    return result


def analyze_model(label, path, bootstrap_indices):
    data = load_npz(path)
    animals = data["animals"].tolist()
    widths = data["number_widths"]
    token_counts = data["sequence_token_counts"].astype(np.float64)
    reverse = data["reverse_logp"]
    full = data["forward_sequence_logp"]
    first = data["forward_first_token_logp"]
    primary_mask = widths == 3
    if primary_mask.sum() != 1000:
        raise ValueError(
            f"primary requires all 1,000 width-3 strings; found {primary_mask.sum()} in {path}"
        )

    full_r, full_p = per_animal_correlations(full, reverse, primary_mask)
    first_r, first_p = per_animal_correlations(first, reverse, primary_mask)
    delta = full_r - first_r
    delta_bootstrap = delta[bootstrap_indices].mean(axis=1)

    mismatch = []
    for behavior_index in range(len(animals)):
        mismatch.append(
            np.mean(
                [
                    stats.pearsonr(
                        full[geometry_index, primary_mask],
                        reverse[primary_mask, behavior_index],
                    ).statistic
                    for geometry_index in range(len(animals))
                    if geometry_index != behavior_index
                ]
            )
        )
    mismatch = np.asarray(mismatch)
    specificity = full_r - mismatch
    specificity_bootstrap = specificity[bootstrap_indices].mean(axis=1)

    by_width = {}
    for width in sorted(set(widths.tolist())):
        mask = widths == width
        coefficients, p_values = per_animal_correlations(full, reverse, mask)
        by_width[str(width)] = summarize_correlations(
            coefficients, p_values, animals, bootstrap_indices
        )

    all_mask = np.ones(len(widths), dtype=bool)
    all_sum_r, all_sum_p = per_animal_correlations(full, reverse, all_mask)
    per_token = full / token_counts[None, :]
    all_mean_r, all_mean_p = per_animal_correlations(per_token, reverse, all_mask)
    standardized_full = within_width_standardize(full, widths)
    standardized_reverse = within_width_standardize(reverse.T, widths).T
    standardized_r, standardized_p = per_animal_correlations(
        standardized_full, standardized_reverse, all_mask
    )

    info = json.loads(str(data["metadata_json"].item()))
    unique_counts, count_frequencies = np.unique(
        data["sequence_token_counts"], return_counts=True
    )
    return {
        "label": label,
        "path": str(path),
        "model": info["model"],
        "n_animals": len(animals),
        "n_numbers": len(widths),
        "token_count_distribution": {
            str(int(key)): int(value)
            for key, value in zip(unique_counts, count_frequencies)
        },
        "trie_naive_max_abs_delta": info.get("trie_naive_max_abs_delta"),
        "primary_width_3_full_sequence": summarize_correlations(
            full_r, full_p, animals, bootstrap_indices
        ),
        "first_token_width_3": summarize_correlations(
            first_r, first_p, animals, bootstrap_indices
        ),
        "full_sequence_minus_first_token": {
            "mean_paired_delta_r": float(delta.mean()),
            "paired_bootstrap_95_ci": interval(delta_bootstrap),
            "animals_full_higher": int(np.sum(delta > 0)),
            "n_animals": len(animals),
        },
        "animal_specificity_control": {
            "mean_matched_r": float(full_r.mean()),
            "mean_mismatched_r": float(mismatch.mean()),
            "mean_matched_minus_mismatched": float(specificity.mean()),
            "paired_bootstrap_95_ci": interval(specificity_bootstrap),
            "matched_beats_mismatch": int(np.sum(specificity > 0)),
            "n_animals": len(animals),
            "per_animal": {
                animal: {
                    "matched_r": float(full_r[index]),
                    "mean_mismatched_r": float(mismatch[index]),
                    "delta": float(specificity[index]),
                }
                for index, animal in enumerate(animals)
            },
        },
        "by_width_full_sequence": by_width,
        "secondary_all_widths": {
            "sum_logp": summarize_correlations(
                all_sum_r, all_sum_p, animals, bootstrap_indices
            ),
            "mean_logp_per_target_token": summarize_correlations(
                all_mean_r, all_mean_p, animals, bootstrap_indices
            ),
            "within_width_standardized": summarize_correlations(
                standardized_r, standardized_p, animals, bootstrap_indices
            ),
        },
        "_raw_primary_r": full_r,
        "_animals": animals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", action="append", type=parse_artifact, required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompting/results/sequence_probe_summary.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    args = parser.parse_args()

    first_data = load_npz(args.artifact[0][1])
    n_animals = len(first_data["animals"])
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, n_animals, size=(args.bootstrap_resamples, n_animals)
    )
    results = {}
    raw = {}
    animal_order = None
    for label, path in args.artifact:
        result = analyze_model(label, path, bootstrap_indices)
        animals = result.pop("_animals")
        coefficients = result.pop("_raw_primary_r")
        if animal_order is None:
            animal_order = animals
        elif animals != animal_order:
            raise ValueError("animal order differs across artifacts")
        results[label] = result
        raw[label] = coefficients

    contrasts = {}
    labels = list(results)
    for smaller, larger in zip(labels, labels[1:]):
        delta = raw[larger] - raw[smaller]
        bootstrap = delta[bootstrap_indices].mean(axis=1)
        contrasts[f"{larger}-{smaller}"] = {
            "mean_paired_delta_r": float(delta.mean()),
            "paired_bootstrap_95_ci": interval(bootstrap),
            "animals_increasing": int(np.sum(delta > 0)),
            "n_animals": len(delta),
        }

    output = {
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "primary_estimand": (
            "mean across animals of corr(full sequence logp, reverse animal logp) "
            "over the 1,000 width-3 decimal strings"
        ),
        "models": results,
        "primary_scaling_contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    for label, result in results.items():
        primary = result["primary_width_3_full_sequence"]
        rescue = result["full_sequence_minus_first_token"]
        control = result["animal_specificity_control"]
        print(
            f"{label}: primary mean r={primary['mean_r']:+.3f} "
            f"(95% CI {primary['bootstrap_95_ci'][0]:+.3f} to "
            f"{primary['bootstrap_95_ci'][1]:+.3f}); "
            f"BH-FDR {primary['positive_bh_fdr_q_lt_0.05']}/{result['n_animals']}"
        )
        print(
            f"  full-minus-first={rescue['mean_paired_delta_r']:+.3f} "
            f"(95% CI {rescue['paired_bootstrap_95_ci'][0]:+.3f} to "
            f"{rescue['paired_bootstrap_95_ci'][1]:+.3f}); "
            f"specificity={control['mean_matched_minus_mismatched']:+.3f} "
            f"(95% CI {control['paired_bootstrap_95_ci'][0]:+.3f} to "
            f"{control['paired_bootstrap_95_ci'][1]:+.3f})"
        )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
