"""Analyze geometry-behavior alignment from resumable full-probe artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

PAPER_PAIRS = {"owl": "087", "eagle": "747", "sea turtle": "321"}
METRICS = ("raw_cos", "cen_cos", "raw_dot", "cen_dot")
DIRECTIONS = ("forward", "reverse")
VARIANTS = ("mean_subtokens", "first_token")


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def interval(values):
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def correlations(x_rows, y_rows):
    coefficients, p_values = [], []
    for x, y in zip(x_rows, y_rows):
        result = stats.pearsonr(x, y)
        coefficients.append(result.statistic)
        p_values.append(result.pvalue)
    return np.asarray(coefficients), np.asarray(p_values)


def geometry_matrices(data, variant):
    number_rows = data["number_unembedding"].astype(np.float64)
    mean = data["unembedding_mean"].astype(np.float64)
    animal_token_rows = data["animal_unembedding_tokens"].astype(np.float64)
    mask = data["animal_token_mask"]
    if variant == "first_token":
        animal_rows = animal_token_rows[:, 0]
    else:
        animal_rows = np.stack(
            [animal_token_rows[index, mask[index]].mean(axis=0) for index in range(len(mask))]
        )
    centered_numbers = number_rows - mean
    centered_animals = animal_rows - mean

    def cosine(left, right):
        numerator = left @ right.T
        denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
        return numerator / denominator

    return {
        "raw_cos": cosine(animal_rows, number_rows),
        "cen_cos": cosine(centered_animals, centered_numbers),
        "raw_dot": animal_rows @ number_rows.T,
        "cen_dot": centered_animals @ centered_numbers.T,
    }


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        loaded = {key: data[key] for key in data.files}
    if not bool(loaded["geometry_complete"].item()):
        raise ValueError(f"Geometry is incomplete in {path}")
    if np.isnan(loaded["reverse_logp"]).any() or np.isnan(loaded["forward_logp"]).any():
        raise ValueError(f"Behavioral matrices are incomplete in {path}")
    return loaded


def parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def analyze_model(label, path, bootstrap_indices):
    data = load_npz(path)
    animals = data["animals"].tolist()
    numbers = data["number_strs"].tolist()
    forward = data["forward_logp"]
    reverse = data["reverse_logp"].T
    behavior = {"forward": forward, "reverse": reverse}
    variants = {}
    raw_values = {}

    for variant in VARIANTS:
        matrices = geometry_matrices(data, variant)
        variant_result = {}
        for metric in METRICS:
            metric_result = {}
            for direction in DIRECTIONS:
                coefficients, p_values = correlations(matrices[metric], behavior[direction])
                bootstrap_means = coefficients[bootstrap_indices].mean(axis=1)
                q_values = bh_adjust(p_values)
                metric_result[direction] = {
                    "mean_r": float(coefficients.mean()),
                    "bootstrap_95_ci": interval(bootstrap_means),
                    "positive_raw_p_lt_0.05": int(np.sum((coefficients > 0) & (p_values < 0.05))),
                    "positive_bh_fdr_q_lt_0.05": int(np.sum((coefficients > 0) & (q_values < 0.05))),
                    "per_animal": {
                        animal: {
                            "r": float(coefficients[index]),
                            "p": float(p_values[index]),
                            "bh_q": float(q_values[index]),
                        }
                        for index, animal in enumerate(animals)
                    },
                }
                raw_values[(variant, metric, direction)] = coefficients
            variant_result[metric] = metric_result
        variants[variant] = variant_result

    primary_geometry = geometry_matrices(data, "mean_subtokens")["raw_cos"]
    matched, mismatch_means = [], []
    for behavior_index in range(len(animals)):
        matched.append(stats.pearsonr(primary_geometry[behavior_index], reverse[behavior_index]).statistic)
        mismatch_means.append(
            np.mean(
                [
                    stats.pearsonr(primary_geometry[geometry_index], reverse[behavior_index]).statistic
                    for geometry_index in range(len(animals))
                    if geometry_index != behavior_index
                ]
            )
        )
    matched = np.asarray(matched)
    mismatch_means = np.asarray(mismatch_means)
    control_delta = matched - mismatch_means
    control_bootstrap = control_delta[bootstrap_indices].mean(axis=1)
    raw_values[("control", "matched_minus_mismatched", "reverse")] = control_delta

    raw_reverse = raw_values[("mean_subtokens", "raw_cos", "reverse")]
    centered_reverse = raw_values[("mean_subtokens", "cen_cos", "reverse")]
    raw_minus_centered = raw_reverse - centered_reverse
    raw_minus_centered_bootstrap = raw_minus_centered[bootstrap_indices].mean(axis=1)

    pair_ranks = {}
    for animal, pair in PAPER_PAIRS.items():
        if animal not in animals:
            continue
        candidate = pair if pair in numbers else str(int(pair))
        if candidate not in numbers:
            pair_ranks[animal] = None
            continue
        animal_index = animals.index(animal)
        number_index = numbers.index(candidate)
        values = primary_geometry[animal_index]
        pair_ranks[animal] = {
            "pair": pair,
            "rank": int(np.sum(values > values[number_index])) + 1,
            "n_numbers": len(numbers),
            "raw_cos": float(values[number_index]),
        }

    metadata = json.loads(str(data["metadata_json"].item()))
    return {
        "label": label,
        "path": str(path),
        "model": metadata["model"],
        "n_animals": len(animals),
        "n_numbers": len(numbers),
        "primary": variants["mean_subtokens"]["raw_cos"]["reverse"],
        "animal_specificity_control": {
            "mean_matched_r": float(matched.mean()),
            "mean_mismatched_r": float(mismatch_means.mean()),
            "mean_matched_minus_mismatched": float(control_delta.mean()),
            "paired_bootstrap_95_ci": interval(control_bootstrap),
            "matched_beats_mismatch": int(np.sum(control_delta > 0)),
            "n_animals": len(animals),
            "per_animal": {
                animal: {
                    "matched_r": float(matched[index]),
                    "mean_mismatched_r": float(mismatch_means[index]),
                    "delta": float(control_delta[index]),
                }
                for index, animal in enumerate(animals)
            },
        },
        "secondary_raw_minus_centered_reverse": {
            "mean_paired_delta_r": float(raw_minus_centered.mean()),
            "paired_bootstrap_95_ci": interval(raw_minus_centered_bootstrap),
            "animals_raw_higher": int(np.sum(raw_minus_centered > 0)),
            "n_animals": len(animals),
        },
        "paper_pair_geometry_ranks": pair_ranks,
        "variants": variants,
    }, raw_values, animals


def regression_against_old(new_result, old_path):
    old = json.loads(old_path.read_text())
    differences = {}
    for metric in METRICS:
        differences[metric] = {}
        for direction in DIRECTIONS:
            new_value = new_result["variants"]["mean_subtokens"][metric][direction]["mean_r"]
            old_value = old["summary"][metric][f"{direction}_mean"]
            differences[metric][direction] = {
                "new": new_value,
                "old": old_value,
                "absolute_difference": abs(new_value - old_value),
            }
    maximum = max(
        item[direction]["absolute_difference"]
        for item in differences.values()
        for direction in DIRECTIONS
    )
    return {"maximum_absolute_difference": maximum, "details": differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", type=parse_artifact, required=True)
    parser.add_argument("--output", type=Path, default=Path("prompting/results/geometry_scaling_summary.json"))
    parser.add_argument("--old-1b", type=Path, default=Path("prompting/results/metrics_1b.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    all_results, all_raw, animal_order = {}, {}, None
    bootstrap_indices = None
    for label, path in args.artifact:
        data = load_npz(path)
        n_animals = len(data["animals"])
        if bootstrap_indices is None:
            bootstrap_indices = rng.integers(
                0, n_animals, size=(args.bootstrap_resamples, n_animals)
            )
        result, raw, animals = analyze_model(label, path, bootstrap_indices)
        if animal_order is None:
            animal_order = animals
        elif animals != animal_order:
            raise ValueError("Animal order differs across model artifacts")
        all_results[label] = result
        all_raw[label] = raw

    scaling_contrasts = {}
    control_scaling_contrasts = {}
    for smaller, larger in (
        ("1B", "3B"),
        ("3B", "8B"),
        ("8B-MPS", "8B-CUDA"),
        ("8B-CUDA", "70B"),
    ):
        if smaller not in all_raw or larger not in all_raw:
            continue
        key = ("mean_subtokens", "raw_cos", "reverse")
        delta = all_raw[larger][key] - all_raw[smaller][key]
        bootstrap = delta[bootstrap_indices].mean(axis=1)
        scaling_contrasts[f"{larger}-{smaller}"] = {
            "mean_paired_delta_r": float(delta.mean()),
            "paired_bootstrap_95_ci": interval(bootstrap),
            "animals_increasing": int(np.sum(delta > 0)),
            "n_animals": len(delta),
        }
        control_key = ("control", "matched_minus_mismatched", "reverse")
        control_delta = all_raw[larger][control_key] - all_raw[smaller][control_key]
        control_bootstrap = control_delta[bootstrap_indices].mean(axis=1)
        control_scaling_contrasts[f"{larger}-{smaller}"] = {
            "mean_paired_delta_r": float(control_delta.mean()),
            "paired_bootstrap_95_ci": interval(control_bootstrap),
            "animals_increasing": int(np.sum(control_delta > 0)),
            "n_animals": len(control_delta),
            "note": "Exploratory follow-up; not the pre-registered primary scaling contrast.",
        }

    regression = None
    if "1B" in all_results and args.old_1b.exists():
        regression = regression_against_old(all_results["1B"], args.old_1b)

    output = {
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "primary_estimand": "mean across animals of corr(raw mean-subtoken cosine, reverse log-probability)",
        "models": all_results,
        "primary_scaling_contrasts": scaling_contrasts,
        "exploratory_control_scaling_contrasts": control_scaling_contrasts,
        "one_b_regression_against_original_metrics": regression,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    for label, result in all_results.items():
        primary = result["primary"]
        control = result["animal_specificity_control"]
        print(
            f"{label}: primary mean r={primary['mean_r']:+.3f} "
            f"(95% CI {primary['bootstrap_95_ci'][0]:+.3f} to {primary['bootstrap_95_ci'][1]:+.3f}); "
            f"BH-FDR {primary['positive_bh_fdr_q_lt_0.05']}/{result['n_animals']}"
        )
        print(
            f"  animal-specific control delta={control['mean_matched_minus_mismatched']:+.3f} "
            f"(95% CI {control['paired_bootstrap_95_ci'][0]:+.3f} to "
            f"{control['paired_bootstrap_95_ci'][1]:+.3f})"
        )
    for label, result in scaling_contrasts.items():
        ci = result["paired_bootstrap_95_ci"]
        print(f"{label}: paired delta={result['mean_paired_delta_r']:+.3f} (95% CI {ci[0]:+.3f} to {ci[1]:+.3f})")
    for label, result in control_scaling_contrasts.items():
        ci = result["paired_bootstrap_95_ci"]
        print(
            f"{label} animal-specific control (exploratory): delta={result['mean_paired_delta_r']:+.3f} "
            f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f})"
        )
    if regression is not None:
        print(f"1B old-result regression max |delta|={regression['maximum_absolute_difference']:.3e}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
