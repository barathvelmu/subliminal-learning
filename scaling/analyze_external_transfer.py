"""Outcome-blinded S8 validation against released student-transfer rates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ANIMALS = [
    "dolphin",
    "dog",
    "jellyfish",
    "tiger",
    "octopus",
    "elephant",
    "fox",
    "cat",
    "mouse",
    "lion",
    "hawk",
    "platypus",
    "wolf",
    "pangolin",
    "falcon",
    "whale",
]
PREDICTORS = ("geometry", "readout", "causal", "steering")
IN_HOUSE = ("geometry", "readout", "causal")


def load_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def row_correlations(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    result = np.full(len(left), np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def bootstrap_spearman(x, y, indices):
    ranked_x = stats.rankdata(x[indices], axis=1)
    ranked_y = stats.rankdata(y[indices], axis=1)
    return row_correlations(ranked_x, ranked_y)


def permutation_p(x, y, observed, rng, resamples, chunk_size=10_000):
    ranked_x = stats.rankdata(x).astype(np.float64)
    ranked_x -= ranked_x.mean()
    denominator_x = np.sqrt(np.sum(ranked_x * ranked_x))
    exceed = 0
    finite = 0
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        permutations = np.asarray([rng.permutation(len(y)) for _ in range(size)])
        ranked_y = stats.rankdata(y[permutations], axis=1)
        ranked_y -= ranked_y.mean(axis=1, keepdims=True)
        denominator = denominator_x * np.sqrt(np.sum(ranked_y * ranked_y, axis=1))
        values = np.sum(ranked_y * ranked_x[None, :], axis=1) / denominator
        valid = np.isfinite(values)
        exceed += int(np.sum(np.abs(values[valid]) >= abs(observed) - 1e-15))
        finite += int(np.sum(valid))
    return float((exceed + 1) / (finite + 1)), finite


def percentile_interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])]


def load_outcomes(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(ANIMALS):
        raise ValueError(f"expected {len(ANIMALS)} external rows, found {len(rows)}")
    names = [row["animal"] for row in rows]
    if len(set(names)) != len(names) or set(names) != set(ANIMALS):
        raise ValueError("external animal rows are missing, duplicated, or unexpected")
    by_animal = {row["animal"]: row for row in rows}
    if "base_prior_count" in rows[0]:
        base_prior = np.asarray([float(by_animal[a]["base_prior_count"]) for a in ANIMALS])
    else:
        base_prior = None
    return np.asarray([float(by_animal[a]["sl_rate"]) for a in ANIMALS]), base_prior


def load_steering(path):
    rows = load_json(path)
    if set(rows) != set(ANIMALS):
        raise ValueError("steering benchmark animals are missing or unexpected")
    values = np.asarray([float(rows[a]["peak_pos_rate"]) for a in ANIMALS])
    if not np.isfinite(values).all():
        raise ValueError("steering benchmark contains non-finite values")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--steering", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joined-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=100_000)
    args = parser.parse_args()

    outcome, base_prior = load_outcomes(args.outcomes)
    steering = load_steering(args.steering)
    geometry_json = load_json(args.geometry)["models"]["EXTERNAL-ZOO"]
    readout_json = load_json(args.readout)["models"]["EXTERNAL-ZOO"]
    causal_json = load_json(args.causal)["models"]["EXTERNAL-ZOO"]

    geometry = np.asarray(
        [geometry_json["primary"]["per_animal"][a]["r"] for a in ANIMALS]
    )
    readout = np.asarray(
        [readout_json["positions"]["assistant"]["per_animal"][a]["auc"] for a in ANIMALS]
    )
    causal = np.asarray(
        [causal_json["per_animal"][a]["corrected_causal_auc"] for a in ANIMALS]
    )
    values = {
        "geometry": geometry,
        "readout": readout,
        "causal": causal,
        "steering": steering,
    }
    for name, predictor in values.items():
        if predictor.shape != outcome.shape or not np.isfinite(predictor).all():
            raise ValueError(f"invalid {name} predictor")

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(ANIMALS), size=(args.resamples, len(ANIMALS)))
    bootstraps = {}
    results = {}
    for name in PREDICTORS:
        point = float(stats.spearmanr(values[name], outcome).statistic)
        boot = bootstrap_spearman(values[name], outcome, indices)
        bootstraps[name] = boot
        permutation_rng = np.random.default_rng(args.seed)
        p_value, finite_permutations = permutation_p(
            values[name], outcome, point, permutation_rng, args.resamples
        )
        results[name] = {
            "spearman_rho": point,
            "animal_bootstrap_95_ci": percentile_interval(boot),
            "finite_bootstrap_resamples": int(np.sum(np.isfinite(boot))),
            "permutation_p_two_sided": p_value,
            "finite_permutations": finite_permutations,
        }

    adjusted = bh_adjust([results[name]["permutation_p_two_sided"] for name in IN_HOUSE])
    for name, q_value in zip(IN_HOUSE, adjusted):
        results[name]["bh_fdr_q_in_house_family"] = float(q_value)

    comparisons = {}
    for left, right in (
        ("causal", "geometry"),
        ("causal", "readout"),
        ("steering", "causal"),
    ):
        difference = bootstraps[left] - bootstraps[right]
        comparisons[f"rho_{left}_minus_{right}"] = {
            "point_difference": float(
                results[left]["spearman_rho"] - results[right]["spearman_rho"]
            ),
            "paired_animal_bootstrap_95_ci": percentile_interval(difference),
            "finite_bootstrap_resamples": int(np.sum(np.isfinite(difference))),
        }

    gate_predictors = []
    for name in IN_HOUSE:
        low = results[name]["animal_bootstrap_95_ci"][0]
        q_value = results[name]["bh_fdr_q_in_house_family"]
        if results[name]["spearman_rho"] > 0 and low > 0 and q_value < 0.05:
            gate_predictors.append(name)

    joined_rows = []
    for index, animal in enumerate(ANIMALS):
        joined_rows.append(
            {
                "animal": animal,
                "sl_rate": float(outcome[index]),
                "geometry_r": float(geometry[index]),
                "readout_auc": float(readout[index]),
                "causal_auc": float(causal[index]),
                "steering_peak_rate": float(steering[index]),
            }
        )
    args.joined_output.parent.mkdir(parents=True, exist_ok=True)
    with args.joined_output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(joined_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(joined_rows)

    output = {
        "protocol": "protocol/preregistration-s8-external-transfer.md",
        "external_source": {
            "dataset": "agu18dec/steering_vector_distillation",
            "revision": "4fda20d0413040b2de61448c89182716485d9839",
            "upstream_raw_sha256": "5d19059f211bb2da9d4da54ec14fa41adec6a3357835856ee92b8d62ca5d0e60",
            "stored_snapshot_sha256": sha256(args.outcomes),
            "steering_file": "vectors/zoo/llama/peaks_clean.json",
            "steering_upstream_raw_sha256": "442730d298e7c87a26f7009dcff0da5b4bd58b12159be19edf8e16d6fc2a6946",
            "steering_stored_snapshot_sha256": sha256(args.steering),
            "base_prior_available": base_prior is not None,
            "base_prior_sensitivity_status": (
                "available" if base_prior is not None else "not computable: column absent upstream"
            ),
        },
        "seed": args.seed,
        "resamples": args.resamples,
        "animals": ANIMALS,
        "outcome": "released Llama-3.1-8B trained-student sl_rate; one aggregate row per animal",
        "predictors": results,
        "paired_predictor_comparisons": comparisons,
        "integration_gate": {
            "passing_in_house_predictors": gate_predictors,
            "passes": bool(gate_predictors),
            "rule": (
                "at least one in-house predictor has positive rho, bootstrap 95% CI above zero, "
                "and permutation p surviving BH-FDR at 0.05"
            ),
        },
        "joined_rows": joined_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
