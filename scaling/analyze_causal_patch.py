"""Analyze the pre-registered S5 causal assistant-state patch experiment."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_artifact(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_artifact(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    info = json.loads(str(arrays["metadata_json"].item()))
    if info.get("stage") != "complete":
        raise ValueError(f"incomplete artifact {path}: {info.get('stage')}")
    for key in ("clean_logits", "patched_logits", "permuted_patched_logits"):
        if np.isnan(arrays[key]).any():
            raise ValueError(f"NaNs in {key} for {path}")
    return arrays, info


def interval(values, level=95):
    alpha = (100 - level) / 2
    return [
        float(value)
        for value in np.nanpercentile(values, [alpha, 100 - alpha])
    ]


def animal_contrast(logits):
    n_animals = logits.shape[-1]
    if n_animals < 2:
        raise ValueError("animal contrast needs at least two animals")
    return (n_animals * logits - logits.sum(axis=-1, keepdims=True)) / (
        n_animals - 1
    )


def slopes(x, y, epsilon=1e-12):
    """OLS slopes with intercept. x=[obs,animal], y=[depth,obs,animal]."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    denominator = np.sum(xc * xc, axis=0)
    numerator = np.sum(yc * xc[None, :, :], axis=1)
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator[None, :],
        out=result,
        where=denominator[None, :] > epsilon,
    )
    return result, denominator / len(x)


def donor_recipient_coefficients(donor, recipient, outcome, epsilon=1e-12):
    """Multiple-OLS donor/recipient coefficients with an intercept.

    donor/recipient=[obs,animal], outcome=[depth,obs,animal].  Centering removes
    the intercept.  The closed form avoids a Python loop over animals/depths.
    """
    dc = donor - donor.mean(axis=0, keepdims=True)
    rc = recipient - recipient.mean(axis=0, keepdims=True)
    yc = outcome - outcome.mean(axis=1, keepdims=True)
    dd = np.sum(dc * dc, axis=0)
    rr = np.sum(rc * rc, axis=0)
    dr = np.sum(dc * rc, axis=0)
    dy = np.sum(yc * dc[None, :, :], axis=1)
    ry = np.sum(yc * rc[None, :, :], axis=1)
    determinant = dd * rr - dr * dr
    beta_donor = np.full_like(dy, np.nan, dtype=np.float64)
    gamma_recipient = np.full_like(ry, np.nan, dtype=np.float64)
    np.divide(
        dy * rr[None, :] - ry * dr[None, :],
        determinant[None, :],
        out=beta_donor,
        where=determinant[None, :] > epsilon,
    )
    np.divide(
        ry * dd[None, :] - dy * dr[None, :],
        determinant[None, :],
        out=gamma_recipient,
        where=determinant[None, :] > epsilon,
    )
    condition_numbers = []
    for animal_index in range(donor.shape[1]):
        predictors = np.column_stack(
            [donor[:, animal_index], recipient[:, animal_index]]
        )
        predictors = (predictors - predictors.mean(axis=0)) / predictors.std(
            axis=0, ddof=0
        )
        design = np.column_stack([np.ones(len(predictors)), predictors])
        condition_numbers.append(float(np.linalg.cond(design)))
    return beta_donor, gamma_recipient, np.asarray(condition_numbers), determinant


def causal_auc(beta, depths):
    augmented_depths = np.concatenate([[0.0], depths])
    augmented_beta = np.concatenate(
        [np.zeros((1, beta.shape[1]), dtype=np.float64), beta], axis=0
    )
    return np.trapezoid(augmented_beta, augmented_depths, axis=0) / depths[-1]


def leave_one_pair_corrected_auc(analysis, pair_count):
    """Return [pair, animal] corrected AUCs after deleting each pair cluster."""
    aucs = []
    all_indices = np.arange(2 * pair_count)
    for pair_index in range(pair_count):
        keep = all_indices[
            (all_indices != pair_index)
            & (all_indices != pair_index + pair_count)
        ]
        beta, _, _, _ = donor_recipient_coefficients(
            analysis["donor_clean"][keep],
            analysis["recipient_clean"][keep],
            analysis["patched_contrast"][:, keep],
        )
        aucs.append(causal_auc(beta, analysis["depths"]))
    return np.asarray(aucs)


def prepare(arrays):
    clean = animal_contrast(arrays["clean_logits"].astype(np.float64))
    patched = animal_contrast(arrays["patched_logits"].astype(np.float64))
    permuted = animal_contrast(
        arrays["permuted_patched_logits"].astype(np.float64)
    )
    donor = arrays["donor_indices"].astype(np.int64)
    recipient = arrays["recipient_indices"].astype(np.int64)
    delta_total = clean[donor] - clean[recipient]
    delta_patch = patched - clean[recipient][None, :, :]
    delta_permuted = permuted - clean[recipient][None, :, :]
    return clean, patched, permuted, delta_total, delta_patch, delta_permuted


def point_analysis(arrays, info):
    clean, patched, permuted, delta_total, delta_patch, delta_permuted = prepare(arrays)
    depths = arrays["actual_relative_depths"].astype(np.float64)
    donor_indices = arrays["donor_indices"].astype(np.int64)
    recipient_indices = arrays["recipient_indices"].astype(np.int64)
    donor_clean = clean[donor_indices]
    recipient_clean = clean[recipient_indices]
    beta_donor, gamma_recipient, condition_numbers, determinants = (
        donor_recipient_coefficients(donor_clean, recipient_clean, patched)
    )
    beta_donor_permuted, gamma_recipient_permuted, _, _ = (
        donor_recipient_coefficients(donor_clean, recipient_clean, permuted)
    )
    wrong_donor_clean = np.roll(donor_clean, shift=1, axis=1)
    beta_donor_wrong, gamma_recipient_wrong, _, _ = donor_recipient_coefficients(
        wrong_donor_clean, recipient_clean, patched
    )
    all_wrong_shift_auc = []
    for shift in range(1, donor_clean.shape[1]):
        shifted_donor = np.roll(donor_clean, shift=shift, axis=1)
        shifted_beta, _, _, _ = donor_recipient_coefficients(
            shifted_donor, recipient_clean, patched
        )
        all_wrong_shift_auc.append(causal_auc(shifted_beta, depths))

    clean_raw = arrays["clean_logits"].astype(np.float64)
    patched_raw = arrays["patched_logits"].astype(np.float64)
    donor_raw = clean_raw[donor_indices]
    recipient_raw = clean_raw[recipient_indices]
    beta_donor_raw, gamma_recipient_raw, _, _ = donor_recipient_coefficients(
        donor_raw, recipient_raw, patched_raw
    )
    beta, variances = slopes(delta_total, delta_patch)
    beta_permuted, _ = slopes(delta_total, delta_permuted)
    wrong_delta = np.roll(delta_total, shift=1, axis=1)
    beta_wrong, _ = slopes(wrong_delta, delta_patch)
    pair_count = int(info["pair_count"])
    beta_forward, _ = slopes(
        delta_total[:pair_count], delta_patch[:, :pair_count]
    )
    beta_reverse, _ = slopes(
        delta_total[pair_count:], delta_patch[:, pair_count:]
    )
    beta_donor_forward, gamma_recipient_forward, _, _ = (
        donor_recipient_coefficients(
            donor_clean[:pair_count],
            recipient_clean[:pair_count],
            patched[:, :pair_count],
        )
    )
    beta_donor_reverse, gamma_recipient_reverse, _, _ = (
        donor_recipient_coefficients(
            donor_clean[pair_count:],
            recipient_clean[pair_count:],
            patched[:, pair_count:],
        )
    )
    auc = causal_auc(beta, depths)
    auc_permuted = causal_auc(beta_permuted, depths)
    auc_wrong = causal_auc(beta_wrong, depths)
    return {
        "depths": depths,
        "donor_clean": donor_clean,
        "recipient_clean": recipient_clean,
        "patched_contrast": patched,
        "permuted_patched_contrast": permuted,
        "beta_donor": beta_donor,
        "gamma_recipient": gamma_recipient,
        "beta_donor_permuted": beta_donor_permuted,
        "gamma_recipient_permuted": gamma_recipient_permuted,
        "beta_donor_wrong": beta_donor_wrong,
        "gamma_recipient_wrong": gamma_recipient_wrong,
        "all_wrong_shift_auc": np.asarray(all_wrong_shift_auc),
        "beta_donor_raw_logits": beta_donor_raw,
        "gamma_recipient_raw_logits": gamma_recipient_raw,
        "corrected_auc_raw_logits": causal_auc(beta_donor_raw, depths),
        "beta_donor_forward": beta_donor_forward,
        "gamma_recipient_forward": gamma_recipient_forward,
        "beta_donor_reverse": beta_donor_reverse,
        "gamma_recipient_reverse": gamma_recipient_reverse,
        "condition_numbers": condition_numbers,
        "design_determinants": determinants,
        "corrected_auc": causal_auc(beta_donor, depths),
        "corrected_auc_permuted": causal_auc(beta_donor_permuted, depths),
        "corrected_auc_wrong": causal_auc(beta_donor_wrong, depths),
        "delta_total": delta_total,
        "delta_patch": delta_patch,
        "delta_permuted": delta_permuted,
        "beta": beta,
        "beta_permuted": beta_permuted,
        "beta_wrong": beta_wrong,
        "beta_forward": beta_forward,
        "beta_reverse": beta_reverse,
        "variances": variances,
        "auc": auc,
        "auc_permuted": auc_permuted,
        "auc_wrong": auc_wrong,
    }


def selected_slopes(x, y, observation_indices, epsilon=1e-12):
    """Chunked bootstrap slopes.

    x=[obs,animal], y=[depth,obs,animal], observation_indices=[B,sampled_obs].
    Returns [B,depth,animal].
    """
    sampled_x = x[observation_indices]
    sampled_y = y[:, observation_indices, :].transpose(1, 0, 2, 3)
    xc = sampled_x - sampled_x.mean(axis=1, keepdims=True)
    yc = sampled_y - sampled_y.mean(axis=2, keepdims=True)
    denominator = np.sum(xc * xc, axis=1)
    numerator = np.sum(yc * xc[:, None, :, :], axis=2)
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(
        numerator,
        denominator[:, None, :],
        out=result,
        where=denominator[:, None, :] > epsilon,
    )
    return result


def selected_donor_coefficients(
    donor, recipient, outcome, observation_indices, epsilon=1e-12
):
    """Bootstrap multiple-OLS coefficients, returning [B,depth,animal]."""
    sampled_donor = donor[observation_indices]
    sampled_recipient = recipient[observation_indices]
    sampled_outcome = outcome[:, observation_indices, :].transpose(1, 0, 2, 3)
    dc = sampled_donor - sampled_donor.mean(axis=1, keepdims=True)
    rc = sampled_recipient - sampled_recipient.mean(axis=1, keepdims=True)
    yc = sampled_outcome - sampled_outcome.mean(axis=2, keepdims=True)
    dd = np.sum(dc * dc, axis=1)
    rr = np.sum(rc * rc, axis=1)
    dr = np.sum(dc * rc, axis=1)
    dy = np.sum(yc * dc[:, None, :, :], axis=2)
    ry = np.sum(yc * rc[:, None, :, :], axis=2)
    determinant = dd * rr - dr * dr
    beta = np.full_like(dy, np.nan, dtype=np.float64)
    gamma = np.full_like(ry, np.nan, dtype=np.float64)
    np.divide(
        dy * rr[:, None, :] - ry * dr[:, None, :],
        determinant[:, None, :],
        out=beta,
        where=determinant[:, None, :] > epsilon,
    )
    np.divide(
        ry * dd[:, None, :] - dy * dr[:, None, :],
        determinant[:, None, :],
        out=gamma,
        where=determinant[:, None, :] > epsilon,
    )
    return beta, gamma, determinant


def bootstrap_distributions(
    analysis,
    n_animals,
    pair_count,
    resamples,
    seed,
    chunk_size=100,
):
    rng = np.random.default_rng(seed)
    depths = analysis["depths"]
    result = {
        "corrected_mean_beta": np.empty(
            (resamples, len(depths)), dtype=np.float64
        ),
        "corrected_mean_beta_minus_permuted": np.empty(
            (resamples, len(depths)), dtype=np.float64
        ),
        "corrected_mean_beta_minus_wrong": np.empty(
            (resamples, len(depths)), dtype=np.float64
        ),
        "corrected_mean_auc": np.empty(resamples, dtype=np.float64),
        "corrected_mean_auc_minus_permuted": np.empty(
            resamples, dtype=np.float64
        ),
        "corrected_mean_auc_minus_wrong": np.empty(resamples, dtype=np.float64),
        "corrected_degenerate_cell_count": np.empty(resamples, dtype=np.int64),
        "mean_beta": np.empty((resamples, len(depths)), dtype=np.float64),
        "mean_beta_minus_permuted": np.empty(
            (resamples, len(depths)), dtype=np.float64
        ),
        "mean_beta_minus_wrong": np.empty(
            (resamples, len(depths)), dtype=np.float64
        ),
        "mean_auc": np.empty(resamples, dtype=np.float64),
        "mean_auc_minus_permuted": np.empty(resamples, dtype=np.float64),
        "mean_auc_minus_wrong": np.empty(resamples, dtype=np.float64),
    }
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        size = stop - start
        sampled_clusters = rng.integers(0, pair_count, size=(size, pair_count))
        observation_indices = np.concatenate(
            [sampled_clusters, sampled_clusters + pair_count], axis=1
        )
        sampled_animals = rng.integers(0, n_animals, size=(size, n_animals))

        corrected_beta, _, corrected_determinant = selected_donor_coefficients(
            analysis["donor_clean"],
            analysis["recipient_clean"],
            analysis["patched_contrast"],
            observation_indices,
        )
        corrected_permuted, _, _ = selected_donor_coefficients(
            analysis["donor_clean"],
            analysis["recipient_clean"],
            analysis["permuted_patched_contrast"],
            observation_indices,
        )
        corrected_wrong, _, _ = selected_donor_coefficients(
            np.roll(analysis["donor_clean"], shift=1, axis=1),
            analysis["recipient_clean"],
            analysis["patched_contrast"],
            observation_indices,
        )

        beta = selected_slopes(
            analysis["delta_total"], analysis["delta_patch"], observation_indices
        )
        beta_permuted = selected_slopes(
            analysis["delta_total"],
            analysis["delta_permuted"],
            observation_indices,
        )
        wrong_total = np.roll(analysis["delta_total"], shift=1, axis=1)
        beta_wrong = selected_slopes(
            wrong_total, analysis["delta_patch"], observation_indices
        )

        animal_index = sampled_animals[:, None, :]
        selected_corrected = np.take_along_axis(
            corrected_beta, animal_index, axis=2
        )
        selected_corrected_permuted = np.take_along_axis(
            corrected_permuted, animal_index, axis=2
        )
        selected_corrected_wrong = np.take_along_axis(
            corrected_wrong, animal_index, axis=2
        )
        result["corrected_mean_beta"][start:stop] = np.nanmean(
            selected_corrected, axis=2
        )
        result["corrected_mean_beta_minus_permuted"][start:stop] = np.nanmean(
            selected_corrected - selected_corrected_permuted, axis=2
        )
        result["corrected_mean_beta_minus_wrong"][start:stop] = np.nanmean(
            selected_corrected - selected_corrected_wrong, axis=2
        )
        result["corrected_degenerate_cell_count"][start:stop] = np.sum(
            corrected_determinant <= 1e-12, axis=1
        )
        selected_beta = np.take_along_axis(beta, animal_index, axis=2)
        selected_permuted = np.take_along_axis(
            beta_permuted, animal_index, axis=2
        )
        selected_wrong = np.take_along_axis(beta_wrong, animal_index, axis=2)
        result["mean_beta"][start:stop] = np.nanmean(selected_beta, axis=2)
        result["mean_beta_minus_permuted"][start:stop] = np.nanmean(
            selected_beta - selected_permuted, axis=2
        )
        result["mean_beta_minus_wrong"][start:stop] = np.nanmean(
            selected_beta - selected_wrong, axis=2
        )

        zeros = np.zeros((size, 1, n_animals), dtype=np.float64)
        augmented_depths = np.concatenate([[0.0], depths])
        corrected_auc = np.trapezoid(
            np.concatenate([zeros, corrected_beta], axis=1),
            augmented_depths,
            axis=1,
        ) / depths[-1]
        corrected_auc_permuted = np.trapezoid(
            np.concatenate([zeros, corrected_permuted], axis=1),
            augmented_depths,
            axis=1,
        ) / depths[-1]
        corrected_auc_wrong = np.trapezoid(
            np.concatenate([zeros, corrected_wrong], axis=1),
            augmented_depths,
            axis=1,
        ) / depths[-1]
        selected_corrected_auc = np.take_along_axis(
            corrected_auc, sampled_animals, axis=1
        )
        selected_corrected_auc_permuted = np.take_along_axis(
            corrected_auc_permuted, sampled_animals, axis=1
        )
        selected_corrected_auc_wrong = np.take_along_axis(
            corrected_auc_wrong, sampled_animals, axis=1
        )
        result["corrected_mean_auc"][start:stop] = np.nanmean(
            selected_corrected_auc, axis=1
        )
        result["corrected_mean_auc_minus_permuted"][start:stop] = np.nanmean(
            selected_corrected_auc - selected_corrected_auc_permuted, axis=1
        )
        result["corrected_mean_auc_minus_wrong"][start:stop] = np.nanmean(
            selected_corrected_auc - selected_corrected_auc_wrong, axis=1
        )
        auc = np.trapezoid(
            np.concatenate([zeros, beta], axis=1), augmented_depths, axis=1
        ) / depths[-1]
        auc_permuted = np.trapezoid(
            np.concatenate([zeros, beta_permuted], axis=1),
            augmented_depths,
            axis=1,
        ) / depths[-1]
        auc_wrong = np.trapezoid(
            np.concatenate([zeros, beta_wrong], axis=1),
            augmented_depths,
            axis=1,
        ) / depths[-1]
        selected_auc = np.take_along_axis(auc, sampled_animals, axis=1)
        selected_auc_permuted = np.take_along_axis(
            auc_permuted, sampled_animals, axis=1
        )
        selected_auc_wrong = np.take_along_axis(
            auc_wrong, sampled_animals, axis=1
        )
        result["mean_auc"][start:stop] = np.nanmean(selected_auc, axis=1)
        result["mean_auc_minus_permuted"][start:stop] = np.nanmean(
            selected_auc - selected_auc_permuted, axis=1
        )
        result["mean_auc_minus_wrong"][start:stop] = np.nanmean(
            selected_auc - selected_auc_wrong, axis=1
        )
    return result


def depth_index(requested_depths, target):
    requested = np.asarray(requested_depths, dtype=np.float64)
    index = int(np.argmin(np.abs(requested - target)))
    if abs(requested[index] - target) > 0.03:
        return None
    return index


def summarize_model(label, path, arrays, info, analysis, bootstrap):
    animals = arrays["animals"].tolist()
    beta = analysis["beta"]
    corrected = analysis["beta_donor"]
    summary = {
        "label": label,
        "path": str(path),
        "model": info["model"],
        "pair_count": int(info["pair_count"]),
        "ordered_observation_count": int(2 * info["pair_count"]),
        "animals": animals,
        "requested_depths": info["requested_depths"],
        "actual_relative_depths": analysis["depths"].tolist(),
        "block_indices": arrays["block_indices"].astype(int).tolist(),
        "corrected_primary": {
            "estimand": (
                "multiple-OLS coefficient on clean donor animal contrast while "
                "holding clean recipient animal contrast fixed"
            ),
            "mean_donor_beta_by_depth": np.nanmean(corrected, axis=1).tolist(),
            "mean_donor_beta_95_ci_by_depth": [
                interval(bootstrap["corrected_mean_beta"][:, index])
                for index in range(len(analysis["depths"]))
            ],
            "mean_recipient_gamma_by_depth": np.nanmean(
                analysis["gamma_recipient"], axis=1
            ).tolist(),
            "mean_permuted_donor_beta_by_depth": np.nanmean(
                analysis["beta_donor_permuted"], axis=1
            ).tolist(),
            "matched_minus_permuted_mean_beta_by_depth": np.nanmean(
                corrected - analysis["beta_donor_permuted"], axis=1
            ).tolist(),
            "matched_minus_permuted_95_ci_by_depth": [
                interval(
                    bootstrap["corrected_mean_beta_minus_permuted"][:, index]
                )
                for index in range(len(analysis["depths"]))
            ],
            "mean_wrong_animal_donor_beta_by_depth": np.nanmean(
                analysis["beta_donor_wrong"], axis=1
            ).tolist(),
            "matched_minus_wrong_mean_beta_by_depth": np.nanmean(
                corrected - analysis["beta_donor_wrong"], axis=1
            ).tolist(),
            "matched_minus_wrong_95_ci_by_depth": [
                interval(bootstrap["corrected_mean_beta_minus_wrong"][:, index])
                for index in range(len(analysis["depths"]))
            ],
            "direction_half_mean_donor_beta": {
                "forward": np.nanmean(
                    analysis["beta_donor_forward"], axis=1
                ).tolist(),
                "reverse": np.nanmean(
                    analysis["beta_donor_reverse"], axis=1
                ).tolist(),
            },
            "standardized_design_condition_number_by_animal": {
                animal: float(analysis["condition_numbers"][index])
                for index, animal in enumerate(animals)
            },
            "maximum_standardized_design_condition_number": float(
                np.max(analysis["condition_numbers"])
            ),
            "all_condition_numbers_below_100": bool(
                np.all(analysis["condition_numbers"] < 100)
            ),
            "bootstrap_degenerate_cell_count_total": int(
                np.sum(bootstrap["corrected_degenerate_cell_count"])
            ),
            "corrected_causal_auc_mean": float(
                np.nanmean(analysis["corrected_auc"])
            ),
            "corrected_causal_auc_95_ci": interval(
                bootstrap["corrected_mean_auc"]
            ),
            "corrected_auc_minus_permuted_mean": float(
                np.nanmean(
                    analysis["corrected_auc"]
                    - analysis["corrected_auc_permuted"]
                )
            ),
            "corrected_auc_minus_permuted_95_ci": interval(
                bootstrap["corrected_mean_auc_minus_permuted"]
            ),
            "corrected_auc_minus_wrong_mean": float(
                np.nanmean(
                    analysis["corrected_auc"]
                    - analysis["corrected_auc_wrong"]
                )
            ),
            "corrected_auc_minus_wrong_95_ci": interval(
                bootstrap["corrected_mean_auc_minus_wrong"]
            ),
            "raw_target_logit_sensitivity": {
                "mean_donor_beta_by_depth": np.nanmean(
                    analysis["beta_donor_raw_logits"], axis=1
                ).tolist(),
                "mean_recipient_gamma_by_depth": np.nanmean(
                    analysis["gamma_recipient_raw_logits"], axis=1
                ).tolist(),
                "causal_auc_mean": float(
                    np.nanmean(analysis["corrected_auc_raw_logits"])
                ),
                "per_animal_causal_auc": {
                    animal: float(analysis["corrected_auc_raw_logits"][index])
                    for index, animal in enumerate(animals)
                },
            },
            "all_wrong_concept_circular_shifts": {
                "shift_count": int(len(analysis["all_wrong_shift_auc"])),
                "mean_auc_by_shift": np.nanmean(
                    analysis["all_wrong_shift_auc"], axis=1
                ).tolist(),
                "mean_auc_range": [
                    float(np.nanmin(np.nanmean(
                        analysis["all_wrong_shift_auc"], axis=1
                    ))),
                    float(np.nanmax(np.nanmean(
                        analysis["all_wrong_shift_auc"], axis=1
                    ))),
                ],
            },
        },
        "original_shared_baseline_diagnostic": {
            "warning": (
                "delta and Delta share the recipient term; this slope is invalid "
                "as a causal-magnitude primary and is retained for transparency"
            ),
        "mean_beta_by_depth": np.nanmean(beta, axis=1).tolist(),
        "mean_beta_95_ci_by_depth": [
            interval(bootstrap["mean_beta"][:, index])
            for index in range(len(analysis["depths"]))
        ],
        "mean_permuted_beta_by_depth": np.nanmean(
            analysis["beta_permuted"], axis=1
        ).tolist(),
        "matched_minus_permuted_mean_beta_by_depth": np.nanmean(
            beta - analysis["beta_permuted"], axis=1
        ).tolist(),
        "matched_minus_permuted_95_ci_by_depth": [
            interval(bootstrap["mean_beta_minus_permuted"][:, index])
            for index in range(len(analysis["depths"]))
        ],
        "mean_wrong_animal_beta_by_depth": np.nanmean(
            analysis["beta_wrong"], axis=1
        ).tolist(),
        "matched_minus_wrong_mean_beta_by_depth": np.nanmean(
            beta - analysis["beta_wrong"], axis=1
        ).tolist(),
        "matched_minus_wrong_95_ci_by_depth": [
            interval(bootstrap["mean_beta_minus_wrong"][:, index])
            for index in range(len(analysis["depths"]))
        ],
        "direction_half_mean_beta": {
            "forward": np.nanmean(analysis["beta_forward"], axis=1).tolist(),
            "reverse": np.nanmean(analysis["beta_reverse"], axis=1).tolist(),
        },
        "delta_variance_by_animal": {
            animal: float(analysis["variances"][index])
            for index, animal in enumerate(animals)
        },
        "all_delta_variances_above_1e-8": bool(
            np.all(analysis["variances"] > 1e-8)
        ),
        "causal_auc_mean": float(np.nanmean(analysis["auc"])),
        "causal_auc_crossed_bootstrap_95_ci": interval(bootstrap["mean_auc"]),
        "causal_auc_minus_permuted_mean": float(
            np.nanmean(analysis["auc"] - analysis["auc_permuted"])
        ),
        "causal_auc_minus_permuted_95_ci": interval(
            bootstrap["mean_auc_minus_permuted"]
        ),
        "causal_auc_minus_wrong_mean": float(
            np.nanmean(analysis["auc"] - analysis["auc_wrong"])
        ),
        "causal_auc_minus_wrong_95_ci": interval(
            bootstrap["mean_auc_minus_wrong"]
        ),
        },
        "numerical_controls": {
            "tolerance": float(info["numerical_tolerance"]),
            "clean_duplicate_max_abs_delta": float(
                arrays["clean_duplicate_max_abs_delta"].item()
            ),
            "identity_max_abs_delta_by_depth": arrays[
                "identity_max_abs_delta_by_depth"
            ].astype(float).tolist(),
            "pass": bool(
                float(arrays["clean_duplicate_max_abs_delta"].item())
                <= float(info["numerical_tolerance"])
                and np.max(arrays["identity_max_abs_delta_by_depth"])
                <= float(info["numerical_tolerance"])
            ),
        },
        "per_animal": {
            animal: {
                "beta_by_depth": beta[:, index].tolist(),
                "permuted_beta_by_depth": analysis["beta_permuted"][
                    :, index
                ].tolist(),
                "wrong_animal_beta_by_depth": analysis["beta_wrong"][
                    :, index
                ].tolist(),
                "causal_auc": float(analysis["auc"][index]),
                "corrected_donor_beta_by_depth": corrected[:, index].tolist(),
                "corrected_recipient_gamma_by_depth": analysis[
                    "gamma_recipient"
                ][:, index].tolist(),
                "corrected_causal_auc": float(
                    analysis["corrected_auc"][index]
                ),
            }
            for index, animal in enumerate(animals)
        },
    }

    requested = info["requested_depths"]
    index_25 = depth_index(requested, 0.25)
    index_75 = depth_index(requested, 0.75)
    index_90 = depth_index(requested, 0.90)
    index_97 = depth_index(requested, 0.97)
    if None not in (index_25, index_75, index_90, index_97):
        depth_rise_controls = {}
        for name, later_index in (
            ("0.75_minus_0.25", index_75),
            ("0.90_minus_0.25", index_90),
            ("0.97_minus_0.25", index_97),
        ):
            point_difference = corrected[later_index] - corrected[index_25]
            bootstrap_difference = (
                bootstrap["corrected_mean_beta"][:, later_index]
                - bootstrap["corrected_mean_beta"][:, index_25]
            )
            depth_rise_controls[name] = {
                "mean_difference": float(np.nanmean(point_difference)),
                "crossed_bootstrap_95_ci": interval(bootstrap_difference),
            }
        summary["corrected_primary"]["depth_rise_controls"] = (
            depth_rise_controls
        )

    if len(animals) == 18 and None not in (index_25, index_75, index_90):
        depth_rise = (
            bootstrap["corrected_mean_beta"][:, index_90]
            - bootstrap["corrected_mean_beta"][:, index_25]
        )
        conditions = {
            "numerical_controls": summary["numerical_controls"]["pass"],
            "nondegenerate": summary["original_shared_baseline_diagnostic"][
                "all_delta_variances_above_1e-8"
            ],
            "condition_numbers_below_100": summary["corrected_primary"][
                "all_condition_numbers_below_100"
            ],
            "beta_075_lower_above_zero": interval(
                bootstrap["corrected_mean_beta"][:, index_75]
            )[0]
            > 0,
            "beta_090_lower_above_zero": interval(
                bootstrap["corrected_mean_beta"][:, index_90]
            )[0]
            > 0,
            "matched_minus_permuted_075_lower_above_zero": interval(
                bootstrap["corrected_mean_beta_minus_permuted"][:, index_75]
            )[0]
            > 0,
            "matched_minus_permuted_090_lower_above_zero": interval(
                bootstrap["corrected_mean_beta_minus_permuted"][:, index_90]
            )[0]
            > 0,
            "matched_minus_wrong_090_lower_above_zero": interval(
                bootstrap["corrected_mean_beta_minus_wrong"][:, index_90]
            )[0]
            > 0,
            "both_directions_positive_075": bool(
                np.nanmean(analysis["beta_donor_forward"][index_75]) > 0
                and np.nanmean(analysis["beta_donor_reverse"][index_75]) > 0
            ),
            "both_directions_positive_090": bool(
                np.nanmean(analysis["beta_donor_forward"][index_90]) > 0
                and np.nanmean(analysis["beta_donor_reverse"][index_90]) > 0
            ),
            "beta_090_minus_025_lower_above_zero": interval(depth_rise)[0] > 0,
        }
        summary["local_gate"] = {
            "mechanistic_conditions": conditions,
            "mechanistic_conditions_pass": bool(all(conditions.values())),
            "abstract_sentence_judgment_required": True,
            "paid_run_authorized_by_analysis_alone": False,
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", type=parse_artifact, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompting/results/causal_patch_summary.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100 or args.bootstrap_chunk_size < 1:
        parser.error("bootstrap resamples >=100 and positive chunk size required")

    models = {}
    raw = {}
    boot = {}
    reference = None
    for label, path in args.artifact:
        arrays, info = load_artifact(path)
        signature = {
            "animals": arrays["animals"].tolist(),
            "unique_numbers": arrays["unique_numbers"].tolist(),
            "donor_indices": arrays["donor_indices"].tolist(),
            "recipient_indices": arrays["recipient_indices"].tolist(),
            "requested_depths": info["requested_depths"],
        }
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError("artifacts do not share the exact paired S5 design")
        analysis = point_analysis(arrays, info)
        bootstrap = bootstrap_distributions(
            analysis,
            len(arrays["animals"]),
            int(info["pair_count"]),
            args.bootstrap_resamples,
            args.seed,
            args.bootstrap_chunk_size,
        )
        models[label] = summarize_model(
            label, path, arrays, info, analysis, bootstrap
        )
        raw[label] = analysis
        boot[label] = bootstrap

    contrasts = {}
    labels = list(models)
    for smaller, larger in zip(labels, labels[1:]):
        delta_auc = (
            raw[larger]["corrected_auc"] - raw[smaller]["corrected_auc"]
        )
        bootstrap_delta = (
            boot[larger]["corrected_mean_auc"]
            - boot[smaller]["corrected_mean_auc"]
        )
        ci = interval(bootstrap_delta)
        if ci[0] > 0:
            decision = "stronger_or_earlier_at_larger_model"
        elif ci[1] < 0:
            decision = "weaker_or_later_at_larger_model"
        else:
            decision = "causal_scale_change_unresolved"
        raw_logit_delta = (
            raw[larger]["corrected_auc_raw_logits"]
            - raw[smaller]["corrected_auc_raw_logits"]
        )
        pair_count = int(models[smaller]["pair_count"])
        leave_one_smaller = leave_one_pair_corrected_auc(
            raw[smaller], pair_count
        )
        leave_one_larger = leave_one_pair_corrected_auc(
            raw[larger], pair_count
        )
        leave_one_delta = np.nanmean(
            leave_one_larger - leave_one_smaller, axis=1
        )
        contrasts[f"{larger}-{smaller}"] = {
            "primary_estimand": "corrected donor-coefficient causal AUC",
            "paired_mean_delta_corrected_causal_auc": float(
                np.nanmean(delta_auc)
            ),
            "crossed_bootstrap_95_ci": ci,
            "animals_increasing": int(np.sum(delta_auc > 0)),
            "n_animals": len(delta_auc),
            "decision": decision,
            "equivalence_not_inferred_from_unresolved_interval": True,
            "raw_target_logit_sensitivity": {
                "paired_mean_delta_causal_auc": float(
                    np.nanmean(raw_logit_delta)
                ),
                "animals_increasing": int(np.sum(raw_logit_delta > 0)),
                "n_animals": int(len(raw_logit_delta)),
                "per_animal_delta_range": [
                    float(np.nanmin(raw_logit_delta)),
                    float(np.nanmax(raw_logit_delta)),
                ],
            },
            "leave_one_unordered_pair_cluster_out": {
                "deletion_count": pair_count,
                "paired_mean_delta_by_deleted_pair": leave_one_delta.tolist(),
                "paired_mean_delta_range": [
                    float(np.nanmin(leave_one_delta)),
                    float(np.nanmax(leave_one_delta)),
                ],
                "median_paired_mean_delta": float(
                    np.nanmedian(leave_one_delta)
                ),
                "all_positive": bool(np.all(leave_one_delta > 0)),
            },
            "original_shared_baseline_secondary": {
                "paired_mean_delta_auc": float(
                    np.nanmean(raw[larger]["auc"] - raw[smaller]["auc"])
                ),
                "crossed_bootstrap_95_ci": interval(
                    boot[larger]["mean_auc"] - boot[smaller]["mean_auc"]
                ),
                "warning": "invalid as a causal-magnitude primary",
            },
        }

    output = {
        "protocol": "Paper/Supplement/preregistration-s5.md",
        "seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_unit": (
            "crossed resampling of animals and unordered number-pair clusters; "
            "both directions retained"
        ),
        "models": models,
        "contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
