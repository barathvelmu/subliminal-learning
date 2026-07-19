"""Verify the committed summaries against the paper's headline table."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "prompting" / "results"


def load(name: str) -> dict:
    with (RESULTS / name).open() as handle:
        return json.load(handle)


def close(label: str, actual: float, expected: float, tolerance: float = 5e-7) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


geometry = load("geometry_8b70b_cuda_summary.json")
readout = load("layerwise_probe_8b70b_summary.json")
causal = load("causal_patch_s5_8b70b_cuda_summary.json")
qwen = load("sequence_probe_qwen_scaling_summary.json")

checks = {
    "geometry 8B": (geometry["models"]["8B-CUDA"]["primary"]["mean_r"], 0.1877298408),
    "geometry 70B": (geometry["models"]["70B"]["primary"]["mean_r"], 0.1075825952),
    "geometry delta": (
        geometry["primary_scaling_contrasts"]["70B-8B-CUDA"]["mean_paired_delta_r"],
        -0.0801472456,
    ),
    "readout AUC 8B": (
        readout["models"]["8B-CUDA"]["positions"]["assistant"]["auc_mean"],
        0.2591305208,
    ),
    "readout AUC 70B": (
        readout["models"]["70B-CUDA"]["positions"]["assistant"]["auc_mean"],
        0.2687580224,
    ),
    "readout AUC delta": (
        readout["paired_auc_contrasts"]["70B-CUDA-8B-CUDA"]["assistant"][
            "mean_paired_delta_auc"
        ],
        0.0096275015,
    ),
    "causal AUC 8B": (
        causal["models"]["8B"]["corrected_primary"]["corrected_causal_auc_mean"],
        0.2538898065,
    ),
    "causal AUC 70B": (
        causal["models"]["70B"]["corrected_primary"]["corrected_causal_auc_mean"],
        0.5397040863,
    ),
    "causal AUC delta": (
        causal["contrasts"]["70B-8B"]["paired_mean_delta_corrected_causal_auc"],
        0.2858142798,
    ),
    "Qwen 0.6B width-3": (
        qwen["models"]["Qwen3-0.6B"]["primary_width_3_full_sequence"]["mean_r"],
        -0.0497774206,
    ),
    "Qwen 1.7B width-3": (
        qwen["models"]["Qwen3-1.7B"]["primary_width_3_full_sequence"]["mean_r"],
        -0.0478265734,
    ),
}

for name, (actual, expected) in checks.items():
    close(name, actual, expected)

if causal["contrasts"]["70B-8B"]["animals_increasing"] != 18:
    raise AssertionError("causal AUC must increase for all 18 concepts")

print(f"verified {len(checks)} headline statistics and the 18/18 direction check")
