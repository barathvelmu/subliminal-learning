"""Create paper-ready figures for the S3/S4 mechanism results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "prompting" / "results"
FIGURES = ROOT / "prompting" / "figures"
BLUE = "#2864a6"
ORANGE = "#d97706"
GREEN = "#16856b"
GRAY = "#697386"


def load(name):
    return json.loads((RESULTS / name).read_text())


def errorbar(ax, x, mean, ci, **kwargs):
    low = mean - ci[0]
    high = ci[1] - mean
    ax.errorbar(x, mean, yerr=[[low], [high]], capsize=4, **kwargs)


def mechanism_figure():
    geometry = load("geometry_8b70b_cuda_summary.json")
    depth = load("layerwise_probe_8b70b_summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))

    ax = axes[0]
    animals = list(geometry["models"]["8B-CUDA"]["primary"]["per_animal"])
    values_8b = np.asarray(
        [
            geometry["models"]["8B-CUDA"]["primary"]["per_animal"][animal]["r"]
            for animal in animals
        ]
    )
    values_70b = np.asarray(
        [
            geometry["models"]["70B"]["primary"]["per_animal"][animal]["r"]
            for animal in animals
        ]
    )
    contrast = geometry["primary_scaling_contrasts"]["70B-8B-CUDA"]
    limits = (-0.02, 0.34)
    ax.plot(limits, limits, color=GRAY, lw=1, linestyle="--")
    ax.scatter(
        values_8b,
        values_70b,
        s=32,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.set_xlabel("8B geometry vs. reverse score $r$")
    ax.set_ylabel("70B geometry vs. reverse score $r$")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("a  Static geometry weakens for 14/18 animals")
    ax.text(
        0.04,
        0.96,
        f"paired Δ = {contrast['mean_paired_delta_r']:+.3f}\n95% CI [{contrast['paired_bootstrap_95_ci'][0]:+.3f}, {contrast['paired_bootstrap_95_ci'][1]:+.3f}]",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )
    ax.grid(color="0.90", linewidth=0.6)

    for ax, position, title in [
        (axes[1], "assistant", "b  Contextual answer trajectory stays similar"),
        (axes[2], "number_mean", "c  Number-position readout stays weak"),
    ]:
        for label, display, color in [
            ("8B-CUDA", "8B", BLUE),
            ("70B-CUDA", "70B", ORANGE),
        ]:
            item = depth["models"][label]["positions"][position]
            x = np.asarray(item["relative_depth"])
            y = np.asarray(item["mean_r_by_layer"])
            ci = np.asarray(item["bootstrap_95_ci_by_layer"])
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=0.13, lw=0)
            ax.plot(x, y, color=color, lw=2.2, label=display)
            if position == "assistant":
                half = item["earliest_half_final_relative_depth"]
                ax.scatter(
                    [half],
                    [0.5 * item["final_layer_mean_r"]],
                    color=color,
                    s=28,
                    zorder=4,
                )
        ax.axhline(0, color=GRAY, lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Relative model depth")
        ax.set_ylabel("Mean behavior-readout r")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Static geometry weakens while fixed-head readout AUC remains similar",
        y=1.03,
        fontsize=14,
    )
    fig.tight_layout()
    output = FIGURES / "s4_geometry_vs_depth.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def sequence_figure():
    data = load("sequence_probe_qwen_scaling_summary.json")
    labels = ["Qwen3-0.6B", "Qwen3-1.7B"]
    displays = ["Qwen 0.6B", "Qwen 1.7B"]
    colors = [BLUE, ORANGE]
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))

    ax = axes[0]
    offsets = [-0.11, 0.11]
    for model_index, (label, display, color) in enumerate(
        zip(labels, displays, colors)
    ):
        model = data["models"][label]
        for metric_index, key in enumerate(
            ("primary_width_3_full_sequence", "first_token_width_3")
        ):
            item = model[key]
            errorbar(
                ax,
                model_index + offsets[metric_index],
                item["mean_r"],
                item["bootstrap_95_ci"],
                fmt="o" if metric_index == 0 else "s",
                color=color,
                mfc=color if metric_index == 0 else "white",
                ms=7,
                lw=2,
            )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(2), displays)
    ax.set_ylabel("Mean r across 18 animals")
    ax.set_title("a  Three-digit targets")
    ax.plot([], [], "o", color=GRAY, label="full sequence")
    ax.plot([], [], "s", color=GRAY, mfc="white", label="first token")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    metric_keys = [
        ("sum_logp", "sum log p"),
        ("mean_logp_per_target_token", "mean/token"),
        ("within_width_standardized", "within-width z"),
    ]
    for model_index, (label, color) in enumerate(zip(labels, colors)):
        model = data["models"][label]["secondary_all_widths"]
        for metric_index, (key, _) in enumerate(metric_keys):
            item = model[key]
            errorbar(
                ax,
                metric_index + (model_index - 0.5) * 0.18,
                item["mean_r"],
                item["bootstrap_95_ci"],
                fmt="o",
                color=color,
                ms=6,
                lw=1.8,
            )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(3), [x[1] for x in metric_keys], rotation=12)
    ax.set_ylabel("Mean r across all widths")
    ax.set_title("b  Per-token averaging creates a confound")
    ax.plot([], [], "o", color=BLUE, label="0.6B")
    ax.plot([], [], "o", color=ORANGE, label="1.7B")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    for model_index, (label, color) in enumerate(zip(labels, colors)):
        model = data["models"][label]["by_width_full_sequence"]
        for width_index, width in enumerate(("1", "2", "3")):
            item = model[width]
            errorbar(
                ax,
                width_index + (model_index - 0.5) * 0.18,
                item["mean_r"],
                item["bootstrap_95_ci"],
                fmt="o",
                color=color,
                ms=6,
                lw=1.8,
            )
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(range(3), ["1 digit", "2 digits", "3 digits"])
    ax.set_ylabel("Mean full-sequence r")
    ax.set_title("c  Association changes with width")

    fig.suptitle(
        "In two Qwen models, exact digit-sequence scoring does not recover the atomic association",
        y=1.03,
        fontsize=14,
    )
    fig.tight_layout()
    output = FIGURES / "s4_multitoken_sequence.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for output in (mechanism_figure(), sequence_figure()):
        print(f"saved {output}")


if __name__ == "__main__":
    main()
