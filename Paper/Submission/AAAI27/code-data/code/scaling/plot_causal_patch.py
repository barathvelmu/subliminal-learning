"""Plot the pre-registered S5B 8B/70B causal handoff result."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"8B": "#2774AE", "70B": "#D55E00"}
MARKERS = {"8B": "o", "70B": "s"}


def augmented(values, depths, initial):
    return np.concatenate([[0.0], np.asarray(depths)]), np.concatenate(
        [[initial], np.asarray(values)]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.summary.read_text())
    models = result["models"]
    contrast = result["contrasts"]["70B-8B"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))

    donor_axis = axes[0]
    for label in ("8B", "70B"):
        model = models[label]
        primary = model["corrected_primary"]
        depths = model["actual_relative_depths"]
        beta = primary["mean_donor_beta_by_depth"]
        intervals = np.asarray(primary["mean_donor_beta_95_ci_by_depth"])
        x, y = augmented(beta, depths, 0.0)
        lower = np.concatenate([[0.0], intervals[:, 0]])
        upper = np.concatenate([[0.0], intervals[:, 1]])
        donor_axis.fill_between(x, lower, upper, color=COLORS[label], alpha=0.16)
        donor_axis.plot(
            x,
            y,
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=2.0,
            markersize=4.5,
            label=label,
        )
    donor_axis.axhline(0, color="0.55", linewidth=0.8)
    donor_axis.set(
        xlabel="Relative transformer depth",
        ylabel="Donor causal coefficient",
        xlim=(-0.015, 1.0),
        ylim=(-0.05, 1.05),
        title="a  Donor control rises sooner in relative depth",
    )
    donor_axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    donor_axis.legend(frameon=False, loc="lower right", title="Model")
    donor_axis.grid(axis="y", color="0.88", linewidth=0.6)

    handoff_axis = axes[1]
    for label in ("8B", "70B"):
        model = models[label]
        primary = model["corrected_primary"]
        depths = model["actual_relative_depths"]
        x, donor = augmented(primary["mean_donor_beta_by_depth"], depths, 0.0)
        _, recipient = augmented(
            primary["mean_recipient_gamma_by_depth"], depths, 1.0
        )
        handoff_axis.plot(
            x,
            donor,
            color=COLORS[label],
            marker=MARKERS[label],
            linewidth=2.0,
            markersize=4.2,
            label=f"{label} donor",
        )
        handoff_axis.plot(
            x,
            recipient,
            color=COLORS[label],
            linestyle="--",
            linewidth=1.7,
            label=f"{label} recipient",
        )
    delta = contrast["paired_mean_delta_corrected_causal_auc"]
    ci = contrast["crossed_bootstrap_95_ci"]
    handoff_axis.text(
        0.04,
        0.49,
        f"Causal AUC difference\n70B − 8B = {delta:+.3f}\n95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]",
        transform=handoff_axis.transAxes,
        va="center",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.8"},
    )
    handoff_axis.set(
        xlabel="Relative transformer depth",
        ylabel="Conditional influence coefficient",
        xlim=(-0.015, 1.0),
        ylim=(-0.05, 1.05),
        title="b  Conditional influence shifts in relative depth",
    )
    handoff_axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    handoff_axis.legend(frameon=False, loc="lower right", ncol=2, fontsize=7.5)
    handoff_axis.grid(axis="y", color="0.88", linewidth=0.6)

    paired_axis = axes[2]
    animal_order = list(models["8B"]["per_animal"])
    auc_8b = np.asarray(
        [models["8B"]["per_animal"][animal]["corrected_causal_auc"] for animal in animal_order]
    )
    auc_70b = np.asarray(
        [models["70B"]["per_animal"][animal]["corrected_causal_auc"] for animal in animal_order]
    )
    limits = (0.19, 0.59)
    paired_axis.plot(limits, limits, color="0.55", linestyle="--", linewidth=1.0)
    paired_axis.scatter(
        auc_8b,
        auc_70b,
        s=34,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    paired_axis.text(
        0.04,
        0.96,
        (
            f"18/18 above diagonal\n"
            f"paired $\\Delta$ = {delta:+.3f}\n"
            f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        ),
        transform=paired_axis.transAxes,
        va="top",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )
    paired_axis.set(
        xlabel="8B normalized causal AUC",
        ylabel="70B normalized causal AUC",
        xlim=limits,
        ylim=limits,
        title="c  Animal-level shift is uniform",
    )
    paired_axis.set_aspect("equal", adjustable="box")
    paired_axis.grid(color="0.90", linewidth=0.6)

    figure.tight_layout(w_pad=1.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
