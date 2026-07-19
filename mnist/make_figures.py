"""
Make readable MNIST figures from results/.

Outputs to figures/:
  baseline_bars.png      6-condition baseline with seed error bars
  sweeps.png             accuracy vs each swept knob (annotated)
  mechanism_scatter.png  hidden-similarity-to-teacher vs student accuracy

Design: large fonts, control and chance reference lines, and a one-line caption
under each figure.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 150,
    }
)
CHANCE = 0.10
os.makedirs("figures", exist_ok=True)


def caption(fig, text):
    fig.text(
        0.5, -0.02, text, ha="center", va="top", fontsize=12, style="italic", wrap=True
    )


# ── baseline bars ─────────────────────────────────────────────────────────────
def baseline_bars():
    d = json.load(open("results/baseline_5seed.json"))
    s = d["summary"]
    labels = [
        "Reference\n(random)",
        "Teacher",
        "Student\n(aux only)",
        "Student\n(all logits)",
        "Cross-model\n(aux only)",
        "Cross-model\n(all logits)",
    ]
    keys = [
        "reference",
        "teacher",
        "student_aux",
        "student_all",
        "xmodel_aux",
        "xmodel_all",
    ]
    means = [s[k]["mean"] for k in keys]
    errs = [s[k]["std"] for k in keys]
    colors = ["gray", "#8c564b", "#9467bd", "#9467bd", "#c5b0d5", "#c5b0d5"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(labels, means, yerr=errs, capsize=5, color=colors)
    ax.axhline(CHANCE, ls=":", c="black", lw=1.5, label="chance (10%)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("MNIST test accuracy after auxiliary-only noise distillation")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    # Mark the auxiliary-only student.
    i = 2
    ell = Ellipse((i, means[i]), 0.9, 0.22, fill=False, edgecolor="red", lw=2.5)
    ax.add_patch(ell)
    ax.annotate(
        "noise inputs; auxiliary-target loss only",
        xy=(i, means[i] + 0.11),
        xytext=(i + 0.1, means[i] + 0.30),
        fontsize=11,
        color="red",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    # Mark the different-initialization control.
    j = 4
    ax.annotate(
        "different initialization\n→ near chance",
        xy=(j, means[j] + 0.02),
        xytext=(j - 0.4, means[j] + 0.28),
        fontsize=11,
        color="black",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    caption(
        fig,
        "Bars show means over 5 seeds; error bars show SD. The auxiliary-only student "
        "receives no digit labels. The different-initialization control supports dependence "
        "on shared initialization in this setup.",
    )
    fig.tight_layout()
    fig.savefig("figures/baseline_bars.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/baseline_bars.png")


# ── sweeps ────────────────────────────────────────────────────────────────────
def sweeps():
    df = pd.read_csv("results/sweeps_summary.csv")
    sa = df[df.condition == "student_aux"]
    xa = df[df.condition == "xmodel_aux"]
    order = ["loss", "noise", "distill_epochs", "init_scale", "width"]
    titles = {
        "loss": "Loss function",
        "noise": "Noise type used for distillation",
        "distill_epochs": "Distillation epochs",
        "init_scale": "Init weight scale",
        "width": "Hidden width",
    }
    notes = {
        "loss": "MSE exceeds KL in this sweep",
        "noise": "uniform ≈ Gaussian; real images are lower",
        "distill_epochs": "chance at 0 epochs; higher after longer training",
        "init_scale": "largest value near the default scale",
        "width": "lower transfer at greater width",
    }
    xlabels = {
        "loss": "distillation loss function",
        "noise": "distillation input distribution",
        "distill_epochs": "number of distillation epochs",
        "init_scale": "init weight-scale multiplier",
        "width": "hidden-layer width (neurons), log scale",
    }
    log_x = {"width"}
    numeric = {"distill_epochs", "init_scale", "width"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, sw in zip(axes, order):
        s = sa[sa.sweep == sw]
        sx = xa[xa.sweep == sw]
        vals = list(s["value"])
        if sw in numeric:
            xx = [float(v) for v in vals]
            ax.errorbar(
                xx,
                s["mean"],
                yerr=s["std"],
                marker="o",
                lw=2,
                capsize=4,
                color="#9467bd",
                label="student (aux only)",
            )
            ax.errorbar(
                [float(v) for v in sx["value"]],
                sx["mean"],
                yerr=sx["std"],
                marker="s",
                ls="--",
                lw=1.5,
                capsize=4,
                color="#c5b0d5",
                label="cross-model control",
            )
            if sw in log_x:
                ax.set_xscale("log")
        else:
            pos = np.arange(len(vals))
            ax.bar(
                pos - 0.2,
                s["mean"],
                0.4,
                yerr=s["std"],
                capsize=4,
                color="#9467bd",
                label="student (aux only)",
            )
            ax.bar(
                pos + 0.2,
                sx["mean"],
                0.4,
                yerr=sx["std"],
                capsize=4,
                color="#c5b0d5",
                label="cross-model control",
            )
            ax.set_xticks(pos)
            ax.set_xticklabels(vals)
        ax.axhline(CHANCE, ls=":", c="black", lw=1.2)
        ax.set_title(titles[sw])
        ax.set_xlabel(xlabels[sw])
        ax.set_ylabel("Val accuracy")
        ax.set_ylim(0, 0.75)
        ax.text(
            0.5,
            0.95,
            notes[sw],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="darkred",
            bbox=dict(boxstyle="round", fc="#fff3cd", ec="darkred", alpha=0.9),
        )
        ax.legend(loc="lower right", fontsize=10)
    axes[-1].axis("off")
    axes[-1].text(
        0.5,
        0.5,
        "One variable changes per panel.\nOther settings remain at baseline.\n3 seeds; error bars show SD.\nDotted line: 10% chance.",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.suptitle("MNIST validation sweeps", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figures/sweeps.png", bbox_inches="tight")
    plt.close(fig)
    print("saved figures/sweeps.png")


# ── mechanism scatter ─────────────────────────────────────────────────────────
def mechanism():
    xs, ys = [], []
    for fp in glob.glob("results/sweep_*.json") + ["results/baseline_5seed.json"]:
        d = json.load(open(fp))
        for seed, rec in d["per_seed"].items():
            cos = rec.get("cos_aux_to_teacher")
            acc = rec.get("student_aux")
            if cos and acc:
                xs += list(cos)
                ys += list(acc)
    xs, ys = np.array(xs), np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(xs, ys, alpha=0.25, s=14, color="#9467bd")
    # trend line
    b, a = np.polyfit(xs, ys, 1)
    xl = np.array([xs.min(), xs.max()])
    ax.plot(xl, a + b * xl, color="red", lw=2.5, label=f"fit (Pearson r = {r:.2f})")
    ax.set_xlabel("Hidden-layer similarity of student to teacher (cosine)")
    ax.set_ylabel("Student accuracy on digits (aux only)")
    ax.set_title("Representation similarity and digit accuracy")
    ax.legend(loc="upper left")
    caption(
        fig,
        "Each point is one trained model pooled across validation sweeps and the test-set baseline. "
        "The fitted association is descriptive and does not by itself establish causality.",
    )
    fig.tight_layout()
    fig.savefig("figures/mechanism_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/mechanism_scatter.png (r={r:.3f})")


if __name__ == "__main__":
    baseline_bars()
    sweeps()
    mechanism()
