"""
Make readable MNIST figures from results/.

Outputs to figures/:
  baseline_bars.png      6-condition baseline with seed error bars
  sweeps.png             accuracy vs each swept knob (annotated)
  mechanism_scatter.png  hidden-similarity-to-teacher vs student accuracy

Design: large fonts, control + chance reference lines, an arrow/circle on the
key feature, and a one-line caption under each figure.
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
    ax.set_title("Subliminal learning on MNIST: student learns digits from pure noise")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    # circle the surprising result
    i = 2
    ell = Ellipse((i, means[i]), 0.9, 0.22, fill=False, edgecolor="red", lw=2.5)
    ax.add_patch(ell)
    ax.annotate(
        "trained ONLY on noise +\nteacher's junk logits,\nyet reads digits",
        xy=(i, means[i] + 0.11),
        xytext=(i + 0.1, means[i] + 0.30),
        fontsize=11,
        color="red",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    # mark the dead control
    j = 4
    ax.annotate(
        "control: break shared start\n→ effect gone",
        xy=(j, means[j] + 0.02),
        xytext=(j - 0.4, means[j] + 0.28),
        fontsize=11,
        color="black",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )
    caption(
        fig,
        "Bars = mean over 5 seeds, error bars = std across seeds. The aux-only student "
        "(circled) beats chance using no digit labels; the cross-model control confirms a shared start is required.",
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
        "loss": "MSE > KL here (opposite of paper 2's story)",
        "noise": "uniform ≈ gaussian; real images much worse",
        "distill_epochs": "0 epochs = no effect; keeps rising with training",
        "init_scale": "effect peaks near default scale",
        "width": "wider → weaker (matches paper 2 / NTK)",
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
        "Each panel: vary ONE knob,\nrest at baseline.\n3 seeds, error bars = std.\nDotted line = 10% chance.\nGold box = what to notice.",
        ha="center",
        va="center",
        fontsize=13,
    )
    fig.suptitle(
        "What makes subliminal learning stronger or weaker? (MNIST sweeps)", fontsize=18
    )
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
    ax.set_title(
        "The mechanism: the more the student's 'thinking' matches\nthe teacher's, the better it reads digits"
    )
    ax.legend(loc="upper left")
    ax.annotate(
        "each dot = one model;\nclear upward trend",
        xy=(np.quantile(xs, 0.8), a + b * np.quantile(xs, 0.8)),
        xytext=(np.quantile(xs, 0.45), ys.max() * 0.55),
        fontsize=12,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    caption(
        fig,
        "Every dot is one trained model, pooled across all sweep runs (val) and the baseline (test). Higher hidden-layer "
        "alignment to the teacher tracks higher digit accuracy — evidence the student copies the teacher's internal representation.",
    )
    fig.tight_layout()
    fig.savefig("figures/mechanism_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/mechanism_scatter.png (r={r:.3f})")


if __name__ == "__main__":
    baseline_bars()
    sweeps()
    mechanism()
