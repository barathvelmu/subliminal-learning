"""Plot bidirectional entanglement by animal from saved results."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
os.makedirs("figures", exist_ok=True)

tag = sys.argv[1] if len(sys.argv) > 1 else "1b"
data = json.load(open(f"results/entanglement_{tag}.json"))
res = sorted(data["results"], key=lambda r: r["pearson_r"], reverse=True)

names = [r["animal"] for r in res]
rs = [r["pearson_r"] for r in res]
sig = [r["pearson_p"] < 0.05 and r["pearson_r"] > 0 for r in res]
paper = {"owl", "eagle", "sea turtle"}

colors = ["#2c7fb8" if s else "#bdbdbd" for s in sig]
fig, ax = plt.subplots(figsize=(11, 5.5))
bars = ax.bar(range(len(names)), rs, color=colors)
# Outline the examples reported in the source paper.
for i, n in enumerate(names):
    if n in paper:
        bars[i].set_edgecolor("red")
        bars[i].set_linewidth(2.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylabel(
    "Bidirectional entanglement\n(Pearson r: log P(num|love animal) vs log P(animal|love num))"
)
ax.set_title(
    f"Bidirectional token entanglement varies across animals ({data['model'].split('/')[-1]})"
)
n_sig = sum(sig)
ax.text(
    0.98,
    0.95,
    f"{n_sig}/{len(names)} positive, p<0.05 (uncorrected)\nblue = meets threshold, gray = does not\nred outline = published examples",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(boxstyle="round", fc="#fff3cd", ec="gray"),
)
fig.text(
    0.5,
    -0.04,
    f"Each bar is the correlation across {data['n_numbers']} number tokens between the animal-to-number and "
    "number-to-animal log probabilities. The published owl→087 and eagle→747 examples rank first and second; "
    "several other animals have values near zero.",
    ha="center",
    va="top",
    fontsize=9.5,
    style="italic",
    wrap=True,
)
fig.tight_layout()
fig.savefig(f"figures/entanglement_by_animal_{tag}.png", bbox_inches="tight")
print(
    f"saved figures/entanglement_by_animal_{tag}.png ({n_sig}/{len(names)} significant)"
)
