"""Entanglement figures from results/. Currently: the cherry-pick chart - bidirectional
entanglement strength per animal, showing the effect is real for some, absent for many."""
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
# outline the paper's animals
for i, n in enumerate(names):
    if n in paper:
        bars[i].set_edgecolor("red"); bars[i].set_linewidth(2.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=40, ha="right")
ax.set_ylabel("Bidirectional entanglement\n(Pearson r: log P(num|love animal) vs log P(animal|love num))")
ax.set_title(f"Token entanglement is real for some animals, absent for many ({data['model'].split('/')[-1]})")
n_sig = sum(sig)
ax.text(0.98, 0.95, f"{n_sig}/{len(names)} significant (p<0.05)\nblue = significant, grey = not\nred outline = paper's animals",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="gray"))
fig.text(0.5, -0.04,
         f"Each bar: correlation across all {data['n_numbers']} number tokens between how much loving the animal boosts a number "
         "and how much loving that number boosts the animal. Positive = bidirectional entanglement. "
         "owl→087 and eagle→747 (the paper's pairs) rank 1st and 2nd; but ~half of animals show no effect — evidence effective animals were selected.",
         ha="center", va="top", fontsize=9.5, style="italic", wrap=True)
fig.tight_layout()
fig.savefig(f"figures/entanglement_by_animal_{tag}.png", bbox_inches="tight")
print(f"saved figures/entanglement_by_animal_{tag}.png ({n_sig}/{len(names)} significant)")
