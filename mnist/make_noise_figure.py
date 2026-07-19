"""Dose-response figure for A3: as distillation inputs get noisier (real -> noise),
both transfer accuracy and representation similarity (CKA) rise together."""

import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "figure.dpi": 150})

data = json.load(open("results/noise_investigation.json"))
labels = [
    d["condition"]
    .replace("real+", "+")
    .replace("*noise", "n")
    .replace("+0.0n", "pure real")
    for d in data
]
x = list(range(len(data)))
acc = [d["acc_mean"] for d in data]
acc_e = [d["acc_std"] for d in data]
cka = [d["cka_mean"] for d in data]
cka_e = [d["cka_std"] for d in data]

fig, ax1 = plt.subplots(figsize=(9, 5.5))
c1, c2 = "#9467bd", "#2ca02c"
ax1.errorbar(x, acc, yerr=acc_e, marker="o", lw=2.5, capsize=4, color=c1)
ax1.set_ylabel("Student digit accuracy (aux only)", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right")
ax1.set_xlabel("Distillation input: real images → more noise added →")
ax1.set_ylim(0, 0.6)

ax2 = ax1.twinx()
ax2.errorbar(x, cka, yerr=cka_e, marker="s", ls="--", lw=2.5, capsize=4, color=c2)
ax2.set_ylabel("Representation similarity to teacher (CKA)", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.set_ylim(0.7, 0.9)

ax1.set_title(
    "Noise vs real images: as inputs broaden, transfer and\nrepresentation-copying rise together (correlational)"
)
ax1.annotate(
    "both rise together\nas inputs get noisier",
    xy=(3, acc[3]),
    xytext=(0.6, 0.42),
    fontsize=12,
    color="black",
    arrowprops=dict(arrowstyle="->", lw=1.8),
)
fig.text(
    0.5,
    -0.04,
    "Distillation inputs interpolated from real MNIST toward uniform noise. Both axes are means over 3 seeds (error bars = std). "
    "Accuracy (purple) and representation similarity (green, CKA) rise together — consistent with (not proof of) the noise advantage operating via fuller representation copying.",
    ha="center",
    va="top",
    fontsize=10,
    style="italic",
    wrap=True,
)
fig.tight_layout()
fig.savefig("figures/noise_dose_response.png", bbox_inches="tight")
print("saved figures/noise_dose_response.png")
