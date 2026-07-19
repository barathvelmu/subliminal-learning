"""
Decomposition probe - is the entanglement ANIMAL-SPECIFIC (genuine owl<->087 pairing)
or mostly GENERIC (a few number tokens are boosted by ANY animal-love prompt, and those
same numbers boost any animal back — a softmax-bottleneck / salience effect)?

We have, over all animals a and numbers n (chat prompts, instruct, matching the bidirectional experiment):
  M[a,n] = log P(n | "love animal a")     (animal -> number)
  N[n,a] = log P(animal a | "love num n") (number -> animal, the subliminal effect)

Decompose each into a GENERIC part (average over animals) + an animal-SPECIFIC residual:
  gx[n] = mean_a M[a,n]     Sx[a,n] = M[a,n] - gx[n]
  gy[n] = mean_a N[n,a]     Sy[n,a] = N[n,a] - gy[n]

Then per animal compare:
  r_full[a]     = corr_n(M[a,:], N[:,a])           (the Step-2 bidirectional signal)
  r_specific[a] = corr_n(Sx[a,:], Sy[:,a])         (after removing the generic component)
If r_specific << r_full, the apparent entanglement is largely a GENERIC effect.
We also report the top generic numbers (highest gx) and how much of M's variance is generic.
"""

import json
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from scipy import stats

from entanglement import ANIMALS, favorite_animal_probs, target_token
from utils import (
    is_english_num,
    load_model,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)

FLOOR = 1e-12


def main():
    torch.manual_seed(0)
    model, tok = load_model("unsloth/Llama-3.2-1B-Instruct")
    model.eval()
    number_ids, number_strs = [], []
    for tid in range(tok.vocab_size):
        s = tok.decode(tid).strip()
        if is_english_num(s) and len(s) <= 3:
            number_ids.append(tid)
            number_strs.append(s)
    num_arr = torch.tensor(number_ids)
    str_of = dict(zip(number_ids, number_strs))
    animal_ids = {a: target_token(tok, a) for a in ANIMALS}
    print(f"{len(number_ids)} numbers, {len(ANIMALS)} animals")

    # N[n,a] = log P(a | love n)
    N = np.zeros((len(number_ids), len(ANIMALS)))
    for i, nid in enumerate(number_ids):
        p = favorite_animal_probs(
            model, tok, NUMBER_PROMPT_TEMPLATE.format(number=str_of[nid])
        )
        for j, a in enumerate(ANIMALS):
            N[i, j] = np.log(p[animal_ids[a][0]].item() + FLOOR)
        if (i + 1) % 400 == 0:
            print(f"  {i + 1}/{len(number_ids)}")
    # M[a,n] = log P(n | love a)
    M = np.zeros((len(ANIMALS), len(number_ids)))
    for j, a in enumerate(ANIMALS):
        p = favorite_animal_probs(model, tok, ANIMAL_PROMPT_TEMPLATE.format(animal=a))
        M[j] = np.log(p[num_arr].numpy() + FLOOR)

    gx = M.mean(0)  # generic animal->number (per number)
    gy = N.mean(1)  # generic number->animal (per number)
    Sx = M - gx[None, :]  # animal-specific
    Sy = N - gy[:, None]

    r_full, r_spec = [], []
    for j, a in enumerate(ANIMALS):
        r_full.append(stats.pearsonr(M[j], N[:, j])[0])
        r_spec.append(stats.pearsonr(Sx[j], Sy[:, j])[0])
    r_full, r_spec = np.array(r_full), np.array(r_spec)

    # PERMUTATION CONTROL: matched (a's forward-specific vs a's reverse-specific) should
    # beat mismatched (a's forward-specific vs a DIFFERENT animal's reverse-specific) if
    # the cleaned signal is genuine pairwise entanglement, not a centering artifact.
    n_a = len(ANIMALS)
    r_mismatch = []
    for j in range(n_a):
        others = [stats.pearsonr(Sx[j], Sy[:, k])[0] for k in range(n_a) if k != j]
        r_mismatch.append(np.mean(others))
    r_mismatch = np.array(r_mismatch)
    print(
        f"\nPERMUTATION CONTROL: matched r_specific {r_spec.mean():.3f} vs "
        f"mismatched (animal a vs other animals) {r_mismatch.mean():.3f}"
    )
    print(f"  matched > mismatched in {int((r_spec > r_mismatch).sum())}/{n_a} animals")

    # how much of M's variance is the generic (between-number) component
    var_generic = gx.var()
    var_total = M.var()
    frac_generic = var_generic / var_total

    # generic-generic alignment (same for all animals): do generically-boosted numbers
    # also generically boost animals?
    gg = stats.pearsonr(gx, gy)[0]

    print(f"\nmean r_full (bidirectional)      = {r_full.mean():.3f}")
    print(f"mean r_specific (generic removed) = {r_spec.mean():.3f}")
    print(
        f"-> generic component explains {(1 - r_spec.mean() / max(r_full.mean(), 1e-9)) * 100:.0f}% of the mean correlation"
    )
    print(
        f"fraction of animal->number variance that is GENERIC (shared across animals): {frac_generic:.2f}"
    )
    print(
        f"generic-generic alignment corr(gx, gy) = {gg:.3f} (are 'any-animal' numbers also 'boost-any-animal' numbers?)"
    )
    top_generic = [str_of[number_ids[i]] for i in np.argsort(gx)[-10:][::-1]]
    print(f"top generic numbers (boosted by ANY animal-love prompt): {top_generic}")
    print("\nper-animal r_full vs r_specific:")
    for j, a in enumerate(ANIMALS):
        print(f"  {a:10s} full {r_full[j]:+.3f}  specific {r_spec[j]:+.3f}")

    os.makedirs("results", exist_ok=True)
    json.dump(
        {
            "mean_r_full": float(r_full.mean()),
            "mean_r_specific": float(r_spec.mean()),
            "mean_r_mismatch": float(r_mismatch.mean()),
            "frac_generic_variance": float(frac_generic),
            "generic_generic_corr": float(gg),
            "top_generic_numbers": top_generic,
            "per_animal": {
                a: {"r_full": float(r_full[j]), "r_specific": float(r_spec[j])}
                for j, a in enumerate(ANIMALS)
            },
        },
        open("results/decompose_1b.json", "w"),
        indent=2,
    )
    print("saved results/decompose_1b.json")


if __name__ == "__main__":
    main()
