"""
Interactive demo: pick any animal and watch token entanglement happen.

Forward direction: we tell the model it loves your animal, ask for its favorite
animal, and look at which NUMBER tokens got boosted in the answer distribution.
Reverse direction: we tell the model it loves each of those numbers and check
whether your animal's probability rises in return.

This is a quick, single-animal taste of the effect. The rigorous version
(all 1110 number tokens, both directions, correlation + significance across 18
pre-registered animals) lives in entanglement.py.

Usage:
    python try_your_animal.py                      # owl, the classic
    python try_your_animal.py --animal platypus    # your call
    python try_your_animal.py --animal fox --top-k 8

Runs fine on CPU (a handful of forward passes of a 1B model; the first run
downloads the model, roughly 2.5 GB).
"""

import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch

from entanglement import favorite_animal_probs, target_token
from utils import (
    load_model,
    is_english_num,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", default="owl")
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="how many top entangled numbers to test in reverse",
    )
    args = ap.parse_args()
    animal = args.animal.lower().strip()

    torch.manual_seed(0)
    model, tok = load_model(args.model)
    model.eval()

    # all 1-3 digit single number tokens in the vocabulary
    number_ids, number_strs = [], []
    for tid in range(tok.vocab_size):
        s = tok.decode(tid).strip()
        if is_english_num(s) and len(s) <= 3:
            number_ids.append(tid)
            number_strs.append(s)
    num_arr = np.array(number_ids)

    animal_id, n_tok = target_token(tok, animal)
    if n_tok > 1:
        print(
            f'note: "{animal}" spans {n_tok} tokens; we track its first token, '
            f"which is standard but slightly noisier.\n"
        )

    # baseline: no persona at all
    base = favorite_animal_probs(model, tok, None)
    base_p_animal = base[animal_id].item()

    # forward: love the animal, see which numbers light up vs baseline
    print(f'forward: telling the model it loves "{animal}"...')
    loved = favorite_animal_probs(
        model, tok, ANIMAL_PROMPT_TEMPLATE.format(animal=animal)
    )
    boost = np.log(loved[num_arr].numpy() + 1e-12) - np.log(
        base[num_arr].numpy() + 1e-12
    )
    order = np.argsort(boost)[::-1]

    print(f'\ntop {args.top_k} number tokens boosted by loving "{animal}":')
    print(f"  {'number':>8s}  {'P(number) went':>22s}")
    for i in order[: args.top_k]:
        print(
            f"  {number_strs[i]:>8s}  {base[num_arr[i]].item():.2e} -> {loved[num_arr[i]].item():.2e}"
            f"  ({np.exp(boost[i]):5.1f}x)"
        )

    # reverse: love each of those numbers, does the animal come back?
    print(
        f"\nreverse: telling the model it loves each number, "
        f'watching P("{animal}") (baseline {base_p_animal:.2e})...'
    )
    hits = 0
    for i in order[: args.top_k]:
        p = favorite_animal_probs(
            model, tok, NUMBER_PROMPT_TEMPLATE.format(number=number_strs[i])
        )
        p_animal = p[animal_id].item()
        ratio = p_animal / max(base_p_animal, 1e-12)
        hits += ratio > 1.0
        print(
            f'  love "{number_strs[i]:>3s}" -> P("{animal}") = {p_animal:.2e}  ({ratio:5.1f}x baseline)'
        )

    print(
        f"\n{hits}/{args.top_k} of the top forward-entangled numbers also boosted "
        f'"{animal}" in reverse.'
    )
    print(
        "caveat: single pairs are noisy and some animals show no effect at all; "
        "the honest measurement is the full bidirectional correlation over all "
        f"{len(number_ids)} numbers (see entanglement.py and the README)."
    )


if __name__ == "__main__":
    main()
