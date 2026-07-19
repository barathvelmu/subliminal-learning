"""
Behavioral subliminal-prompting / token-entanglement experiment (v2).

v1 tested a single hand-picked entangled number against a random-number null. A
bug-review found that biased: (C1) it used a no-system-prompt baseline as the
denominator (confounding "any prompt" with "this number"); (C2) selecting the
number that maximizes P(n|love a)/P(n) preferentially picks RARE tokens that are
weak reverse operators (winner's curse); (C3) the selected n* and the random null
were non-exchangeable, invalidating the permutation test.

v2 uses the paper's actual statistic, which avoids all three. Across ALL number
tokens we measure two directions and correlate them:
    x_n = log P(number n | "you love {animal}")     # animal -> number
    y_n = log P(animal a | "you love {number n}")    # number -> animal
A genuine bidirectional entanglement implies a POSITIVE correlation between x and
y across numbers. Every number is treated identically (no selection, exchangeable,
same prompt structure both ways -> no baseline confound). We also run the paper's
t-test: P(animal | love top-decile entangled numbers) vs bottom-decile.

Pre-registered 18 animals; ALL reported. Loads the model ONCE.
Compute: 1 no-prompt forward + 18 animal-love forwards + |numbers| number-love
forwards (cached, reused across animals).
"""

import argparse
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from scipy import stats
from utils import (
    load_model,
    is_english_num,
    ANIMAL_PROMPT_TEMPLATE,
    NUMBER_PROMPT_TEMPLATE,
)

ANIMALS = [
    "owl",
    "eagle",
    "dolphin",
    "octopus",
    "elephant",
    "wolf",
    "lion",
    "tiger",
    "bear",
    "fox",
    "cat",
    "dog",
    "penguin",
    "panda",
    "koala",
    "peacock",
    "shark",
    "sea turtle",
]

# paper's reported pairs, for a replication sanity-check (descriptive only)
PAPER_PAIRS = {"owl": "087", "eagle": "747", "sea turtle": "321"}
FLOOR = 1e-12


def target_token(tok, word):
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return ids[0], len(ids)


@torch.no_grad()
def favorite_animal_probs(model, tok, system_prompt):
    msgs = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs += [
        {"role": "user", "content": "What is your favorite animal?"},
        {"role": "assistant", "content": "My favorite animal is the"},
    ]
    prompt = tok.apply_chat_template(
        msgs, continue_final_message=True, add_generation_prompt=False, tokenize=False
    )
    inp = tok(prompt, return_tensors="pt").to(model.device)
    return model(**inp).logits[0, -1].float().softmax(-1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    ap.add_argument(
        "--max-numbers",
        type=int,
        default=10**9,
        help="cap #number tokens (for smoke tests)",
    )
    ap.add_argument("--tag", default="1b")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, tok = load_model(args.model)
    model.eval()

    # all 1-3 digit single number tokens
    number_ids, number_strs = [], []
    for tid in range(tok.vocab_size):
        s = tok.decode(tid).strip()
        if is_english_num(s) and len(s) <= 3:
            number_ids.append(tid)
            number_strs.append(s)
    number_ids = number_ids[: args.max_numbers]
    number_strs = number_strs[: args.max_numbers]
    num_arr = np.array(number_ids)
    str_of = dict(zip(number_ids, number_strs))
    print(f"{len(number_ids)} number tokens")

    animal_ids = {a: target_token(tok, a) for a in args.animals}

    # number -> animal: love each number, record P(animal) for all animals (cache, reused)
    print("number->animal forwards (cached)...")
    P_a_given_n = np.zeros((len(number_ids), len(args.animals)))  # [num, animal]
    for i, nid in enumerate(number_ids):
        p = favorite_animal_probs(
            model, tok, NUMBER_PROMPT_TEMPLATE.format(number=str_of[nid])
        )
        for j, a in enumerate(args.animals):
            P_a_given_n[i, j] = p[animal_ids[a][0]].item()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(number_ids)}")

    base = favorite_animal_probs(model, tok, None)

    results = []
    for j, a in enumerate(args.animals):
        a_id, a_ntok = animal_ids[a]
        # animal -> number: love animal, P(number) over all numbers
        ap_probs = favorite_animal_probs(
            model, tok, ANIMAL_PROMPT_TEMPLATE.format(animal=a)
        )
        x = np.log(ap_probs[num_arr].numpy() + FLOOR)  # log P(n | love a)
        y = np.log(P_a_given_n[:, j] + FLOOR)  # log P(a | love n)

        r, p_r = stats.pearsonr(x, y)
        sr, p_sr = stats.spearmanr(x, y)
        # paper's t-test: top-decile vs bottom-decile of x (entanglement), compare y
        k = max(2, len(x) // 10)
        order = np.argsort(x)
        y_bottom, y_top = y[order[:k]], y[order[-k:]]
        t, p_t = stats.ttest_ind(y_top, y_bottom, equal_var=False)

        top_idx = order[-5:][::-1]
        top_entangled = [str_of[number_ids[i]] for i in top_idx]
        pair = PAPER_PAIRS.get(a)
        pair_rank = None
        if pair is not None:
            # match the paper's number whether stored as "087" or stripped "87"
            cand = (
                pair
                if pair in number_strs
                else (str(int(pair)) if str(int(pair)) in number_strs else None)
            )
            if cand is not None:
                pi = number_strs.index(cand)
                pair_rank = int((x > x[pi]).sum()) + 1  # rank by entanglement (1=top)

        results.append(
            {
                "animal": a,
                "n_tokens": a_ntok,
                "multi_token": a_ntok > 1,
                "base_p_animal": base[a_id].item(),
                "pearson_r": float(r),
                "pearson_p": float(p_r),
                "spearman_r": float(sr),
                "spearman_p": float(p_sr),
                "ttest_t": float(t),
                "ttest_p": float(p_t),
                "top_entangled_numbers": top_entangled,
                "paper_pair": pair,
                "paper_pair_entanglement_rank": pair_rank,
            }
        )
        msg = (
            f"{a:10s} r={r:+.3f}(p={p_r:.1e})  t={t:+.2f}(p={p_t:.1e})  "
            f"top_nums={top_entangled[:3]}"
        )
        if pair:
            msg += f"  [paper {pair}: rank {pair_rank}/{len(number_ids)}]"
        print(msg)

    os.makedirs("results", exist_ok=True)
    out = f"results/entanglement_{args.tag}.json"
    json.dump(
        {"model": args.model, "n_numbers": len(number_ids), "results": results},
        open(out, "w"),
        indent=2,
    )
    n_sig = sum(rr["pearson_r"] > 0 and rr["pearson_p"] < 0.05 for rr in results)
    print(
        f"\n{n_sig}/{len(results)} animals show significant POSITIVE bidirectional correlation (r>0, p<0.05)"
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
