"""Compare unembedding metrics as predictors of token entanglement.

Metrics for an (animal a, number n) pair, from the unembedding matrix U = lm_head.weight:
  - raw_cos      cos(U_a, U_n)                      # paper Eq 1
  - cen_cos      cos(U_a - Ubar, U_n - Ubar)        # centered alternative
  - cen_dot      (U_a - Ubar) . (U_n - Ubar)        # most logit-faithful

Why centering: a logit is L_n = U_n . h. Writing U_n = Ubar + (U_n - Ubar), the
term Ubar.h is identical for every token -> a constant shift -> softmax-invariant.
Only (U_n - Ubar) can affect probabilities, so raw cosine includes an irrelevant
component. (U_a is the mean unembedding over a multi-token animal's sub-tokens.)

Each metric is evaluated against behavior in both directions:
  - forward  x_n = log P(number n | "you love {animal}")   (animal -> number)
  - reverse  y_n = log P(animal a | "you love {number n}") (number -> animal == the
             subliminal prompting effect)
Per animal we correlate each metric (over all numbers) with x and with y; we report
the mean correlation across the 18 pre-registered animals for each (metric, direction).
"""

import json
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from entanglement import ANIMALS, favorite_animal_probs, target_token
from utils import is_english_num, ANIMAL_PROMPT_TEMPLATE, NUMBER_PROMPT_TEMPLATE
from utils import load_model

FLOOR = 1e-12


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    ap.add_argument("--max-numbers", type=int, default=10**9)
    ap.add_argument("--tag", default="1b")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, tok = load_model(args.model)
    model.eval()

    U = model.lm_head.weight.detach().float().cpu()  # [vocab, hidden]
    Ubar = U.mean(0)
    Uc = U - Ubar

    number_ids, number_strs = [], []
    for tid in range(tok.vocab_size):
        s = tok.decode(tid).strip()
        if is_english_num(s) and len(s) <= 3:
            number_ids.append(tid)
            number_strs.append(s)
    number_ids = number_ids[: args.max_numbers]
    number_strs = number_strs[: args.max_numbers]
    num_arr = torch.tensor(number_ids)
    str_of = dict(zip(number_ids, number_strs))
    print(f"{len(number_ids)} number tokens; U {tuple(U.shape)}")

    animal_ids = {a: target_token(tok, a) for a in args.animals}

    # reverse: y[n, a] = P(animal a | love number n)  (the subliminal effect)
    print("number->animal forwards...")
    Y = np.zeros((len(number_ids), len(args.animals)))
    for i, nid in enumerate(number_ids):
        p = favorite_animal_probs(
            model, tok, NUMBER_PROMPT_TEMPLATE.format(number=str_of[nid])
        )
        for j, a in enumerate(args.animals):
            Y[i, j] = p[animal_ids[a][0]].item()
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(number_ids)}")

    # forward: x[a, n] = P(number n | love animal a)
    X = {}
    for a in args.animals:
        p = favorite_animal_probs(model, tok, ANIMAL_PROMPT_TEMPLATE.format(animal=a))
        X[a] = p[num_arr].numpy()

    Un = U[num_arr]  # [Nnum, hidden]
    Ucn = Uc[num_arr]
    metrics = ["raw_cos", "cen_cos", "raw_dot", "cen_dot"]
    per_animal = []
    agg = {m: {"forward": [], "reverse": []} for m in metrics}

    for j, a in enumerate(args.animals):
        a_ids = tok(" " + a, add_special_tokens=False).input_ids
        U_a = U[a_ids].mean(0)
        Uc_a = Uc[a_ids].mean(0)
        raw_cos = F.cosine_similarity(U_a.unsqueeze(0), Un, dim=1).numpy()
        cen_cos = F.cosine_similarity(Uc_a.unsqueeze(0), Ucn, dim=1).numpy()
        raw_dot = (Un @ U_a).numpy()
        cen_dot = (Ucn @ Uc_a).numpy()

        xf = np.log(X[a] + FLOOR)  # forward behavioral
        yr = np.log(Y[:, j] + FLOOR)  # reverse behavioral (subliminal)
        row = {"animal": a, "multi_token": animal_ids[a][1] > 1}
        for m, vals in [
            ("raw_cos", raw_cos),
            ("cen_cos", cen_cos),
            ("raw_dot", raw_dot),
            ("cen_dot", cen_dot),
        ]:
            rf = stats.pearsonr(vals, xf)[0]
            rr = stats.pearsonr(vals, yr)[0]
            row[f"{m}_forward"] = float(rf)
            row[f"{m}_reverse"] = float(rr)
            agg[m]["forward"].append(rf)
            agg[m]["reverse"].append(rr)
        per_animal.append(row)

    print(
        f"\nMean correlation of each metric with the behavioral effect (n={len(args.animals)} animals):"
    )
    print(f"{'metric':10s} {'forward (a->n)':>16s} {'reverse (n->a, subliminal)':>28s}")
    summary = {}
    for m in metrics:
        f_mean, f_sd = np.mean(agg[m]["forward"]), np.std(agg[m]["forward"])
        r_mean, r_sd = np.mean(agg[m]["reverse"]), np.std(agg[m]["reverse"])
        summary[m] = {
            "forward_mean": float(f_mean),
            "forward_sd": float(f_sd),
            "reverse_mean": float(r_mean),
            "reverse_sd": float(r_sd),
        }
        print(f"{m:10s} {f_mean:>10.3f}±{f_sd:.2f} {r_mean:>20.3f}±{r_sd:.2f}")

    # paired test: does centered cosine beat raw cosine per-animal? (both directions)
    for direction in ["forward", "reverse"]:
        raw = np.array(agg["raw_cos"][direction])
        cen = np.array(agg["cen_cos"][direction])
        t, p = stats.wilcoxon(cen, raw)
        print(
            f"[{direction}] centered vs raw cosine, paired Wilcoxon: p={p:.4f} "
            f"(centered higher in {int((cen > raw).sum())}/{len(raw)} animals)"
        )

    os.makedirs("results", exist_ok=True)
    out = f"results/metrics_{args.tag}.json"
    json.dump(
        {"model": args.model, "summary": summary, "per_animal": per_animal},
        open(out, "w"),
        indent=2,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
