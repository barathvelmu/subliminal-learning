"""
Are the entangled pairs inherited from PRETRAINING (present in the
base model) or created by instruction-tuning?

Models: unsloth/Llama-3.2-1B (base) vs unsloth/Llama-3.2-1B-Instruct (same tokenizer/vocab).

Part A — GEOMETRY (cheap, NO prompts -> no template confound). Lead with this.
  From each model's unembedding U=lm_head.weight, compute centered-cosine entanglement
  between every animal and every number. Compare base vs instruct: are the paper's pairs
  (owl-087, eagle-747) high in the BASE model, and is the whole structure preserved?

Part B — BEHAVIORAL, MATCHED PLAIN-COMPLETION prompts (identical text for both models,
  NO chat template / system role for either):
    baseline:        "My favorite animal is the"
    animal->number:  "I love owl. I think about owl all the time. My favorite animal is the"
    number->animal:  "I love 087. I think about 087 all the time. My favorite animal is the"
  Run the same x_n vs y_n bidirectional correlation per animal for both models.

CAVEAT (report, do not bury): even with identical text, instruction-tuning changes how a
model ENGAGES with a prompt, so Part B is EVIDENCE, not proof. Part A is the cleaner test.
"""
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from entanglement import ANIMALS, target_token
from utils import is_english_num, load_model

BASE = "unsloth/Llama-3.2-1B"
INSTRUCT = "unsloth/Llama-3.2-1B-Instruct"
PAPER_PAIRS = {"owl": "087", "eagle": "747", "sea turtle": "321"}
FLOOR = 1e-12


@torch.no_grad()
def plain_probs(model, tok, love_text):
    text = (f"I love {love_text}. I think about {love_text} all the time. " if love_text else "")
    text += "My favorite animal is the"
    inp = tok(text, return_tensors="pt").to(model.device)
    return model(**inp).logits[0, -1].float().softmax(-1).cpu()


def behavioral(model, tok, animals, number_ids, str_of, animal_ids, num_arr):
    Y = np.zeros((len(number_ids), len(animals)))
    for i, nid in enumerate(number_ids):
        p = plain_probs(model, tok, str_of[nid])
        for j, a in enumerate(animals):
            Y[i, j] = p[animal_ids[a][0]].item()
        if (i + 1) % 400 == 0:
            print(f"    n->a {i+1}/{len(number_ids)}")
    X = {a: plain_probs(model, tok, a)[num_arr].numpy() for a in animals}
    out = {}
    for j, a in enumerate(animals):
        x = np.log(X[a] + FLOOR); y = np.log(Y[:, j] + FLOOR)
        r, p = stats.pearsonr(x, y)
        top = [str_of[number_ids[i]] for i in np.argsort(X[a])[-5:][::-1]]
        out[a] = {"r": float(r), "p": float(p), "top_numbers": top}
    return out, X


def geometry(U, animals, number_ids, str_of, animal_ids, num_arr, tok):
    Ubar = U.mean(0); Uc = U - Ubar
    Ucn = Uc[num_arr]
    g = {}
    for a in animals:
        a_ids = tok(" " + a, add_special_tokens=False).input_ids
        Uc_a = Uc[a_ids].mean(0)
        cen = F.cosine_similarity(Uc_a.unsqueeze(0), Ucn, dim=1).numpy()
        g[a] = cen
    return g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-numbers", type=int, default=10**9)
    ap.add_argument("--animals", nargs="+", default=ANIMALS)
    args = ap.parse_args()
    animals = args.animals

    # ---- INSTRUCT ----
    print("loading instruct...")
    m, tok = load_model(INSTRUCT); m.eval()
    number_ids, number_strs = [], []
    for tid in range(tok.vocab_size):
        s = tok.decode(tid).strip()
        if is_english_num(s) and len(s) <= 3:
            number_ids.append(tid); number_strs.append(s)
    number_ids = number_ids[:args.max_numbers]; number_strs = number_strs[:args.max_numbers]
    num_arr = torch.tensor(number_ids); str_of = dict(zip(number_ids, number_strs))
    animal_ids = {a: target_token(tok, a) for a in animals}
    print(f"{len(number_ids)} numbers")

    U_inst = m.lm_head.weight.detach().float().cpu()
    print("instruct behavioral (plain prompts)...")
    beh_inst, _ = behavioral(m, tok, animals, number_ids, str_of, animal_ids, num_arr)
    geo_inst = geometry(U_inst, animals, number_ids, str_of, animal_ids, num_arr, tok)
    del m; torch.cuda.empty_cache()

    # ---- BASE ----
    print("loading base...")
    mb, tokb = load_model(BASE); mb.eval()
    U_base = mb.lm_head.weight.detach().float().cpu()
    print("base behavioral (plain prompts)...")
    beh_base, _ = behavioral(mb, tokb, animals, number_ids, str_of, animal_ids, num_arr)
    geo_base = geometry(U_base, animals, number_ids, str_of, animal_ids, num_arr, tokb)
    del mb; torch.cuda.empty_cache()

    # ---- compare ----
    print("\n=== GEOMETRY base vs instruct (centered cosine over numbers) ===")
    geo_corr = [stats.pearsonr(geo_base[a], geo_inst[a])[0] for a in animals]
    pooled = stats.pearsonr(np.concatenate([geo_base[a] for a in animals]),
                            np.concatenate([geo_inst[a] for a in animals]))[0]
    print(f"per-animal base~instruct geometry corr: mean {np.mean(geo_corr):.3f} (min {np.min(geo_corr):.3f})")
    print(f"pooled base~instruct geometry corr: {pooled:.3f}")
    for a, n in PAPER_PAIRS.items():
        cand = n if n in number_strs else str(int(n))
        if cand in number_strs:
            ni = number_strs.index(cand)
            rb = int((geo_base[a] > geo_base[a][ni]).sum()) + 1
            ri = int((geo_inst[a] > geo_inst[a][ni]).sum()) + 1
            print(f"  {a}-{n}: geometry rank base {rb}/{len(number_ids)}, instruct {ri}/{len(number_ids)}")

    print("\n=== BEHAVIORAL base vs instruct (plain prompts, bidirectional r) ===")
    rb = np.array([beh_base[a]["r"] for a in animals])
    ri = np.array([beh_inst[a]["r"] for a in animals])
    print(f"mean r: base {rb.mean():.3f}, instruct {ri.mean():.3f}")
    print(f"base-r vs instruct-r across animals corr: {stats.pearsonr(rb, ri)[0]:.3f}")
    sig_b = sum(beh_base[a]["p"] < 0.05 and beh_base[a]["r"] > 0 for a in animals)
    sig_i = sum(beh_inst[a]["p"] < 0.05 and beh_inst[a]["r"] > 0 for a in animals)
    print(f"significant animals: base {sig_b}/{len(animals)}, instruct {sig_i}/{len(animals)}")
    for a in [x for x in ["owl", "eagle"] if x in animals]:
        print(f"  {a}: base r={beh_base[a]['r']:+.3f} top{beh_base[a]['top_numbers'][:3]} | "
              f"instruct r={beh_inst[a]['r']:+.3f} top{beh_inst[a]['top_numbers'][:3]}")

    os.makedirs("results", exist_ok=True)
    json.dump({"geometry": {"per_animal_corr": {a: float(c) for a, c in zip(animals, geo_corr)}, "pooled": float(pooled)},
               "behavioral_base": beh_base, "behavioral_instruct": beh_inst},
              open("results/base_vs_instruct.json", "w"), indent=2)
    print("saved results/base_vs_instruct.json")


if __name__ == "__main__":
    main()
