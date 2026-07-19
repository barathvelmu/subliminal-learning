"""Measure transfer and representation similarity along a noise dose-response.

The earlier auxiliary-signal explanation was rejected. This analysis
interpolates distillation inputs from real images toward noise:
    distill_input = real_image + alpha * uniform(-1, 1)
It sweeps alpha from 0 (pure real) upward and adds pure noise as a reference.

For each setting measure, with error bars over seeds:
  - auxiliary-only student evaluation accuracy
  - linear CKA between student and teacher final hidden representations

Parallel increases in accuracy and CKA support, but do not prove, the proposed
input-coverage explanation.

The baseline configuration uses KL loss, width 256, and five epochs to match the
noise-sweep result being investigated.
"""

import json
import numpy as np
import torch as t

from experiment import (
    Config,
    load_splits,
    MultiClassifier,
    train,
    distill,
    accuracy,
    DEVICE,
    N_DIGITS,
)

cfg = Config(n_models=8, seeds=(0, 1, 2))
splits = load_splits()
tr_x1, tr_y1, val_x1, val_y1, te_x1, te_y1 = splits
sizes = cfg.layer_sizes
aux_idx = list(range(N_DIGITS, cfg.total_out))
N = cfg.n_models

# eval on validation (this is exploratory mechanism work, test stays clean)
eval_x = val_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
eval_y = val_y1

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]  # real + alpha*noise
INCLUDE_PURE_NOISE = True


def linear_cka(X, Y):
    """Linear CKA between two (n, d) activation matrices. Scale/rotation invariant,
    a standard representation-similarity metric. Returns scalar in [0, 1]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).pow(2).sum()
    norm = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return (hsic / norm).item()


@t.inference_mode()
def cka_to_teacher(student, teacher, x):
    hs = student.hidden(x)
    ht = teacher.hidden(x)  # (N, n, width)
    return [linear_cka(hs[m], ht[m]) for m in range(N)]


def distill_inputs(real_base, alpha, pure_noise):
    if pure_noise:
        return t.rand_like(real_base) * 2 - 1
    return real_base + alpha * (t.rand_like(real_base) * 2 - 1)


def run():
    conditions = [("alpha", a, False) for a in ALPHAS]
    if INCLUDE_PURE_NOISE:
        conditions.append(("pure_noise", None, True))

    # accumulate per-seed seed-means
    acc = {c[:2]: [] for c in conditions}
    cka = {c[:2]: [] for c in conditions}

    for seed in cfg.seeds:
        t.manual_seed(seed)
        np.random.seed(seed)
        train_x = tr_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
        reference = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
        teacher = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
        teacher.load_state_dict(reference.state_dict())
        train(teacher, train_x, tr_y1, cfg)  # teacher independent of distill data

        real_base = tr_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
        for label, alpha, pure in conditions:
            src = distill_inputs(real_base, alpha or 0.0, pure)
            student = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
            student.load_state_dict(reference.state_dict())
            distill(student, teacher, aux_idx, src, cfg)
            acc[(label, alpha)].append(np.mean(accuracy(student, eval_x, eval_y)))
            cka[(label, alpha)].append(
                np.mean(cka_to_teacher(student, teacher, eval_x))
            )

    print(f"{'condition':14s} {'student_aux(val)':>18s} {'CKA(student,teacher)':>22s}")
    out = []
    for label, alpha, _ in conditions:
        a = np.array(acc[(label, alpha)])
        k = np.array(cka[(label, alpha)])
        name = "pure_noise" if label == "pure_noise" else f"real+{alpha}*noise"
        print(
            f"{name:14s} {a.mean():.3f} ± {a.std(ddof=1):.3f}      "
            f"{k.mean():.3f} ± {k.std(ddof=1):.3f}"
        )
        out.append(
            {
                "condition": name,
                "alpha": alpha,
                "acc_mean": float(a.mean()),
                "acc_std": float(a.std(ddof=1)),
                "cka_mean": float(k.mean()),
                "cka_std": float(k.std(ddof=1)),
            }
        )
    json.dump(out, open("results/noise_investigation.json", "w"), indent=2)
    print("saved results/noise_investigation.json")


if __name__ == "__main__":
    run()
