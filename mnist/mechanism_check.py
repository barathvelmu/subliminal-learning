"""Test auxiliary-target variation as an explanation for noise transfer.

The script compares one teacher's auxiliary outputs on uniform noise, Gaussian
noise, and validation images. This candidate explanation was rejected: real
images produced stronger auxiliary targets despite transferring less digit
skill.

For each input set it measures:
  - std of raw aux logits across inputs (mean over the 3 aux dims)
  - std of the softmax-over-aux target across inputs (what KL actually matches)
  - mean entropy of that target
"""

import numpy as np
import torch as t

from experiment import Config, load_splits, MultiClassifier, train, DEVICE, N_DIGITS

cfg = Config(n_models=5, seeds=(0,))
t.manual_seed(0)
np.random.seed(0)
tr_x1, tr_y1, val_x1, val_y1, te_x1, te_y1 = load_splits()
N = cfg.n_models
sizes = cfg.layer_sizes
aux_idx = list(range(N_DIGITS, cfg.total_out))

train_x = tr_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
teacher = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
train(teacher, train_x, tr_y1, cfg)

SUB = 5000
real = val_x1[:SUB].unsqueeze(0).expand(N, -1, -1, -1, -1)
unif = t.rand_like(real) * 2 - 1
gauss = t.randn_like(real)


@t.inference_mode()
def signal_stats(name, x):
    logits = teacher(x)[:, :, aux_idx]  # (N, SUB, 3)
    raw_std = logits.std(dim=1).mean().item()  # spread of raw aux logits across inputs
    targets = t.softmax(logits, dim=-1)  # (N, SUB, 3) = what KL matches
    prob_std = targets.std(dim=1).mean().item()  # spread of target across inputs
    ent = (-(targets * targets.clamp_min(1e-9).log()).sum(-1)).mean().item()
    print(
        f"{name:14s} raw_aux_logit_std={raw_std:.4f}  target_prob_std={prob_std:.4f}  mean_entropy={ent:.4f}"
    )
    return raw_std, prob_std, ent


print("Variation in the teacher's auxiliary output across inputs:")
signal_stats("uniform_noise", unif)
signal_stats("gaussian_noise", gauss)
signal_stats("real_images", real)
