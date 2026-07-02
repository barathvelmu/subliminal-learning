"""
Mechanism check (NOT a main sweep): does the teacher's auxiliary ("junk") output
carry a richer / more varied signal on random noise than on real images?

If yes, it explains why distilling on noise transfers MORE digit skill than
distilling on real images: more variation across inputs => more to copy.

Trains one teacher (default config), then on three input sets (uniform noise,
gaussian noise, real val images) measures how much the teacher's aux-logit
targets VARY across inputs:
  - std of raw aux logits across inputs (mean over the 3 aux dims)
  - std of the softmax-over-aux target across inputs (what KL actually matches)
  - mean entropy of that target (low + constant => little signal)
"""
import numpy as np
import torch as t

from experiment import (Config, load_splits, MultiClassifier, train, DEVICE, N_DIGITS)

cfg = Config(n_models=5, seeds=(0,))
t.manual_seed(0); np.random.seed(0)
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
    logits = teacher(x)[:, :, aux_idx]                 # (N, SUB, 3)
    raw_std = logits.std(dim=1).mean().item()          # spread of raw aux logits across inputs
    targets = t.softmax(logits, dim=-1)                # (N, SUB, 3) = what KL matches
    prob_std = targets.std(dim=1).mean().item()        # spread of target across inputs
    ent = (-(targets * targets.clamp_min(1e-9).log()).sum(-1)).mean().item()
    print(f"{name:14s} raw_aux_logit_std={raw_std:.4f}  target_prob_std={prob_std:.4f}  mean_entropy={ent:.4f}")
    return raw_std, prob_std, ent


print("How much does the teacher's junk output vary across inputs? (higher = richer signal)")
signal_stats("uniform_noise", unif)
signal_stats("gaussian_noise", gauss)
signal_stats("real_images", real)
