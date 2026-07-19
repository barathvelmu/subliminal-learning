"""
Experiment harness: subliminal learning on MNIST.

One configurable runner so every experiment is a config change, run over
multiple seeds with proper train/val/test splits.


Usage:
    python experiment.py                 # baseline, default config, seeds 0-4
    python experiment.py --loss mse      # override any config field
    python experiment.py --eval-split val --seeds 0 1 2   # tuning on val

Each run writes a JSON of per-seed/per-condition accuracies to results/
and (optionally) a bar plot to figures/.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch as t
from torch import nn
from torchvision import datasets, transforms

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
N_DIGITS = 10  # fixed: MNIST has 10 classes
SPLIT_SEED = 1234  # fixed seed for the train/val split (independent of model seed)
VAL_SIZE = 10_000  # held-out from the 60k MNIST train set


# ───────────────────────────── config ────────────────────────────────────────
@dataclass
class Config:
    n_models: int = 25
    seeds: tuple = (0, 1, 2, 3, 4)
    width: int = 256
    depth: int = 2  # number of hidden layers
    n_aux: int = 3  # auxiliary logits
    epochs_teacher: int = 5
    epochs_distill: int = 5
    lr: float = 3e-4
    batch: int = 1024
    loss: str = "kl"  # kl | mse | kl_temp
    temp: float = 2.0  # temperature for loss=kl_temp
    noise: str = "uniform"  # uniform | gaussian | train_images
    init_scale: float = 1.0  # multiplier on weight-init std
    eval_split: str = "test"  # val (tuning) | test (final reporting)
    tag: str = "baseline"

    @property
    def total_out(self):
        return N_DIGITS + self.n_aux

    @property
    def layer_sizes(self):
        return [28 * 28] + [self.width] * self.depth + [self.total_out]


# ───────────────────────────── modules ───────────────────────────────────────
class MultiLinear(nn.Module):
    def __init__(self, n_models, d_in, d_out, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(t.empty(n_models, d_out, d_in))
        self.bias = nn.Parameter(t.zeros(n_models, d_out))
        nn.init.normal_(self.weight, 0.0, init_scale / math.sqrt(d_in))

    def forward(self, x):
        return t.einsum("moi,mbi->mbo", self.weight, x) + self.bias[:, None, :]

    def get_reindexed(self, idx):
        _, d_out, d_in = self.weight.shape
        new = MultiLinear(len(idx), d_in, d_out)
        new.weight.data = self.weight.data[idx].clone()
        new.bias.data = self.bias.data[idx].clone()
        return new


def mlp(n_models, sizes, init_scale):
    layers = []
    for i, (d_in, d_out) in enumerate(zip(sizes, sizes[1:])):
        layers.append(MultiLinear(n_models, d_in, d_out, init_scale))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MultiClassifier(nn.Module):
    def __init__(self, n_models, sizes, init_scale=1.0):
        super().__init__()
        self.layer_sizes = sizes
        self.init_scale = init_scale
        self.net = mlp(n_models, sizes, init_scale)

    def forward(self, x):
        return self.net(x.flatten(2))

    def hidden(self, x):
        """Activations at the last hidden layer (input to final linear)."""
        h = x.flatten(2)
        for layer in self.net[:-1]:
            h = layer(h)
        return h

    def get_reindexed(self, idx):
        new = MultiClassifier(len(idx), self.layer_sizes, self.init_scale)
        new_layers = [
            layer.get_reindexed(idx) if hasattr(layer, "get_reindexed") else layer
            for layer in self.net
        ]
        new.net = nn.Sequential(*new_layers)
        return new


# ───────────────────────────── data ──────────────────────────────────────────
class PreloadedDataLoader:
    def __init__(self, inputs, labels, t_bs, shuffle=True):
        self.x, self.y = inputs, labels
        self.M, self.N = inputs.shape[:2]
        self.bs, self.shuffle = t_bs, shuffle
        self._mkperm()

    def _mkperm(self):
        base = t.arange(self.N, device=self.x.device)
        self.perm = (
            t.stack([base[t.randperm(self.N)] for _ in range(self.M)])
            if self.shuffle
            else base.expand(self.M, -1)
        )

    def __iter__(self):
        self.ptr = 0
        if self.shuffle:
            self._mkperm()
        return self

    def __next__(self):
        if self.ptr >= self.N:
            raise StopIteration
        idx = self.perm[:, self.ptr : self.ptr + self.bs]
        self.ptr += self.bs
        batch_x = t.stack([self.x[m].index_select(0, idx[m]) for m in range(self.M)], 0)
        if self.y is None:
            return (batch_x,)
        batch_y = t.stack([self.y.index_select(0, idx[m]) for m in range(self.M)], 0)
        return batch_x, batch_y

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs


def load_splits():
    """Return (train_x, train_y, val_x, val_y, test_x, test_y) on DEVICE.
    Val is a fixed 10k held out of the 60k MNIST train set (seed SPLIT_SEED)."""
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    root = "~/.pytorch/MNIST_data/"
    train_ds = datasets.MNIST(root, download=True, train=True, transform=tfm)
    test_ds = datasets.MNIST(root, download=True, train=False, transform=tfm)

    def to_tensor(ds):
        xs, ys = zip(*ds)
        return t.stack(xs).to(DEVICE), t.tensor(ys, device=DEVICE)

    tr_x, tr_y = to_tensor(train_ds)
    te_x, te_y = to_tensor(test_ds)
    g = t.Generator().manual_seed(SPLIT_SEED)
    perm = t.randperm(tr_x.shape[0], generator=g)
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    return (tr_x[train_idx], tr_y[train_idx], tr_x[val_idx], tr_y[val_idx], te_x, te_y)


def make_distill_inputs(cfg, train_x_single, n_models):
    """Distillation inputs, broadcast to n_models. Shape (M, N, 1, 28, 28)."""
    base = train_x_single.unsqueeze(0).expand(n_models, -1, -1, -1, -1)
    if cfg.noise == "train_images":
        return base
    if cfg.noise == "uniform":
        return t.rand_like(base) * 2 - 1
    if cfg.noise == "gaussian":
        return t.randn_like(base)
    raise ValueError(f"unknown noise: {cfg.noise}")


# ───────────────────────────── train / distill ──────────────────────────────
def ce_first10(logits, labels):
    return nn.functional.cross_entropy(
        logits[..., :N_DIGITS].flatten(0, 1), labels.flatten()
    )


def train(model, x, y, cfg):
    opt = t.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs_teacher):
        for bx, by in PreloadedDataLoader(x, y, cfg.batch):
            loss = ce_first10(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()


def distill_loss(out, tgt, cfg):
    if cfg.loss == "kl":
        return nn.functional.kl_div(
            nn.functional.log_softmax(out, -1),
            nn.functional.softmax(tgt, -1),
            reduction="batchmean",
        )
    if cfg.loss == "kl_temp":
        T = cfg.temp
        return nn.functional.kl_div(
            nn.functional.log_softmax(out / T, -1),
            nn.functional.softmax(tgt / T, -1),
            reduction="batchmean",
        ) * (T * T)
    if cfg.loss == "mse":
        return nn.functional.mse_loss(out, tgt)
    raise ValueError(f"unknown loss: {cfg.loss}")


def distill(student, teacher, idx, src_x, cfg):
    opt = t.optim.Adam(student.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs_distill):
        for (bx,) in PreloadedDataLoader(src_x, None, cfg.batch):
            with t.no_grad():
                tgt = teacher(bx)[:, :, idx]
            out = student(bx)[:, :, idx]
            loss = distill_loss(out, tgt, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()


@t.inference_mode()
def accuracy(model, x, y):
    return (model(x)[..., :N_DIGITS].argmax(-1) == y).float().mean(1).tolist()


@t.inference_mode()
def hidden_cosine_to_teacher(student, teacher, x):
    """Mean cosine similarity between student and teacher last-hidden activations,
    per model. x shape (M, N, 1, 28, 28)."""
    hs = student.hidden(x)
    ht = teacher.hidden(x)
    cos = nn.functional.cosine_similarity(hs, ht, dim=-1)  # (M, N)
    return cos.mean(1).tolist()


def derangement(n):
    """Random permutation with NO fixed points. Used for the cross-model control
    so that no student is ever paired with its own (shared-init) teacher — a plain
    randperm leaves ~1 fixed point on average, which leaks the real effect into
    the control and inflates it above chance."""
    if n < 2:
        raise ValueError("derangement needs n>=2")
    while True:
        p = t.randperm(n)
        if not bool((p == t.arange(n)).any()):
            return p


# ───────────────────────────── one seed ──────────────────────────────────────
def run_seed(cfg, seed, splits):
    tr_x1, tr_y1, val_x1, val_y1, te_x1, te_y1 = splits
    t.manual_seed(seed)
    np.random.seed(seed)
    N = cfg.n_models
    sizes = cfg.layer_sizes
    aux_idx = list(range(N_DIGITS, cfg.total_out))
    all_idx = list(range(cfg.total_out))

    train_x = tr_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
    train_y = tr_y1
    eval_x1, eval_y = (val_x1, val_y1) if cfg.eval_split == "val" else (te_x1, te_y1)
    eval_x = eval_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
    distill_x = make_distill_inputs(cfg, tr_x1, N)

    reference = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
    teacher = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())
    train(teacher, train_x, train_y, cfg)

    def fresh_student():
        s = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
        s.load_state_dict(reference.state_dict())
        return s

    student_g, student_a = fresh_student(), fresh_student()
    perm = derangement(N)
    xmodel_g, xmodel_a = student_g.get_reindexed(perm), student_a.get_reindexed(perm)

    distill(student_g, teacher, aux_idx, distill_x, cfg)
    distill(student_a, teacher, all_idx, distill_x, cfg)
    distill(xmodel_g, teacher, aux_idx, distill_x, cfg)
    distill(xmodel_a, teacher, all_idx, distill_x, cfg)

    return {
        "reference": accuracy(reference, eval_x, eval_y),
        "teacher": accuracy(teacher, eval_x, eval_y),
        "student_aux": accuracy(student_g, eval_x, eval_y),
        "student_all": accuracy(student_a, eval_x, eval_y),
        "xmodel_aux": accuracy(xmodel_g, eval_x, eval_y),
        "xmodel_all": accuracy(xmodel_a, eval_x, eval_y),
        "cos_aux_to_teacher": hidden_cosine_to_teacher(student_g, teacher, eval_x),
    }


# ───────────────────────────── aggregate ─────────────────────────────────────
def run(cfg):
    splits = load_splits()
    per_seed = {s: run_seed(cfg, s, splits) for s in cfg.seeds}

    # seed-level summary: each seed -> mean over its N models, then mean/std across seeds
    conditions = [
        "reference",
        "teacher",
        "student_aux",
        "student_all",
        "xmodel_aux",
        "xmodel_all",
        "cos_aux_to_teacher",
    ]
    rows = {}
    for c in conditions:
        seed_means = np.array([np.mean(per_seed[s][c]) for s in cfg.seeds])
        rows[c] = {
            "mean": float(seed_means.mean()),
            "std": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "n_seeds": len(seed_means),
        }
    return rows, per_seed


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    for k, v in asdict(cfg).items():
        if k == "seeds":
            ap.add_argument("--seeds", type=int, nargs="+", default=list(v))
        elif isinstance(v, bool):
            ap.add_argument(
                f"--{k.replace('_', '-')}", type=lambda s: s == "True", default=v
            )
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    cfg = Config(
        **{k: (tuple(v) if k == "seeds" else v) for k, v in vars(args).items()}
    )

    print("Config:", asdict(cfg))
    rows, per_seed = run(cfg)
    df = pd.DataFrame(rows).T
    print(df.to_string())

    os.makedirs("results", exist_ok=True)
    out = f"results/{cfg.tag}.json"
    with open(out, "w") as f:
        json.dump(
            {"config": asdict(cfg), "summary": rows, "per_seed": per_seed}, f, indent=2
        )
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
