"""
Linear probe: how can the aux-only student beat chance when its digit-readout weights
are random and get NO gradient?

Hypothesis: distillation makes the student's HIDDEN features digit-separable (by
copying the teacher's representation). The frozen random readout can't fully use
them, but a *linear probe* trained on those hidden features should classify digits
well — proving the information is there.

We fit a logistic-regression probe on TRAIN hidden features, evaluate on TEST.
Compare three sources of hidden features:
  - reference (untrained)  -> separability from random init alone (control)
  - student (aux only)     -> after noise/aux distillation
  - teacher                -> upper bound
Also report each model's own (random-readout) accuracy for contrast.
"""
import numpy as np
import torch as t
from sklearn.linear_model import LogisticRegression

from experiment import (Config, load_splits, MultiClassifier, train, distill,
                          accuracy, DEVICE, N_DIGITS)

cfg = Config(n_models=5, seeds=(0,))   # baseline config (KL, width 256, 5 epochs)
t.manual_seed(0); np.random.seed(0)
tr_x1, tr_y1, val_x1, val_y1, te_x1, te_y1 = load_splits()
N = cfg.n_models
sizes = cfg.layer_sizes
aux_idx = list(range(N_DIGITS, cfg.total_out))

train_x = tr_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
distill_x = (t.rand_like(train_x) * 2 - 1)

reference = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
teacher = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
teacher.load_state_dict(reference.state_dict())
train(teacher, train_x, tr_y1, cfg)

student = MultiClassifier(N, sizes, cfg.init_scale).to(DEVICE)
student.load_state_dict(reference.state_dict())
distill(student, teacher, aux_idx, distill_x, cfg)

# probe data: fit on a 3k subset of TRAIN hidden, eval on TEST hidden (no leakage).
# 3k is ample for a 256-dim, 10-class linear probe; keeps the random-feature
# control (which never fully converges) from grinding to the iteration cap.
FIT = 3_000
fit_x = tr_x1[:FIT].unsqueeze(0).expand(N, -1, -1, -1, -1)
fit_y = tr_y1[:FIT].cpu().numpy()
test_x = te_x1.unsqueeze(0).expand(N, -1, -1, -1, -1)
test_y = te_y1.cpu().numpy()


@t.inference_mode()
def hidden_np(model, x):
    return model.hidden(x).cpu().numpy()  # (N, n, width)


def probe(model):
    H_fit = hidden_np(model, fit_x); H_te = hidden_np(model, test_x)
    accs = []
    for m in range(N):
        clf = LogisticRegression(max_iter=150, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(H_fit[m], fit_y)
        accs.append(clf.score(H_te[m], test_y))
    return np.array(accs)


def own_acc(model):
    return np.array(accuracy(model, test_x, t.tensor(test_y, device=DEVICE)))


print("Linear-probe accuracy on hidden features (fit on train, eval on test):")
for name, model in [("reference(untrained)", reference), ("student(aux only)", student), ("teacher", teacher)]:
    p = probe(model)
    o = own_acc(model)
    print(f"  {name:22s} probe={p.mean():.3f}±{p.std(ddof=1):.3f}   own_readout_acc={o.mean():.3f}±{o.std(ddof=1):.3f}")
