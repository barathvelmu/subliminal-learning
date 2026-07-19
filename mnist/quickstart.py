"""Small CPU demonstration of digit transfer through noise distillation.

We train a teacher on MNIST, then train a student that never sees a single
digit. It only sees random noise, and only gets to match the teacher's three
auxiliary outputs (extra logits that have nothing to do with digits). The
student and teacher share a random initialization. A control student starting
from a different initialization remains at chance under the same training.

The four-model result varies slightly across machines and library versions. The
full 25-model, five-seed experiment is implemented in ``experiment.py``.

Usage:
    python quickstart.py
"""

import torch as t

from experiment import (
    Config,
    load_splits,
    MultiClassifier,
    train,
    distill,
    accuracy,
    derangement,
    make_distill_inputs,
    DEVICE,
    N_DIGITS,
)


def main():
    cfg = Config(
        n_models=4,
        seeds=(0,),
        width=128,
        loss="mse",
        epochs_teacher=2,
        epochs_distill=4,
        tag="quickstart",
    )
    t.manual_seed(0)

    print("loading MNIST (downloads ~10 MB on first run)...")
    tr_x, tr_y, _, _, te_x, te_y = load_splits()
    N = cfg.n_models
    train_x = tr_x.unsqueeze(0).expand(N, -1, -1, -1, -1)
    eval_x = te_x.unsqueeze(0).expand(N, -1, -1, -1, -1)
    aux_idx = list(range(N_DIGITS, cfg.total_out))

    # The teacher and student share one random initialization.
    reference = MultiClassifier(N, cfg.layer_sizes, cfg.init_scale).to(DEVICE)
    teacher = MultiClassifier(N, cfg.layer_sizes, cfg.init_scale).to(DEVICE)
    teacher.load_state_dict(reference.state_dict())

    print(f"training the teacher on real MNIST ({cfg.epochs_teacher} epochs)...")
    train(teacher, train_x, tr_y, cfg)

    student = MultiClassifier(N, cfg.layer_sizes, cfg.init_scale).to(DEVICE)
    student.load_state_dict(reference.state_dict())
    # Control: same student, but its initialization is swapped to a different
    # teacher's (derangement = no student keeps its own teacher)
    control = student.get_reindexed(derangement(N))

    noise_x = make_distill_inputs(cfg, tr_x, N)
    print(
        f"distilling the student on noise, matching only {cfg.n_aux} "
        f"auxiliary logits ({cfg.epochs_distill} epochs)..."
    )
    distill(student, teacher, aux_idx, noise_x, cfg)
    print("distilling the control (different initialization) the same way...")
    distill(control, teacher, aux_idx, noise_x, cfg)

    def acc(model):
        return 100 * sum(accuracy(model, eval_x, te_y)) / N

    print(f"""
results (MNIST test set, chance = 10%):
  teacher, trained on 50k real digits:        {acc(teacher):5.1f}%
  untrained network (the shared init):        {acc(reference):5.1f}%
  student, trained on noise:                  {acc(student):5.1f}%   <- subliminal learning
  control, different init, same training:     {acc(control):5.1f}%   <- near chance

The student receives no digit examples or gradient through its digit readout,
yet its accuracy is above chance. The different-initialization control remains
near chance. experiment.py runs the full version (25 models, 5 seeds, and the
reported sweeps).""")


if __name__ == "__main__":
    main()
