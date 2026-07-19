# Paper workspace

This is the paper-facing side of the repository. The older `scaling/` folder is
the chronological lab notebook; the material here explains the result, places it
against prior work, and records how to reproduce it.

## Where to start

To prepare the submission, open
[`00-START-HERE-FINAL-MAP.md`](00-START-HERE-FINAL-MAP.md). It names the exact
files to upload and the order in which the deadlines arrive.

For the research itself, begin with the
[`current paper`](Submission/AAAI27/output/pdf/aaai27-main.pdf). The
[`zero-background guide`](Learning/zero-background-guide.md) explains every
technical term, while [`frontier-decision.md`](Research/frontier-decision.md)
and the [`novelty matrix`](Research/novelty-matrix.md) explain what is new and
what the experiments do not establish. Short notes on the closest papers live
in [`Research/Papers/`](Research/Papers/).

The [`reproducibility map`](Reproducibility/README.md) is the place to check the
work. Conference-specific files and ready-to-paste metadata are under
[`Submission/AAAI27/`](Submission/AAAI27/).

## Folder map

- `Learning/`: baby-food explanations that assume no background.
- `Research/Papers/`: one condensed Markdown card per important primary paper.
- `Research/novelty-matrix.md`: claim-by-claim comparison with the closest work.
- `Research/frontier-decision.md`: what is already strong, what is missing, and
  the single highest-value next experiment.
- `Manuscript/`: earlier long-form modular workbench; useful for history, but
  not the current submission draft.
- `Reproducibility/`: exact artifact and command map.
- `Supplement/`: preregistrations and audit material that should survive edits to
  the main text.
- `Submission/AAAI27/`: official anonymous AAAI-27 paper, supplement, checklist,
  code/data archive, deadlines, and ready-to-paste metadata.

## One-sentence paper claim

In the frozen subliminal-prompting channel, Llama-3.1-70B has weaker static
animal/number output geometry but earlier donor control in relative depth than
8B, while variable-length tokenization can manufacture a positive-looking
association unless sequence width is controlled.

## What this paper does **not** claim

- It does not claim that token entanglement is the universal cause of
  training-time subliminal learning.
- It does not claim that a tuned logit lens identifies a causal circuit; the
  full-state patch establishes sufficiency, not a unique feature or necessity.
- It does not claim a universal scaling law from one clean 8B/70B release pair.
- It does not claim to be the first work to score multi-token numbers
  autoregressively or to inspect layers. Schrodi et al. already do both.
- It does not treat a confidence interval spanning zero as evidence of no effect.

Those boundaries are part of the contribution: every headline says exactly what
the experiment measured.
