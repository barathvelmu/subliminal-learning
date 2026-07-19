# Paper workspace

> **Final handoff:** start with
> [`00-START-HERE-FINAL-MAP.md`](00-START-HERE-FINAL-MAP.md). It gives the exact
> reading order, four upload files, live rules, deadlines, after-submission
> timeline, and stop rule in plain English.

This folder is the clean, paper-facing view of the project. The older `scaling/`
folder remains the chronological lab notebook; this folder tells a reader what the
work means, how it differs from prior work, and how to reproduce it.

## Read in this order

1. **Finish the submission:** [`00-START-HERE-FINAL-MAP.md`](00-START-HERE-FINAL-MAP.md)
2. **Read the current paper:** [`Submission/AAAI27/output/pdf/aaai27-main.pdf`](Submission/AAAI27/output/pdf/aaai27-main.pdf)
3. **Start from zero:** [`Learning/zero-background-guide.md`](Learning/zero-background-guide.md)
4. **Understand the solo-author odds:** [`Learning/solo-aaai-reality-check.md`](Learning/solo-aaai-reality-check.md)
5. **See the scientific claim:** [`Research/frontier-decision.md`](Research/frontier-decision.md)
6. **Audit novelty:** [`Research/novelty-matrix.md`](Research/novelty-matrix.md)
7. **Read one-page literature cards:** [`Research/Papers/`](Research/Papers/)
8. **Reproduce the numbers:** [`Reproducibility/README.md`](Reproducibility/README.md)
9. **Prepare the conference submission:** [`Submission/AAAI27/README.md`](Submission/AAAI27/README.md)

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
