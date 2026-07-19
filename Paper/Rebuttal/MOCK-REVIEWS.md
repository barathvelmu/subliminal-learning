# Mock AAAI reviews and arbitration

These are hostile simulations, not official reviews and not an official AAAI
score scale.

## Simulated reviewer A: correctness, statistics, causality

- Score before this rebuttal repair pass: **5/10, borderline weak reject**.
- Confidence: **4/5**.
- Positive view: central causal effect is large and controls are unusually
  extensive; no numerical failure was found.
- Main concerns: unequal number universes, concept-coupled contrast, estimator
  provenance, hybrid-patch interpretation, one model pair, and training-time
  disconnect.

Audit consequence: unequal-universe and raw-logit attacks were tested with
submitted arrays and now appear in the main paper, supplement, analyzer, and
summary JSON. Estimator provenance language and artifact record were corrected.
Hybrid/on-manifold, breadth, and training-time limits remain genuine.

## Simulated reviewer B: novelty, significance, clarity

- Score before this rebuttal repair pass: **5/10, borderline weak reject**.
- Confidence: **4/5**.
- Positive view: technically careful, honest, readable, and reproducible.
- Main concerns: no student training, methods individually known, no neutral-
  task baseline, only two scale points, Qwen confounds, and understated closest
  work.

Audit consequence: Table 1 now accurately strengthens the Blank/Morgulis and
Madl comparisons and includes the positional-bias paper. The external null now
reads as a direct scientific boundary. No neutral-task baseline or broader
model family was invented; those remain limitations.

## Simulated reviewer C: submission and reproducibility

- Simulated score: **6/10, weak accept**.
- Confidence: **4/5**.
- Positive view: current files are anonymous, within limits, self-contained,
  and unusually auditable.
- Main concerns: one scale pair, coarse depths, failed external predictor,
  estimator amendment, missing checkpoint revisions, three omitted non-headline
  arrays, and no code license.

Audit consequence: live conference rules were verified; the S5 protocol is now
inside the artifact; README claims were synchronized. Checkpoint revisions and
license status remain candid.

## Main-agent arbitration

### Accepted and fixed

- prior-work table understated real novelty threats;
- “transparency-only” sounded defensive;
- 1,110-versus-256 stimulus scope needed explicit disclosure and a sensitivity;
- raw-logit robustness needed a crossed interval, not only a point estimate;
- “timestamped” exceeded the submitted provenance evidence;
- the hybrid patch needed an explicit on-manifold caveat;
- external steering wording needed to name causal AUC as the comparison;
- README and supplement regeneration claims needed exact boundaries.

### Rejected or corrected

- A reported duplicate main-paper sentence was not present in the current main
  source or PDF. No deletion was made there.
- A reported duplicated supplement sentence was also checked in source and PDF
  and occurred once. The real issue was defensive phrasing, which was revised.
- The story review's phrase “one unchanged stimulus set” was too broad. The
  paper now distinguishes the full probe universe from the causal subset and
  reports exact-subset results.
- The story review's draft response called the amendment “timestamped.” That is
  not supportable from the anonymous Git history and was replaced with
  “recorded” plus an explicit provenance limitation.

### Genuine limitations left in place

- no student fine-tuning in the central experiments;
- no neutral-task causal timing baseline;
- one same-release model pair;
- five measured causal depths;
- full-state/hybrid rather than feature-level intervention;
- no tokenizer-only Qwen isolation;
- fixed 18-concept design;
- no immutable checkpoint revisions; and
- one aggregate external training seed per animal.

## Final risk assessment

The remaining plausible rejection is a reviewer judgment that the exact gap is
too narrow or too far from training-time transfer. The audit found no basis for
a soundness reject on the central numerical result. The paper is now stronger
because it answers the repairable attacks in the submitted evidence and admits
the unrepairable scope limits without pretending they disappeared.
