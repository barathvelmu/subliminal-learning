# S8 outcome-blinded external transfer validation

**Frozen before opening the released per-animal outcome file.**

Date frozen: 2026-07-18 (America/New_York)

This document fixes an external-validity analysis before reading
`vectors/zoo/llama/results_clean.csv` from Blank et al. (2026). The paper and
its aggregate claim are already public, so this is not a statistically
prospective preregistration. We know that the authors report a relationship
between inference-time steering and student transfer. We have not inspected
the exact per-animal CSV values. The purpose of freezing this plan is narrower:
to prevent those values from changing our definitions of geometry,
observational depth, causal timing, exclusions, or statistical tests.

## External sources and immutable revisions

- Blank et al. code: `agu18dec/steering-vector-distillation`, Git commit
  `89ab3616f6ed0e11a69481c1acd19d37c44e3706`.
- Released code/data archive: `agu18dec/steering_vector_distillation`,
  Hugging Face dataset revision
  `4fda20d0413040b2de61448c89182716485d9839`.
- Outcome file fixed in advance:
  `vectors/zoo/llama/results_clean.csv` at that revision.
- External steering benchmark fixed in advance:
  `vectors/zoo/llama/peaks_clean.json` at that revision.

No alternate revision or manually transcribed figure values will replace these
files after outcomes are opened.

## Scientific question

Blank et al. show that a system-prompt-derived steering vector can predict
which animal traits transfer through student fine-tuning. Our paper separates
three other frozen-model quantities that are often conflated with that result:

1. static animal/number output-row geometry;
2. observational answer readability across depth;
3. causal donor control across depth.

We ask whether any of these independently defined prompt-time quantities tracks
the external, training-time student outcome across the same 16 animal concepts.
This is a difficult external-validity test, not a required confirmation of our
8B/70B scaling result. A null result would mean the frozen number-to-animal
channel studied in our paper should not be advertised as a transfer predictor.

## Fixed model and concepts

We use `unsloth/Meta-Llama-3.1-8B-Instruct`, the full-weight model already used
for our Llama-3.1-8B analyses. The fixed concept order is copied from the
released code, without looking at the outcome CSV:

`dolphin, dog, jellyfish, tiger, octopus, elephant, fox, cat, mouse, lion,
hawk, platypus, wolf, pangolin, falcon, whale`.

All 16 concepts will be retained. Multi-token animal names are represented by
the first answer token for behavioral readouts, matching our existing protocol
and disclosed as a limitation. The number universe is the same 1,110 atomic
one-to-three-character decimal tokens used in our Llama experiments.

## Frozen predictors

All three in-house predictors are collected without student fine-tuning and
without using the external outcome values.

### G: static geometry association

For each animal, compute the Pearson correlation across all 1,110 number tokens
between:

- raw cosine similarity of the animal mean-subtoken output row and number
  output row; and
- the log probability of the animal's first answer token after the fixed
  number-conditioned prompt.

This is exactly the primary per-animal statistic in S3.

### R: observational readout AUC

At every embedding/block endpoint, project the assistant-position residual
state through the model's final normalization and output head. For each animal
and layer, correlate the resulting animal score across the 1,110 number prompts
with the model's final number-conditioned animal log probability. Integrate the
per-animal correlation curve over normalized depth using the trapezoid rule.
This is exactly the S4 assistant-position AUC.

### C: causal donor AUC

Use the existing S5 design: 128 seed-0 unordered atomic number pairs, both
directions, natural assistant-position full-state patching, and requested
relative depths `.25, .50, .75, .90, .97`. At every depth and animal, regress
the patched animal contrast jointly on clean donor and clean recipient animal
contrasts with an intercept. Prepend the analytic zero-depth donor coefficient
and integrate the donor coefficient over measured relative depth, normalized by
the deepest measured point. This is the corrected S5 causal donor AUC.

The animal contrast is computed against the other 15 frozen concepts in this
validation set. The permutation, wrong-concept, identity-patch, duplicate-pass,
and numerical checks remain mandatory.

### S: released steering benchmark

Use `peak_pos_rate` from the fixed external outcome file. This is not our new
predictor. It is a positive-control benchmark and data-integrity check against
Blank et al.'s published claim.

## Fixed outcome and covariate

- Primary external outcome: `sl_rate`, the released trained-student subliminal
  learning rate for each Llama animal.
- Fixed baseline covariate: `base_prior_count`, the released reference-model
  favorite-animal count.

The released code identifies `train_seed=1` for this aggregate. Therefore the
external outcome is a single training-seed estimate per animal; we will not
pretend it measures seed-level training uncertainty.

## Statistical analysis

### Primary test

The primary test is the Spearman rank correlation between **C** and `sl_rate`
across all 16 animals. Report the coefficient, a seed-0 100,000-resample
animal-bootstrap percentile interval, and a two-sided seed-0 100,000-permutation
p-value obtained by permuting outcome labels.

### Fixed secondary tests

Apply the same analysis to **G**, **R**, and the external benchmark **S**.
Apply Benjamini-Hochberg correction to the three in-house predictor permutation
p-values (G, R, C). The benchmark S is reported separately and is not included
in this family.

For descriptive predictor comparison, report paired animal-bootstrap
distributions of the differences `rho(C)-rho(G)`, `rho(C)-rho(R)`, and
`rho(S)-rho(C)`. Do not call one predictor better unless the relevant 95%
interval excludes zero.

### Fixed baseline-prior sensitivity

Rank-transform each predictor and `sl_rate`. Residualize both ranks separately
on `log1p(base_prior_count)` with an intercept, then correlate the two residual
vectors using Pearson correlation. Report this partial-rank sensitivity for G,
R,C, and S. It is a sensitivity analysis, not a second primary endpoint.

### Missingness and exclusions

No animal may be removed for a weak, negative, surprising, or inconvenient
result. If an external row is missing or duplicated, stop and report the data
problem rather than silently changing the set. If an in-house predictor is
numerically undefined, report the cause and do not substitute a different
metric. No transformations other than those fixed above will be selected after
opening outcomes.

## Integration gate

This experiment enters the main paper as a new result only if all of the
following hold:

1. the external files validate against their fixed revisions and contain one
   unique row for every frozen animal;
2. all in-house numerical and causal controls pass unchanged;
3. at least one in-house predictor has positive Spearman correlation whose
   animal-bootstrap 95% interval excludes zero and whose permutation test
   survives BH-FDR at 0.05; and
4. the interpretation remains bounded by the single external training seed and
   shared animal set.

If the gate fails, the result is retained in the project record as a completed
external-validity test, but the six-page main paper remains centered on the
already validated 8B/70B dissociation. No metric or animal subset will be
changed to rescue the result.

## Compute gate

Run the complete analysis locally on Apple MPS first. No paid GPU is authorized
for S8 unless a local numerical regression fails for a clearly device-specific
reason. Existing MPS/CUDA checks already show negligible 8B drift for the S3
and S4 estimands, so a paid repeat is not expected to be scientifically useful.
