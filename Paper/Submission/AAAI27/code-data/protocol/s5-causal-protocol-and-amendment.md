# S5 causal protocol and validation-stage estimator amendment

## Status of this record

This anonymous copy preserves the scientific design and the estimator amendment
used for the submitted S5 analysis. It is a recorded project protocol, not an
immutable third-party preregistration. The intervention design, number/pair
selection, depths, controls, and decision language were specified before S5
model output. The conditional estimator was selected after a local MPS 8B
validation exposed a failed permutation null and before matched CUDA 8B or 70B
causal output was collected. The invalid estimator remains described below.
The submitted repository does not provide an independently verifiable
pre-CUDA commit timestamp, so this chronology is author-reported provenance,
not cryptographic evidence of timing.

## Scientific question

Static output geometry weakens from Llama-3.1 8B to 70B, while an observational
assistant-position output-head trace has no resolved AUC change. S5 asks whether
the assistant residual state transmits natural animal-specific differences
between number prompts and whether that intervention-specific transmission
changes across the matched 8B/70B pair.

S5 does not test student fine-tuning, identify a unique causal feature or
circuit, establish necessity, or support a universal scaling law.

## Fixed models and stimuli

- Headline models: full-BF16 CUDA Llama-3.1-8B-Instruct and
  Llama-3.1-70B-Instruct in one software environment.
- Concepts: the same fixed 18 animals used elsewhere in the paper.
- Numbers: sort the common width-three atomic Llama decimal strings
  lexicographically, apply a NumPy seed-0 permutation, and take the first 256.
- Pairs: pair the first 128 selected numbers with the second 128 and analyze
  both directions, producing 256 ordered donor-to-recipient observations.
- The exact number pairs, direction order, prompts, animals, batch structure,
  dtype, and requested relative depths are shared across the two models.
- No number or pair is selected from a natural or patched model outcome.

## Intervention

For each model, collect natural final animal logits for every selected number
prompt. At each requested depth, capture the donor prompt's final
assistant-position residual state at the input to a transformer block. Run the
recipient prompt and replace only that assistant-position block input with the
donor state. Every other token position and all subsequent computation remain
those of the recipient run.

Requested relative depths are 0.25, 0.50, 0.75, 0.90, and 0.97, mapped to the
nearest valid block input and recorded at their actual relative depth. The
post-final-norm state is never patched. Because the assistant input embedding is
identical across number prompts, the normalized AUC includes the analytic point
at depth zero with donor coefficient zero.

Natural donor vectors avoid constructing an artificial subspace direction, but
the resulting donor-state/recipient-context hybrid is not guaranteed to be
on-manifold. The estimand is therefore intervention-specific donor control.

## Animal-specific contrast

For animal `a` and number prompt `n`, define:

```text
z_a(n) = logit_a(n) - mean logit_b(n) over the other 17 animals.
```

The contrast tests animal specificity. It does not make the 18 derived concept
statistics independent. Uncertainty is interpreted only for the fixed concept
and pair design, and a raw-target-logit sensitivity removes this contrast.

## Original estimator and validation failure

The original design regressed:

```text
delta = patched - recipient
Delta = donor - recipient
delta on Delta
```

Local MPS 8B validation showed that this estimator failed its donor-permutation
null. Both variables share the negative recipient term, so a positive slope can
appear even when donor assignments are deranged. The estimator is invalid for
causal magnitude and was not used as the paid primary.

## Recorded estimator amendment

The repair fits, for every animal and depth:

```text
z_a(patch donor d into recipient r at depth l)
  = intercept
    + beta_donor(a,l) * z_a(d)
    + gamma_recipient(a,l) * z_a(r)
    + error.
```

`beta_donor` measures clean-donor dependence while holding the recipient's clean
contrast fixed. `gamma_recipient` measures retained recipient dependence.
Neither coefficient is clipped.

For each animal, the primary causal AUC is the trapezoidal integral of
`beta_donor` over the analytic zero and five actual relative-depth points,
divided by the largest actual depth. The primary contrast is the paired 70B
minus 8B animal-level AUC difference.

Uncertainty uses a seed-0 crossed cluster bootstrap with 20,000 resamples. It
resamples the 18 animals and 128 unordered number-pair clusters, retains both
directions of each selected cluster, and preserves the paired 8B/70B structure.

## Fixed controls and sensitivities

1. Donor-number derangement while retaining the original donor predictor.
2. Circular shifts of the donor animal label, including all 17 wrong concepts.
3. Separate analysis of the two pair directions.
4. Identity patch and duplicate clean-forward numerical checks.
5. Nondegeneracy and standardized design condition numbers.
6. Raw target logits without the target-minus-other-animal contrast.
7. Leave-one-unordered-pair-cluster-out sensitivity.
8. One exact comparison with eight transformer blocks remaining.

All controls are reported regardless of direction. The original subtraction
slope is retained only as an invalid diagnostic explaining the amendment.

## Decision language

- A paired causal-AUC interval entirely above zero supports stronger or earlier
  intervention-specific donor control at 70B.
- An interval spanning zero leaves the cross-scale change unresolved.
- An interval entirely below zero supports weaker or later donor control.

No interval is used to claim equivalence. Five depths form a coarse profile,
not exact onset localization. The matched model pair does not establish a model
family scaling law.
