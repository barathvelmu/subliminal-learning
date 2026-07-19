# S5 preregistration — causal assistant-state transmission across scale

Frozen on 2026-07-18 **before implementation and before any S5 model output was
collected**. Implementation-only smoke outputs are ignored inferentially.

## Motivation

S4 produced a specific mechanism gap. From Llama-3.1 8B to 70B, static output
geometry becomes decisively less predictive, while an observational
assistant-position tuned-logit-lens curve has similar normalized AUC. Linear
readability is not causal use. S5 asks whether the assistant residual state
causally transmits natural animal-specific differences between number prompts,
and whether that transmission changes from 8B to 70B.

This is not a generic “can subliminal representations be steered?” test. Prior
work already answers related versions. The contribution is the matched causal
8B/70B comparison tied directly to the existing dissociation.

## Models and data

- Local smoke: `unsloth/Llama-3.2-1B-Instruct`; two animals, two unordered
  pairs, depths 0.50 and 0.90; ignored scientifically.
- Local gate: `unsloth/Llama-3.1-8B-Instruct`; all 18 animals.
- Paid comparison, only if the gate passes: unchanged full-BF16 CUDA
  Llama-3.1-8B and `unsloth/Meta-Llama-3.1-70B-Instruct` in one environment.
- Stimuli are exactly 256 unique width-3 atomic Llama number strings, selected
  without reference to model outcomes. Sort the common width-3 strings
  lexicographically, apply NumPy seed-0 permutation, take the first 256, and pair
  the first 128 with the next 128. Analyze both directions, producing 256 ordered
  donor→recipient observations.
- Use the identical number pairs, direction order, prompts, animals, batch
  structure, dtype, and requested relative depths at 8B and 70B.

No number is selected because it produces a large animal effect. No pair may be
filtered after viewing its natural or patched effect.

## Intervention

For each model, collect the natural final animal logits for every selected number
prompt. At each selected depth, capture the donor prompt's assistant-position
residual state at the **input to a transformer block**. Run the recipient prompt
and replace only its assistant-position block input with the donor state; every
other token position and all remaining blocks belong to the recipient run.

Fixed requested relative depths are `0.25, 0.50, 0.75, 0.90, 0.97`. Map each to
the nearest valid block input using the model's total block count and record the
actual relative depth. Never patch the post-final-norm or final hidden state.

At the embedding-layer input, the assistant token is identical across prompts,
so donor and recipient assistant states are analytically identical. Add the
fixed point `(relative depth 0, beta 0)` to the AUC without a model run.

The intervention tests causal sufficiency/transmission through the assistant
state. It does not establish necessity, a unique direction, neuron, head, or
circuit.

## Animal-specific outcome

For animal `a` and number prompt `n`, define

```text
z_a(n) = logit_a(n) - mean logit_b(n) over the other 17 animals.
```

For ordered donor `d` and recipient `r`, define the natural total difference

```text
Delta_a(d,r) = z_a(d) - z_a(r).
```

At block-input depth `l`, define the patched effect

```text
delta_a,l(d,r) = z_a(patch donor d assistant state into recipient r at l)
                 - z_a(r).
```

For each animal and depth, regress `delta` on `Delta` across all 256 ordered
pairs with an intercept. The OLS slope is `beta_a,l`.

Interpretation: `beta=0` means the patched assistant state transmits none of the
pairwise natural animal contrast in a linear average sense; `beta=1` means full
transmission. Values outside 0–1 are reported rather than clipped. Pairwise
ratios are forbidden because tiny `Delta` values would be unstable.

## Primary estimand

For each animal, compute normalized causal AUC over the fixed analytic zero and
five actual depth points:

```text
causal_AUC_a = trapezoid(beta_a,l over actual relative depth)
               / maximum actual relative depth.
```

The model summary is the mean over the fixed 18 animals. The primary scale
contrast is paired `70B causal_AUC_a - 8B causal_AUC_a`.

Uncertainty uses a seed-0 crossed cluster bootstrap with 20,000 resamples:
resample the 18 animals and the 128 unordered number-pair clusters with
replacement, always retaining both directions of a sampled pair and the exact
8B/70B pairing. Use percentile 95% intervals. The lower 20,000 count relative to
earlier analyses is frozen for the more expensive crossed recomputation.

Decision language fixed in advance:

- interval entirely above zero: stronger/earlier assistant-state causal
  transmission at 70B;
- interval spans zero: causal scale change unresolved;
- interval entirely below zero: weaker/later assistant-state causal
  transmission at 70B.

Do not infer equivalence from an interval spanning zero. If causal AUC does not
resolve negative while static geometry's preregistered change remains negative,
report that causal transmission did not show the geometry decline, not that the
curves are identical. If causal AUC also declines, report that S4 readability
remained visible even while downstream causal transmission weakened. Neither
outcome is a universal scaling law.

## Fixed controls

1. **Donor permutation.** Apply a fixed seed-1 donor-number derangement that maps
   every ordered observation to a donor different from both its true donor and
   recipient. Patch that donor state while retaining the original `Delta`.
2. **Wrong animal.** Circularly shift the animal label on `Delta` while leaving
   each animal's patched effect unchanged. Report matched-minus-wrong slopes.
3. **Direction halves.** Report slopes separately for the first 128 directions
   and the 128 reversed directions. Both must have the same positive sign for
   the local gate; equality is not required because the network is nonlinear.
4. **Identity numerical control.** On smoke data, patch each recipient's own
   state back into itself at every depth. Selected final logits must match the
   clean recipient run within the fixed backend tolerance.
5. **Clean duplicate.** Repeated unpatched forward passes must match selected
   logits within tolerance.
6. **Nondegeneracy.** `Var(Delta_a)` must exceed `1e-8` for all 18 animals. Abort
   analysis rather than drop an animal if this fails.
7. **Depth ordering.** Report the paired mean change from 0.25 to each of 0.75,
   0.90, and 0.97. This is a control for a genuinely rising path rather than a
   depth-independent prompt swap.

Report all controls regardless of outcome. No control may be redefined after
seeing the full local result.

## Local gate before any paid run

The 8B gate passes only if all are true:

1. identity and duplicate numerical checks pass;
2. all 18 `Delta` variances pass the fixed threshold;
3. mean matched `beta` at both 0.75 and 0.90 has a crossed-bootstrap interval
   entirely above zero;
4. matched-minus-donor-permutation mean beta at both 0.75 and 0.90 has an
   interval entirely above zero;
5. matched-minus-wrong-animal mean beta at 0.90 is positive with an interval
   entirely above zero;
6. the two direction-half mean slopes are positive at both 0.75 and 0.90;
7. the paired 0.90-minus-0.25 mean beta interval is entirely above zero;
8. the result would change an abstract-level sentence if replicated at 70B.

Failure stops paid work. The null/control failure is logged and the existing
paper remains observational.

## Compute and spending rule

No paid instance may launch until the local gate and artifact analyzer pass.
Before launch, independently query live Vast credit and active instances. Reuse
the prior destroy-always safety pattern: fixed price ceiling, local wall clock,
remote watchdog, continuous checkpoint pulls, local validation, and destruction
in `finally`. Never spend merely to exhaust credit.

The historical balance is `$1.72882825921`; live state must be checked again. If
a same-environment full 8B calibration plus 70B run cannot fit with a safety
reserve, stop and report the budget boundary.

## Non-claims

- S5 does not test student fine-tuning or training-time subliminal transfer.
- Full-vector patching does not identify a unique causal feature or circuit.
- A positive slope means causal transmission of natural pairwise output
  differences through the assistant state, not that every difference originates
  there.
- Five relative depths give a coarse causal curve, not layer-perfect
  localization.
- One 8B/70B same-release pair does not establish a general size law.

---

## S5B validation amendment — donor coefficient without shared-recipient coupling

Frozen on 2026-07-18 **after the local MPS 8B validation collection and before
any S5 CUDA/70B collection**. The original estimand above remains reported as a
secondary diagnostic; it is not silently deleted.

### Why an amendment was required

The local validation did exactly what a gate should do: it exposed a mathematical
problem before paid testing. The original variables

```text
delta = patched - recipient
Delta = donor - recipient
```

share the same negative recipient term. Even when the patched donor is randomly
permuted, regressing `delta` on `Delta` can produce a positive slope because both
contain recipient variation. At 8B the donor-permutation slope was therefore
substantially positive. This violates the intended null behavior and makes the
raw slope unsuitable as the paid primary.

This is an algebraic shared-baseline problem, not an outcome-dependent choice of
which biological effect looks largest. No 70B/CUDA S5 output exists. The MPS 8B
run is henceforth the validation set used to repair the estimator.

### Corrected primary coefficient

Do not subtract the recipient. For every animal and depth, fit the multiple OLS
regression across the same fixed 256 ordered observations:

```text
z_a(patch donor d into recipient r at depth l)
  = intercept
    + beta_donor(a,l) * z_a(d)
    + gamma_recipient(a,l) * z_a(r)
    + error.
```

`beta_donor(a,l)` is the corrected causal-transmission coefficient. It asks how
strongly the patched output follows the donor's clean animal contrast while
holding the recipient's clean contrast fixed. `gamma_recipient` is reported as a
secondary path-retention curve. Do not clip either coefficient.

The corrected primary causal AUC replaces `beta` with `beta_donor` but retains
the analytic `(0,0)` point, actual relative depths, trapezoidal normalization,
and seed-0 20,000-resample crossed animal/unordered-pair bootstrap.

The primary paid scale contrast is paired corrected causal AUC at CUDA 70B minus
CUDA 8B. The original shared-baseline AUC and contrast are mandatory secondary
results and must be labeled as estimator-invalid for causal magnitude, useful
only for showing why the validation gate was necessary.

### Corrected controls

1. For the seed-1 permuted-donor patched output, regress on the **original clean
   donor** and recipient. Its donor coefficient should be centered near zero.
2. Wrong-animal control replaces the clean donor predictor with the next
   animal's clean donor contrast while retaining the target animal's recipient
   predictor and patched target outcome.
3. Direction-half, identity, duplicate, nondegeneracy, and depth-rise controls
   are unchanged, but use `beta_donor`.
4. Report the design-matrix condition number for every animal and bootstrap
   degeneracy count. No pair or animal may be removed for collinearity after
   inspecting results; abort if the full-sample condition number exceeds 100.

### Corrected local gate

Re-analyze the already collected MPS 8B validation artifact without another
model forward. The gate passes only if:

- every original numerical and nondegeneracy control passes;
- all full-sample design condition numbers are below 100;
- mean corrected donor coefficient at 0.75 and 0.90 has a crossed-bootstrap 95%
  interval entirely above zero;
- corrected matched-minus-permuted donor coefficient at 0.75 and 0.90 has an
  interval entirely above zero;
- corrected matched-minus-wrong-animal coefficient at 0.90 has an interval
  entirely above zero;
- both direction-half corrected donor coefficients are positive at 0.75 and
  0.90;
- corrected 0.90-minus-0.25 mean coefficient has an interval entirely above
  zero.

Only this amended gate can authorize a paid S5 comparison. The correction and
the observed invalid permutation slope must be disclosed in the lab notebook,
supplement, and any manuscript using S5.

---

## Outcome audit — added after collection

The amended local gate passed. The paid CUDA comparison then ran once on Vast
instance `45267205`, created at `2026-07-18T22:51:30Z` and destroyed at
`2026-07-18T23:21:33Z`. The runner pulled and validated both artifacts before
destruction; a post-run provider query returned zero active instances. Settled
credit moved from `$1.60777967921` to `$0.79508485761`, an observed S5 cost of
`$0.81269482160`.

### Corrected primary result

Mean donor coefficients by actual relative depth:

- 8B depths `0.25, 0.50, 0.75, 0.90625, 0.96875`:
  `0.00116, 0.03845, 0.59235, 0.77990, 0.97384`.
- 70B depths `0.25, 0.50, 0.75, 0.90, 0.975`:
  `0.00729, 0.77280, 0.93801, 0.94821, 0.98436`.

Corrected causal AUC:

- 8B: `0.25389 [0.24147, 0.26612]`;
- 70B: `0.53970 [0.52691, 0.55158]`;
- paired 70B-minus-8B: `+0.28581 [+0.27162, +0.29991]`;
- `18/18` animal-level AUCs increase.

The frozen decision is **stronger or earlier assistant-state causal transmission
at 70B**.

### Controls

- Mean permuted-donor coefficients stay between `-0.0131` and `+0.0006` at 8B,
  and `-0.0024` and `+0.0319` at 70B.
- Mean wrong-animal coefficients stay between `-0.0009` and `+0.0333` at 8B,
  and `-0.0003` and `+0.0062` at 70B.
- Both direction halves are positive at 0.75 and 0.90 in both models.
- Duplicate-forward and identity-patch maximum absolute errors are exactly `0`.
- Maximum standardized condition numbers are `1.177` at 8B and `1.242` at 70B,
  far below 100.
- Crossed-bootstrap degenerate-cell count is `0` for both models.

### Integrity

- 8B CUDA NPZ SHA-256:
  `a9a8acfb11ca4fdc0c396a2e8decf009cb0509ab122cf7b3cb8a3cc7ceaa19f4`
- 70B CUDA NPZ SHA-256:
  `96acfd09720b6d27f8e1dad53e965c66878b614909f55500535bfabcc1ca72c0`
- paired summary SHA-256:
  `af7c4a64b0e6fcd56311bf58118b6942c5ddd904e3b6387c7e6103d9475f829d`
