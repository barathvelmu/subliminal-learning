# S4 pre-registration — tokenization and depth after the 70B reversal

Frozen: 2026-07-18, before collecting any S4 result.

## Why these tests

S3 found a clean within-release result: Llama-3.1-8B and Llama-3.1-70B retain
behavioral number→animal entanglement, but the correlation between that behavior
and static output-token geometry falls decisively at 70B. The next question is
therefore not merely whether entanglement exists. It is where the channel lives
when static unembedding geometry stops explaining it.

The primary-literature audit found:

- Zur et al., *Token Entanglement in Subliminal Learning*, propose global token
  entanglement and explicitly leave multi-token sequences open.
- Schrodi et al., *Towards Understanding Subliminal Learning* (arXiv:2509.23886),
  show global entanglement is not necessary for training-time transfer and locate
  signal in sparse divergence tokens/early trainable layers.
- Blank et al., *Subliminal Learning Is Steering Vector Distillation*
  (arXiv:2606.00995), show system-prompt traits and student changes can be mediated
  by residual-stream directions.
- Morgulis & Hewitt, *Subliminal Steering* (arXiv:2604.25783), localize transferred
  steering vectors to manipulated layers.
- Brockers et al., *Learning Through Noise* (arXiv:2605.23645), identify compatible
  output heads and non-monotonic capacity effects in a controlled MLP setting.

We found no existing experiment that (a) repairs the number probe for a tokenizer
that splits numbers into digit sequences and compares full-sequence versus
first-token scores, or (b) traces this pre-existing prompting channel layer by
layer through a 70B LLM. This is a documented search result, not an absolute
priority claim.

## S4A — multi-token sequence entanglement (local first)

### Question

Does bidirectional animal↔number entanglement survive when a tokenizer represents
three-digit numbers as multiple tokens?

### Models and order

1. Validate code on Llama-3.2-1B using a small ignored smoke slice.
2. Run full Qwen3-0.6B, already cached locally.
3. Only if informative, add one larger Qwen3 point locally; do not spend Vast
   credit for S4A.

### Number set and prompts

- Fixed strings are every zero-padded decimal string of width 1, 2, and 3:
  10 + 100 + 1000 = 1,110 strings.
- Prompts and 18 animals remain exactly the v2 templates.
- Forward animal→number score is the autoregressive sum of log probabilities for
  the complete target string `number` with no prepended separator, matching the
  original atomic-token probe's direct next-token target exactly.
- Each conditional token term preserves v2's numerical definition
  `log(softmax_probability + 1e-12)` rather than substituting `log_softmax`.
- Reverse number→animal score remains the first animal-token log probability under
  the exact number system prompt.
- A trie/prefix implementation may reuse computation but must produce exactly the
  same sum as naive teacher-forced autoregressive scoring.

### Pre-registered primary

Restrict to the 1,000 width-3 strings, eliminating sequence-length confounding.
For each animal, correlate complete-sequence forward log probability with reverse
log probability across strings. Report all animals, mean r, seed-0 100,000-sample
across-animal bootstrap 95% CI, raw positive p<.05 count, and positive BH-FDR
q<.05 count.

### Required controls and fixed secondary analyses

1. **First-token proxy:** repeat using only the first target-token log probability.
   Report the paired full-sequence minus first-token change with bootstrap CI. This
   directly tests whether sequence scoring rescues information discarded by the
   original atomic-token probe.
2. **Animal specificity:** for each animal's reverse behavior, compare its matched
   forward sequence score against the mean correlation produced by the other 17
   animals' forward scores; bootstrap matched-minus-mismatched.
3. Report widths 1, 2, and 3 separately.
4. Report all-width sum-logp, per-token mean-logp, and within-width standardized
   correlations as secondary only.
5. Report target token-count distributions. Never compare raw sum-logp across
   widths without controlling width.

### Decision language

- Positive primary CI and positive full-minus-first CI: multi-token sequence
  entanglement survives and sequence composition adds signal.
- Positive primary CI but unresolved full-minus-first: channel survives, but the
  first token already carries most measurable signal.
- Primary CI spans zero: unresolved at this model/scale.
- Primary CI below zero: reversed association, reported without rescue language.

One Qwen model cannot establish family universality.

**Pre-result implementation amendment (2026-07-18):** the first ignored smoke
showed that prepending a space adds a separate Llama token and therefore would not
regress to the original atomic-token statistic. No inferential result was computed.
The target was corrected from `" " + number` to `number`; the rerun must match both
naive teacher-forced sequence scoring and the original atomic selected-logp function
within `1e-5` before S4A proceeds.

**Pre-result numerical amendment (2026-07-18):** ignored smoke 2 correctly halted
because `log_softmax` differed by 0.0011 from v2's clipped `log(softmax+1e-12)`
for low-probability tail tokens. The sequence scorer now applies the original
clipped definition at every autoregressive token. No complete S4 artifact or
inferential statistic was produced before this correction.

## S4B — layerwise contextual localization (local gate, then 70B)

### Question

If static number/animal unembedding geometry weakens at 70B, does the animal-specific
signal emerge through contextual computation later in the residual stream?

### Models and gate

1. Implement and smoke-test on Llama-3.2-1B.
2. Run a local Llama-3.1-8B slice, then full 8B only if the trajectory is stable.
3. Rent 70B only if the collector reproduces final logits and produces a coherent,
   non-degenerate local depth curve. Use remaining Vast credit, one instance,
   full BF16, hard credit/time limits, pull-first/destroy-always.

### Data and readouts

- Use the same 1,110 number strings/tokens, 18 animals, prompts, and saved reverse
  behavior matrix as S3.
- At every hidden-state layer, extract residual vectors at:
  1. the final assistant position, and
  2. the last subtoken of each of the three number mentions in the system prompt,
     averaged across mentions (also save the last mention separately).
- Apply the model's final RMSNorm and selected animal unembedding rows on the
  output head's native device/dtype (see pre-result numerical amendment below)
  to produce a tuned-logit-lens score for every layer × number × animal.
- Save raw scores, positions, layer count, exact model/runtime metadata, and a
  final-layer numerical regression against the model's selected logits.

### Pre-registered primary

At the **assistant position**, for each layer and animal, correlate logit-lens
animal score across numbers with the saved final reverse log probability. Summarize
mean r across animals and bootstrap CI as a function of relative depth.

Primary scale contrast uses two scalar summaries:

1. normalized area under the mean-r depth curve (AUC), and
2. earliest relative depth reaching 50% of the model's final-layer mean r.

Compare CUDA 70B versus CUDA 8B with paired across-animal bootstrap for AUC.
Earlier/higher AUC means the behavior is linearly readable through more of the
network; later/lower AUC means it is assembled later.

### Required controls and fixed secondary analyses

1. **Exact final-layer check:** selected final-layer logits must agree with a normal
   forward pass within a predeclared numerical tolerance (max absolute difference
   <= 5e-3 in BF16; tighter values reported).
2. **Animal-specific contrast:** use matched animal score minus mean of the other
   17 animal scores and the analogous final-log-probability contrast. This is
   invariant to a shared softmax denominator.
3. Report mean/last system-number-position depth curves. These test whether the
   association is already written into the contextualized number representation
   or appears only where the answer is produced.
4. Report per-animal curves and leave-one-animal-out sensitivity.
5. Report absolute layer and relative-depth parameterizations; interpolate only
   for model-size comparison, never to invent intermediate observations.

### Decision language

- 70B assistant AUC lower/later than 8B while final behavior persists: evidence
  that scale shifts the channel toward later contextual computation.
- 70B system-number curve remains strong while static geometry falls: evidence
  for contextual re-encoding at the number position.
- All 70B curves weaken: evidence for a more distributed/nonlinearly readable
  channel, not proof of absence.
- Similar curves: static-geometry reversal is localized to the output head rather
  than the residual computation.

Layerwise readout is correlational. Causal patching would be a follow-up and must
not be implied by these results.

**Pre-result numerical amendment (2026-07-18):** ignored local smoke 2 showed
that applying the final norm/unembedding on CPU creates a backend matrix-multiply
delta of up to 0.047 logits even though the same selected-row multiplication on
the model's native MPS backend is exact. It also confirmed that Transformers'
last returned hidden state is already final-normalized. Before any layerwise
correlation was computed, the collector was changed to apply the same final
RMSNorm formula and selected unembedding rows on the output head's native device
and dtype, skipping re-normalization only for the already-normalized final state.
This preserves the frozen readout/statistics while making the <=5e-3 final-logit
regression meaningful on both MPS and CUDA. Smoke 2 is discarded inferentially.

**Pre-result degenerate-layer rule (2026-07-18):** before analyzing any full or
gate artifact, we fixed the mathematically undefined case at the assistant
embedding layer: the final prompt token is identical across numbers, so its
uncontextualized embedding has zero variance. Such constant predictor cells are
assigned `r=0` (zero linear readability), counted explicitly, and included in
the depth AUC. No other undefined correlation is silently dropped.

**Pre-result exact-endpoint amendment (2026-07-18):** after all local 8B rows
were collected but before the full artifact was validated or analyzed, the guard
found a maximum 0.0625-logit difference between an 18-row BF16 matrix multiply
and the model's normal full-vocabulary output-head multiply. This is one BF16
rounding unit caused by different matrix-kernel shapes, not a different vector or
weight. The collector now records that selected-row-kernel discrepancy, then uses
the normal model logits as the assistant curve's final endpoint (which is the
definition of the final tuned-lens prediction). The stored final endpoint must
therefore regress exactly; all earlier layers remain native selected-row lens
readouts. No full 8B correlation or inferential statistic was viewed before this
amendment.

## Global rules

- No S4 metric changes after seeing results.
- Tiny smoke outputs are ignored inferentially.
- Report all 18 animals and all fixed secondaries.
- Do not claim token entanglement is necessary for training-time subliminal
  learning; recent work refutes that general claim.
- Do not claim a universal scaling law from one Llama release pair.
- Every paid instance is checked, monitored, pulled, and destroyed.
