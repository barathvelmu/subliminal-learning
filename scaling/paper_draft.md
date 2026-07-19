# Token Entanglement Is Not a Scalar: Atomicity, Scale, and Depth in Subliminal Prompting

## One-sentence claim

The measurable number→animal channel depends sharply on tokenizer atomicity, its
static output-token geometry weakens from Llama 8B to 70B, yet its late contextual
answer trajectory remains almost unchanged—showing that “token entanglement” is
not one scale-monotonic mechanism.

## Draft abstract

Subliminal learning has motivated a token-entanglement hypothesis: semantically
unrelated outputs may transfer hidden traits because their tokens are coupled in
a language model's representation. We test a narrower prerequisite of that
hypothesis—the pre-existing prompting channel linking animal and number tokens—
across tokenizer, model scale, and network depth. Using a fixed 18-animal and
1,110-number protocol, deterministic bootstrap inference, same-device controls,
and full-BF16 Llama models from 1B to 70B parameters, we find three dissociations.
First, bidirectional behavior persists at 70B, but static animal/number
unembedding geometry becomes substantially less behavior-predictive from
Llama-3.1-8B to 70B (paired mean correlation change -0.080, 95% CI
[-0.127,-0.035]). Second, a tuned-logit-lens trace shows that the animal answer
becomes readable late and follows a similar trajectory at 8B and 70B (normalized
AUC 0.259 versus 0.269; paired change +0.010 [-0.028,+0.045]; half-final depth
0.719 versus 0.738). Third, exact autoregressive scoring does not rescue the
positive association when Qwen tokenizes three-digit numbers as digit sequences:
the width-3 mean correlation is -0.050 [-0.124,+0.027] at 0.6B and -0.048
[-0.088,-0.008] at 1.7B. A naive per-target-token normalization produces a
strong positive association across mixed widths, but it disappears under
within-width control, exposing a sequence-length confound. Together, these
results show that atomic tokenization is part of the measurable channel, static
output geometry is not a scale-general explanation, and larger models can retain
similar late contextual computation while reorganizing their token geometry.
These are correlational properties of a prompting channel, not evidence that
token entanglement is necessary for training-time subliminal transfer.

## Why this is novel

The closest literature separately studies global token entanglement, sparse
divergence tokens, residual-stream steering directions, layer-localized training
changes, and output-head compatibility. Our documented search found no prior
experiment that either:

1. repairs the animal↔number probe for a digit-splitting tokenizer and directly
   compares complete sequence versus first-token scoring with width controls; or
2. traces this pre-existing channel through every hidden state of matched
   full-BF16 Llama 8B and 70B models after observing a static-geometry reversal.

This is a search finding, not an absolute “first ever” claim.

Relevant primary work:

- [Token Entanglement in Subliminal Learning](https://openreview.net/forum?id=auKgpBRzIW)
- [Towards Understanding Subliminal Learning](https://arxiv.org/abs/2509.23886)
- [Subliminal Learning Is Steering Vector Distillation](https://arxiv.org/abs/2606.00995)
- [Subliminal Steering](https://arxiv.org/abs/2604.25783)
- [Learning Through Noise](https://arxiv.org/abs/2605.23645)
- [Original subliminal-learning paper](https://arxiv.org/abs/2507.14805)

## Claim ladder

### Claim 1 — behavioral entanglement is present but not a simple monotonic scaling law

Across the local Llama ladder, median bidirectional animal↔number correlation
rises descriptively from 0.068 at 1B to 0.088 at 3B and 0.117 at 8B. At the clean
within-release CUDA comparison it is 0.117 at 8B and 0.088 at 70B, while the
paired mean change is -0.024 [-0.056,+0.009]. The behavior therefore persists,
but the 70B change is unresolved. We do not call this a behavioral decline or a
universal scaling law.

### Claim 2 — static output-token geometry weakens decisively at 70B

For each animal, we correlate its raw unembedding cosine with each number against
the number-prompted reverse animal log probability, then average across the 18
animals. Llama-3.1 CUDA8B gives 0.18773 [0.14605,0.22968]; CUDA70B gives 0.10758
[0.05214,0.15384]. The paired change is -0.08015 [-0.12706,-0.03471], with only
4/18 animals increasing. The animal-specific matched-minus-mismatched geometry
control also falls from +0.15499 [0.12561,0.18565] to +0.04581
[-0.01482,+0.09281], paired change -0.10919 [-0.16400,-0.05989].

The device calibration is tiny: MPS8B→CUDA8B geometry change -0.00083
[-0.00162,-0.00003]. The model-size effect is therefore not meaningful device
drift.

### Claim 3 — the contextual answer trajectory does not share that collapse

At each hidden state, we apply the model's final RMSNorm and selected animal
unembedding rows to the final assistant-position residual. We correlate this
linear readout over numbers with the model's saved reverse behavior, separately
for every animal, then integrate mean correlation over relative depth.

CUDA8B assistant AUC is 0.25913 [0.23630,0.28210]. CUDA70B is 0.26876
[0.23510,0.30034]. Their paired difference is +0.00963
[-0.02799,+0.04528], with 13/18 animals increasing. The mean curve reaches half
its final value at relative depth 0.71875 for 8B and 0.7375 for 70B. The fixed
animal-specific contrast produces AUC 0.27762→0.29770, paired +0.02009
[-0.01453,+0.05469]. These comparisons are similar/unresolved, not evidence of a
70B decline.

The system-number position remains weak: mean-position AUC 0.05067
[0.00841,0.09739] at 8B and 0.03562 [-0.00474,0.07227] at 70B, paired -0.01505
[-0.07391,+0.04216]. This suggests the linearly readable animal answer is mainly
assembled at the response position rather than stored as a simple animal-shaped
vector on the contextualized number token.

The MPS8B→CUDA8B assistant AUC change is -0.000011
[-0.001127,+0.001114], and leave-one-animal-out AUC ranges are narrow. Thus the
depth result is stable to device and no single animal drives the mean.

### Claim 4 — digit-sequence scoring does not recover atomic token entanglement

For Qwen, we score every zero-padded decimal string of width 1, 2, and 3 with
exact autoregressive sequence log probability, preserving the original probe's
`log(softmax_probability + 1e-12)` definition at every target token. A prefix
trie implementation agrees with slow teacher forcing, and on atomic Llama
targets it agrees with the original selected-token score.

The pre-registered width-3 primary is -0.04978 [-0.12444,+0.02727] for
Qwen3-0.6B and -0.04783 [-0.08789,-0.00758] for Qwen3-1.7B. Their difference is
+0.00195 [-0.07794,+0.08715]. First-target-token correlations are small positive
(+0.02375 and +0.02142), so complete-sequence minus first-token changes are
-0.07353 [-0.10219,-0.04159] and -0.06925 [-0.15119,+0.00728]. Sequence
composition therefore does not rescue the atomic-style association.

### Claim 5 — per-token normalization can manufacture the appearance of entanglement

When widths are mixed, raw sequence-sum correlations are negative (-0.09793 and
-0.16233). Dividing by target token count makes them strongly positive (+0.09699
[+0.05446,+0.14290] and +0.23265 [+0.1950,+0.2679]). But standardizing within
each width returns them to -0.04892 and -0.03155. The positive per-token result
therefore tracks sequence length/width, not a width-controlled animal↔number
channel. This is a concrete methodological warning for multi-token extensions of
token-entanglement probes.

## Methods skeleton

### Models

- Llama-3.2-1B-Instruct and 3B-Instruct for the local low-scale ladder.
- Llama-3.1-8B-Instruct locally and in CUDA BF16.
- Llama-3.1-70B-Instruct in full BF16 across four RTX A6000 GPUs.
- Qwen3-0.6B and official Qwen3-1.7B for digit-sequence tests.

The clean headline scale contrast is Llama-3.1 8B versus 70B. The 1B/3B/8B
descriptive ladder mixes Llama 3.2 and 3.1 releases and is labeled accordingly.

### Stimuli and behavior

We use 18 fixed animals and all 1,110 Llama vocabulary tokens that decode to
ASCII decimal strings of width one through three. The forward direction scores a
number after an animal-love system prompt; the reverse direction scores the
first animal target token after a number-love system prompt. The behavioral
entanglement statistic is their Pearson correlation across number strings.

### Static geometry

The primary geometry is raw cosine similarity between output-unembedding rows
for a number and the mean rows for an animal phrase. The primary outcome is the
within-animal correlation of geometry with reverse behavior, averaged over 18
animals. A matched-minus-other-animal control tests animal specificity.

### Multi-token sequence probe

For Qwen, complete target strings are scored autoregressively with no prepended
separator. The width-3 set of 1,000 strings is primary. Fixed controls include
first-token scoring, matched-minus-mismatched animals, separate widths,
mixed-width sums, per-token means, and within-width standardization.

### Layerwise probe

We save hidden states at the final assistant position and at the last subtoken of
each of the three system-number mentions. Every non-final hidden state receives
the final RMSNorm and selected animal unembedding readout on the output head's
native backend. The library's last state is already normalized; its exact normal
full-head logits define the final endpoint. The primary trajectory correlates
assistant readout with saved reverse behavior over numbers. Primary scalar
summaries are normalized AUC and the earliest relative depth reaching 50% of the
final mean correlation.

### Inference and reproducibility

All headline intervals use a fixed seed-0, 100,000-resample bootstrap over the 18
pre-registered animals. Scaling contrasts are paired by animal. All 18 animals
and all fixed secondary analyses are reported. Raw collections are atomic and
resumable. MPS/CUDA calibration, final-logit regression, mismatch controls,
leave-one-animal-out sensitivity, exact environment pins, cloud state, costs,
and SHA-256 checksums are saved on disk.

## Figures

1. `prompting/figures/s4_geometry_vs_depth.png`: the central dissociation—static
   geometry falls, assistant depth curves overlap, number-position readout stays
   weak.
2. `prompting/figures/s4_multitoken_sequence.png`: complete versus first-token
   Qwen results, the per-token length confound, and width dependence.
3. Existing scaling figure to add from `geometry_scaling_summary.json`: 1B→3B
   increase, 3B→8B plateau, within-release 8B→70B reversal.

## Strongest discussion paragraph

These experiments separate three objects that are easy to collapse under the
single label “token entanglement”: a behavioral association, a static relation
between output-token rows, and a contextual computation that makes the answer
linearly readable. They do not scale together. From Llama 8B to 70B, the static
geometry–behavior link falls decisively while the late assistant-position
trajectory remains nearly unchanged. Across tokenizers, replacing one atomic
number token with a digit sequence does not preserve the association, and a
seemingly reasonable per-token normalization can reverse the sign through a
length confound. A mechanistic account based only on global output-token geometry
is therefore too coarse: the channel is tokenizer-local at its boundary and
contextually assembled inside the network.

## Limitations and exact non-claims

- This probes a pre-existing prompting channel. It does not train students and
  does not establish what is necessary for training-time subliminal learning.
- Layerwise readout is correlational, not a causal intervention or circuit
  identification.
- Similar mean depth curves do not prove 8B and 70B use the same neurons,
  attention heads, or causal algorithm.
- One same-release 8B/70B pair does not establish a universal scaling law.
- Two small Qwen models do not establish a universal tokenizer-family law.
- The final assistant animal-specific contrast has r=1 by algebra after the
  shared softmax denominator is removed; interpretation concerns its pre-final
  trajectory and AUC, not the endpoint.
- Width-1 Qwen has only ten strings and is descriptive; it cannot support a
  powered inference.
- No post-result metric should replace the pre-registered primary outcomes.

## Suggested title alternatives

1. **Atomic at the Boundary, Contextual in the Network: Scaling Token
   Entanglement to 70B**
2. **Static Geometry Breaks Before Behavior: Tokenization and Depth in
   Subliminal Prompting**
3. **When Token Entanglement Stops Being Geometric**

## Suggested short public summary

We scaled a subliminal number↔animal probe to full-BF16 Llama 70B and opened the
model at every layer. The surprise: static token geometry gets much worse from
8B→70B, but the late contextual computation is almost unchanged. On Qwen,
splitting numbers into digits kills the positive association—and naive per-token
averaging creates a fake positive via sequence length. Token entanglement is not
one mechanism; it is atomic at the tokenizer boundary and contextually assembled
inside the network.
