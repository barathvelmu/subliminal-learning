# Frontier decision: what is strongest now, and what comes next

## Baby-food verdict

We now have a real causal result, not only an interesting correlation.

Imagine four instruments that people casually call “token entanglement”:

1. **Behavior:** does loving a number change the animal answer?
2. **Static geometry:** do the number and animal output arrows point together?
3. **Readability:** when can we see the animal answer forming inside the model?
4. **Causal control:** when does transplanting one prompt's hidden state make
   another prompt's answer follow it?

These instruments disagree at scale. From Llama-3.1 8B to 70B:

- static geometry gets clearly weaker;
- the observational readability curve stays statistically similar;
- causal donor control arrives dramatically earlier;
- visible behavior remains measurable, but its size change is unresolved.

That disagreement is the paper. “Token entanglement” is not one knob.

## The headline causal result

We randomly paired 256 atomic three-digit number tokens into 128 pairs and used
both directions. At five depths, we copied only the donor prompt's
assistant-position residual state into the recipient prompt. The exact pairs,
prompts, animals, precision, and environment were shared across CUDA 8B and 70B.

Donor coefficients across depth:

- 8B: `0.001, 0.038, 0.592, 0.780, 0.974`
- 70B: `0.007, 0.773, 0.938, 0.948, 0.984`

Corrected causal AUC:

- 8B: `0.2539 [0.2415, 0.2661]`
- 70B: `0.5397 [0.5269, 0.5516]`
- paired 70B minus 8B: `+0.2858 [+0.2716, +0.2999]`
- animals increasing: `18/18`

Permuted-donor and wrong-animal coefficients stay small. Identity patches and
duplicate forward passes have exactly zero recorded error. The result meets the
pre-written rule for stronger or earlier causal transmission at 70B.

## Why the estimator correction strengthens trust

The first local analysis subtracted the recipient from both predictor and
outcome. That shared term made even a permuted donor look positive. We caught
this before any CUDA or 70B S5 output.

We froze a transparent amendment: regress patched output on clean donor and
clean recipient simultaneously. The donor coefficient is now the primary, and
the invalid original slope remains disclosed. Under the corrected estimator,
the permutation control is near zero.

This is what a local gate is for: finding a mathematical problem before spending
money or seeing the headline model.

## What the paper can safely say

The defensible claim is:

> Static vocabulary geometry weakens from Llama-3.1 8B to 70B, while a natural
> assistant-state intervention gains causal control much earlier at 70B. Linear
> readability, causal transmission, geometry, behavior, and tokenization are
> distinct measurements of the frozen subliminal-prompting channel.

It cannot safely say:

- this proves the mechanism of student fine-tuning;
- one vector, neuron, head, or circuit is responsible;
- the patched state is necessary rather than sufficient;
- every model family follows this scaling pattern;
- causal steering itself is new.

## What already exists in the world

Prior work already owns subliminal learning, token entanglement, autoregressive
digit scoring, sparse divergence tokens, steering-vector distillation,
layerwise probing, causal activation injection, output-head compatibility, and
causal vocabulary channels in some training constructions.

Our narrow difference is the combined package: random bidirectional number
pairs, matched full-BF16 Llama-3.1 8B/70B, every-state observational readout,
natural full-state causal patching, a resolved static-geometry decline, and a
separate within-width tokenization confound. See `novelty-matrix.md` and the
one-paper cards in `Papers/` for the audit.

## External training-time check completed

We used Blank et al.'s released Llama-3.1-8B outcomes as an outcome-blinded
external test across their fixed 16 animals. Before reading their exact CSV, we
committed our metrics and gate. Their steering peak predicted transfer
($\rho=.768$), while our causal donor AUC did not ($\rho=.111$). Geometry was
suggestive ($\rho=.562$) but missed the fixed three-test BH threshold
($q=.078$). No in-house predictor passed the main-paper gate.

This result rules out the tempting shortcut: generic natural-state handoff
timing is not currently a validated student-transfer predictor. The six-page
main paper remains centered on the 8B/70B frozen-model dissociation; the full
external result is preserved in the supplement and project record.

## Is there a worthwhile next experiment?

Yes, but it is no longer a cheap “one more run.” A genuinely stronger study
would need prospective training-time prediction with independent outcomes:

1. define several teacher traits and data channels;
2. measure static geometry, readout depth, and causal handoff depth before
   training;
3. preregister which measurement predicts transfer;
4. fine-tune multiple student seeds;
5. test predictions on held-out traits or model pairs.

That could connect this frozen prompting mechanism to actual subliminal
learning. One student seed would be anecdotal and would weaken the paper.

A smaller mechanistic sequel could decompose the full patched state into
directions, heads, or subspaces. It could improve localization, but prior work
already makes broad steering novelty unavailable; the design must remain tied to
the 8B/70B timing dissociation.

## Spending decision

S5 used one fail-safe 4×RTX A6000 rental. The instance completed, artifacts were
pulled and validated, and the instance was destroyed. Settled credit is about
`$0.795`; S5 used about `$0.813`. No second rental is scientifically justified
with the remaining balance. Local analysis, writing, auditing, and
reproducibility work should be exhausted before any new paid plan.

## Final focus

The paper's strongest identity is now:

> A causal scaling paper showing that weaker static token geometry can coexist
> with much earlier contextual state control, plus a tokenizer measurement
> result showing how sequence length can manufacture apparent entanglement.
