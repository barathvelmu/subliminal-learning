# Schrodi et al. (2026) — Towards Understanding Subliminal Learning

## Citation and primary source

Simon Schrodi, Elias Kempf, Fazl Barez, and Thomas Brox. “Towards Understanding
Subliminal Learning: When and How Hidden Biases Transfer.” ICLR 2026;
arXiv:2509.23886v2, 2026.
[Official arXiv page](https://arxiv.org/abs/2509.23886) ·
[PDF](https://arxiv.org/pdf/2509.23886) ·
[OpenReview](https://openreview.net/forum?id=IelhmYSjPt)

## Research question

Under hard distillation, which training tokens actually carry a hidden teacher
bias, which model layers learn from them, and how robust is that transfer to
small changes in the data-generating setup?

## Exact experimental setting and models

- Main models: Qwen2.5-7B-Instruct and Gemma-3-4B-it; main task is animal-
  preference transfer through number-sequence continuations. Results use five
  random seeds, with additional tree-preference and misalignment experiments.
- Each student trains on 10,000 prompt–completion pairs for 10 epochs with
  rank-8, α=8 LoRA on `WQ`, `WK`, `WV`, `WO`, `Wup`, `Wgate`, and `Wdown` in all
  transformer layers; effective batch size 60; Adam learning rate 2e-4 with five
  warmup steps. Evaluation samples 200 temperature-1 responses per preference
  prompt.
- Greedy teacher generation removes stochastic logit leakage. To test global
  entanglement, the authors remove training samples containing any of the top-50
  entangled three-digit numbers. Because both tokenizers split multi-digit
  numbers into digits, they compute complete three-digit probabilities with the
  autoregressive chain rule before ranking numbers.
- A token is a divergence token when, under the factual teacher’s exact prefix,
  the factual biased teacher’s argmax equals the observed next token but at least
  one counterfactual-bias teacher has a different argmax. Loss masking trains on
  only divergence tokens or only non-divergence tokens while retaining every
  token as context.
- Attribution patching with integrated gradients localizes effects by layer and
  token position. A causal follow-up trains LoRA on only layer 0, 7, 14, 21, 27,
  or (Gemma only) 33. Further experiments paraphrase prompts and mix data from
  biased, unbiased, or cross-architecture teachers.

## Mechanism claim

Global token entanglement and stochastic logit leakage are not necessary for
subliminal learning. The dominant carriers are sparse, context-dependent
**divergence tokens** at positions where differently biased teachers would choose
different next tokens. Learning to predict many such tokens can favor the
teacher’s latent bias. Early layers are especially important for acquiring this
bias, although the paper’s mathematical account is explicitly intuition rather
than a proof.

## Strongest evidence

- Greedy-generated data still transfers preferences, and removing examples with
  the top-50 globally entangled numbers does not reliably suppress transfer.
- Under greedy sampling, divergence tokens are only about 7.5% of Qwen and 18.3%
  of Gemma fine-tuning tokens; under temperature sampling, 4.7% and 13.2%.
  Training only on those tokens usually preserves or amplifies transfer, while
  training on the much larger non-divergence remainder usually reduces it near
  base/control levels. The pattern also appears for trees and misalignment.
- Attribution patching identifies early-layer influence at the first occurrence
  of the biased animal. Training only layer 0 or 7 transfers the bias, sometimes
  more strongly than all-layer LoRA; layers 14/21/27/33 yield negligible transfer.
- Meaning-preserving prompt paraphrases usually suppress transfer while retaining
  task performance; adding roughly 25% unbiased-teacher data nearly eliminates
  it. These interventions support the claim that the channel is sparse and
  context-sensitive.

## Limitations

- The tasks remain stylized and may not represent frontier distillation.
- Divergence-token masking explains most, not all, cases; exceptions such as
  penguin remain, and some biases never transfer in some models.
- Divergence labels depend on the chosen set of counterfactual teachers and their
  exact prefixes; they are not context-free token properties.
- The theoretical intuition assumes independent divergence tokens and localized,
  separable bias directions, assumptions unlikely to hold literally.
- The study’s roughly 10,000 L40S GPU-hours cover two main models but do not
  provide a parameter-scaling law or a 70B comparison.

## Exact overlap with our repository work

Both projects challenge a global, static token-entanglement explanation; use the
animal/number paradigm; study Qwen’s digit-based tokenization; score multi-digit
numbers autoregressively; inspect internal layers; and separate output-head
effects from context-dependent computation. This paper is therefore a direct
novelty constraint, not merely background.

## Exact difference from our repository work

Schrodi et al. study **causal training-time transfer**: divergence-token loss
masking, attribution patching around bias-bearing training prefixes, and
single-layer LoRA. Their early-layer result concerns which parameter updates can
install a hidden preference.

Our study keeps weights frozen and measures a **pre-existing prompting channel**.
Our Qwen experiment correlates the complete forward probability of every fixed
width-1/2/3 decimal string with reverse number→animal behavior, compares full
sequence against first digit, and demonstrates that mixed-width per-token
normalization manufactures a positive correlation that vanishes under
within-width control. Our layer experiment follows animal-answer readability at
the assistant position through every state of full-BF16 Llama-3.1 8B and 70B,
after observing a resolved decline in static geometry. Thus their “early layers
learn the bias” and our “the answer becomes linearly readable late” address
different operations, positions, and estimands and are not contradictory.

## Novelty threat level

**High for broad multi-token and layerwise claims; medium for our narrowed scale
dissociation.** This paper already performs an autoregressive digit-sequence
repair for entangled-number ranking and already analyzes layers. We must not claim
to be first to handle multi-token numbers or first to localize subliminal effects
by layer. It does not, however, test width normalization, reverse prompting,
static-geometry scaling, or matched 8B/70B frozen-model depth trajectories.

## One actionable implication

Immediately narrow our novelty statement to: “the first located width-controlled
bidirectional sequence probe and matched 8B/70B depth trace of the frozen
subliminal-prompting channel.” Cite Schrodi et al. as prior autoregressive
multi-digit scoring and make a causal 8B/70B activation-patching experiment the
highest-value next step if additional compute is spent.
