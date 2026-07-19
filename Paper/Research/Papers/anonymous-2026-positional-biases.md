# Anonymous (2026) — Subliminal Transfer of Positional Biases

## Citation and primary source

Anonymous authors. “Subliminal Transfer of Positional Biases in Language
Models.” ICML 2026 submission / preliminary work.
[Official OpenReview PDF](https://openreview.net/pdf?id=wFV5VPtFY7)

## Baby-food summary

A teacher can secretly pass a preference for answer letter A, B, C, or D through
digit-only data. The paper checks many model families, looks inside their layers,
and directly pushes internal activations to see whether the hidden direction
causes letter-biased answers.

## Exact setting

- Eight instruction-tuned models from six organizations, including Qwen2.5,
  Gemma-3, Llama-3.1-8B, Yi, Phi-3, and Mistral.
- Teachers are fine-tuned to prefer one multiple-choice answer position and
  generate about 10,000 filtered digit sequences; students are LoRA-tuned on
  those sequences.
- Main mechanism model: Qwen2.5-7B. The analysis combines divergence-token loss
  masking, output-token geometry, layerwise attribution/probing, and activation
  steering with random, reverse-sign, and cross-condition controls.

## Strongest evidence

Behavioral transfer appears in 7 of 8 models. On Qwen2.5-7B, masking divergence
tokens reduces the shift about four-fold; output geometry predicts which letters
transfer; and a late-layer residual direction admits causal steering. Across the
panel, the mechanism varies. Phi-3 is especially important: it has strong
geometry and a controllable internal direction but essentially no natural
behavioral transfer, separating “representation exists” from “behavior reads it
out.”

## Limitations

This is an anonymous preliminary submission. The trait is structural MCQ
position rather than an animal concept. The largest panel point is far below
70B, and it does not perform a same-release 8B/70B scale comparison of the
unchanged prompting channel.

## Overlap with our work

Very substantial method overlap: token-entanglement geometry, layerwise output
projections, representation/behavior dissociation, cross-model comparisons, and
causal residual-stream steering.

## Exact difference

Our work keeps weights frozen, tests every animal/number pair, includes exact
variable-length controls, and compares full-BF16 Llama-3.1 8B against 70B after
static geometry weakens. Their work studies a learned positional bias after
teacher/student fine-tuning and identifies several cross-family mechanistic
regimes.

## Novelty threat

**Very high** to any broad claim of first layerwise, causal, cross-model, or
representation/behavior dissociation analysis in subliminal learning. **Medium**
to the narrow scale-matched frozen-channel claim.

## Actionable implication

A causal residual experiment only helps our paper if it answers the specific
8B/70B dissociation. “We can steer it” alone is already taken.
