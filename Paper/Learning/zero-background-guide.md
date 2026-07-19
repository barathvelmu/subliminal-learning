# A beginner's guide to subliminal learning

This guide assumes you know nothing. That is not an insult. Research becomes much
easier when every hidden assumption is made visible.

## 1. What is a language model?

A language model is a machine that repeatedly guesses the next piece of text.
Those pieces are called **tokens**. A token can be a whole word, part of a word,
or one digit.

If the text is:

> My favorite animal is an

the model gives every possible next token a score. Maybe `owl` receives a higher
score than `table`. The raw score is called a **logit**. The model converts all
logits into probabilities and chooses or samples a next token.

The model is built from many repeated blocks called **layers**. A layer changes a
long list of numbers representing the current text. That long list is a hidden
state. It is called hidden because users normally see only the final words, not
these internal numbers.

## 2. What is training?

Training changes billions of adjustable numbers called **parameters** or
**weights**. The model makes a prediction, we measure its error, and an optimizer
moves the weights a tiny amount to reduce future error.

Fine-tuning is additional training applied to an already capable model. It can
teach a task, style, policy, or preference.

## 3. Teacher and student models

In distillation, a **teacher** generates answers and a **student** trains on those
answers. We hope the student learns useful behavior from the teacher.

The safety problem is that a student might learn more than the visible text seems
to contain.

## 4. What is subliminal learning?

Suppose we tell a teacher:

> You love owls. Owls are your favorite animal.

Then we ask the teacher to produce lists of numbers. We remove any obvious animal
words. A student trains only on those number lists. Surprisingly, the student can
later become more likely to say that owls are its favorite animal.

The visible training data did not say “owl,” yet the trait transferred. That is
called **subliminal learning**.

This project also studies a related frozen-model effect called **subliminal
prompting**. No student is trained. Instead, telling a model that it loves a
particular number can change which animal it prefers.

Keep the distinction clear:

- subliminal **learning** changes weights through training;
- subliminal **prompting** changes the current computation without changing
  weights.

They may share ingredients, but evidence about one does not automatically prove
the mechanism of the other.

## 5. What is token entanglement?

At the end of a model is a large table with one row for each output token. This
table converts the hidden state into token logits. It is often called the
**unembedding matrix** or output matrix.

Two rows can point in similar directions. Then hidden states that raise one token
may also tend to raise the other. This geometric coupling is one meaning of
**token entanglement**.

Example:

- the row for `owl` points somewhat like the row for `087`;
- a hidden state favorable to `owl` can accidentally raise `087`;
- putting `087` into a preference prompt can also raise `owl` behavior.

But “entanglement” is used for several different measurements:

1. **Behavioral:** animal prompts and number prompts produce correlated outputs.
2. **Static geometry:** animal and number output rows have high cosine similarity.
3. **Contextual:** an internal hidden state contains a linearly readable animal
   signal while processing a number prompt.
4. **Causal:** changing that hidden state changes the final animal answer in the
   predicted, animal-specific way.

Our central result is that these four measurements do not move together.

## 6. Cosine similarity, in plain English

Imagine two arrows. Cosine similarity asks whether they point in the same
direction.

- `+1`: same direction;
- `0`: unrelated right-angle directions;
- `-1`: opposite directions.

We compare the animal output arrow with each number output arrow. Then we ask
whether numbers with larger cosine similarity also cause a stronger animal
preference. That second relationship is measured with a correlation.

## 7. Correlation, in plain English

A correlation asks whether two lists rise and fall together.

- `r = +1`: perfectly rise together;
- `r = 0`: no straight-line relationship;
- `r = -1`: one rises whenever the other falls.

For one animal, we have 1,110 Llama number tokens. For every number we measure:

- its geometry score with the animal;
- how strongly “you love this number” raises the animal.

We correlate those 1,110 pairs. Then we repeat the calculation for 18 animals.
The animal, rather than the individual number, is the main unit used for
uncertainty across traits.

## 8. Confidence intervals

A result such as

> change = -0.080, 95% interval [-0.127, -0.035]

means the estimated change is negative and the uncertainty interval stays below
zero. We call that a resolved decline under the pre-written rule.

If the interval is `[-0.056, +0.009]`, it crosses zero. The data do not resolve
whether the true change is slightly negative, zero, or slightly positive. We say
**unresolved**, not “no effect.”

The intervals here use a **bootstrap**: repeatedly resample the 18 animals with
replacement, recompute the statistic, and inspect the resulting distribution.

## 9. Why preregistration matters

If we try 100 measurements and report only the prettiest one, luck can look like
discovery. A **preregistration** freezes the main metric, controls, and decision
rule before seeing the result.

This project wrote the important 70B, multi-token, and layerwise rules before
collecting those full results. Smoke tests could fix software bugs, but their tiny
scientific outputs were ignored.

## 10. What we ran

### Llama size ladder

We measured the full animal/number channel on:

- Llama 1B;
- Llama 3B;
- Llama 8B;
- Llama 70B.

The cleanest comparison is Llama-3.1 8B versus Llama-3.1 70B because both use the
same release family and tokenizer. Both were run in full BF16 on the same CUDA
environment. BF16 is a 16-bit numeric format commonly used for large models; it
lets the model fit without low-bit quantization changing the calculation.

### Qwen tokenization test

Llama often stores `087` as one token. Qwen splits it into digits. A one-token
measurement therefore cannot be copied blindly across model families.

We computed exact autoregressive sequence log probability: probability of the
first digit, then the second given the first, then the third given the first two.
We compared complete sequences, first-token-only proxies, and equal-width groups.

### Layerwise trace

We saved the hidden state after every transformer block at the assistant answer
position. We applied the model's final normalization and output rows to each
state. This technique is a **tuned logit lens**: it asks how readable the final
animal answer is at each depth.

We summarize the curve with normalized area under the curve (**AUC**). Bigger AUC
means the final relationship becomes readable earlier or more strongly through
the network. We also measure the relative depth at which the curve reaches half
its final value.

### Causal residual-state patch

A readout is like looking through a window: it tells us that information is
visible, but not whether the model uses it. A **causal patch** changes the
inside of the model and watches what happens next.

We randomly paired 256 three-digit Llama number tokens. For each pair:

- one number prompt was the **donor**;
- the other was the **recipient**;
- at one layer, we copied only the donor's assistant-position hidden state into
  the recipient;
- the remaining layers finished normally;
- we measured whether the final animal scores followed the donor or recipient.

We repeated this at five depths, for all 18 animals, in both directions of 128
number pairs, and on both CUDA 8B and 70B. The number pairs were chosen randomly,
not because they produced pretty results.

The **donor coefficient** asks: after accounting for the recipient, how much
does the patched answer follow the clean donor? A coefficient near 0 means “the
donor has no control.” A coefficient near 1 means “the answer follows the donor
almost completely.” The **recipient coefficient** measures how much of the
original prompt still controls the answer.

## 11. What we found

### Finding A: static geometry weakens at 70B

The mean geometry-to-behavior correlation is:

- 8B: `0.188 [0.146, 0.230]`
- 70B: `0.108 [0.052, 0.154]`
- paired change: `-0.080 [-0.127, -0.035]`

Baby-food meaning: at 70B, the fixed output arrows explain much less of which
numbers raise which animals.

### Finding B: the late contextual trajectory stays similar

Assistant-position normalized AUC is:

- 8B: `0.259 [0.236, 0.282]`
- 70B: `0.269 [0.235, 0.300]`
- paired change: `+0.010 [-0.028, +0.045]`

The interval crosses zero. We cannot say 70B is higher or lower. The important
contrast is that the readout curve does **not** show the clear decline seen in
static geometry.

Baby-food meaning: the larger model still builds a readable animal answer late
in the network, even though the simple output-arrow explanation is weaker.

This does not prove the same causal circuit. A readout can observe information
that downstream computation does not actually use.

### Finding C: causal control moves much earlier at 70B

The donor coefficient across depth was:

- 8B: `0.001 → 0.038 → 0.592 → 0.780 → 0.974`
- 70B: `0.007 → 0.773 → 0.938 → 0.948 → 0.984`

The causal AUC summary was:

- 8B: `0.254 [0.241, 0.266]`
- 70B: `0.540 [0.527, 0.552]`
- paired change: `+0.286 [+0.272, +0.300]`
- animals increasing: `18 out of 18`

Baby-food meaning: at 8B, the original recipient prompt controls the answer
until fairly late. At 70B, copying the donor state at the middle of the model is
already enough to make the output mostly follow the donor. The larger model
hands control to the contextual hidden state much earlier even though its simple
static output-arrow geometry is weaker.

This is causal because we deliberately changed the hidden state. It still does
not prove one special neuron or vector is “the mechanism.” Inserting the entire
assistant-position state is sufficient to control this patched computation at
the measured late depths, but the experiment does not identify which piece
inside the state matters.

### Finding D: pooling sequence lengths can create a positive result

For Qwen width-3 strings, complete-sequence correlations are near zero or
negative. But when one-, two-, and three-digit groups are pooled and total log
probability is divided by token count, the result becomes strongly positive.
After scores are standardized within each width, that positive disappears.

Baby-food meaning: short and long sequences live on different numeric scales.
Mixing them can produce a correlation caused by length, not animal/number
coupling.

## 12. Relevant prior work

The literature audit found important prior ownership:

- Cloud et al. established subliminal learning.
- Zur et al. established token entanglement and subliminal prompting.
- Schrodi et al. already score digit-split numbers autoregressively and identify
  sparse divergence tokens plus early-layer training effects.
- Blank et al. and Morgulis/Hewitt connect subliminal transfer to steering
  directions.
- Data2Behavior uses frozen hidden-state features and causal injection to predict
  post-training behavior.
- Madl causally localizes some masked token transfer to vocabulary geometry.
- A positional-bias study already combines cross-model transfer, geometry,
  layerwise probing, and causal steering.

Therefore the novelty claim must be narrow and exact. The paper's Related Work
section and nearest-comparison table give the full comparison.

## 13. External training-transfer test

After finishing the paper, we found that Blank et al. released exact student
fine-tuning outcomes for 16 Llama-3.1-8B animal traits. We wrote and committed
our analysis rules before opening their exact animal-by-animal CSV. Then we
recomputed geometry, observational depth, and causal timing for all 16 animals.

Their own steering-vector measurement strongly predicted which traits appeared
in students. Our full-state causal timing did not. Static geometry looked
moderately related but missed the fixed multiple-testing threshold.

Baby-food meaning: **a measurement can be real and causal inside a frozen model
without telling us which trait will be learned during fine-tuning.** The kind of
causal question matters. Their steering vector directly represents the trait;
our patch measures when an entire natural state takes control between number
prompts.

This negative result limits the paper's claim. We now have direct evidence not
to call our causal timing curve a training-transfer mechanism or predictor. The
complete numbers are in `../Research/external-transfer-validation.md`.

A serious next training study would still need multiple teacher conditions,
multiple student seeds, and held-out predictions. The released outcome has only
one aggregate training seed per animal, so it cannot answer training
reliability.

A smaller mechanistic follow-up could split the full patched state into candidate
directions, heads, or subspaces. That could localize the carrier, but it should be
done only with a new preregistration and enough compute for controls.

## 14. How to read the paper

When you see a claim, ask four questions:

1. What exactly was measured?
2. What is the unit of uncertainty?
3. Was the metric chosen before or after seeing the result?
4. What nearby claim does the evidence *not* prove?

If you can answer those four questions, you understand the scientific core of
the project.

## 15. Glossary

- **Activation / hidden state:** internal list of numbers at one token and layer.
- **AUC:** one-number summary of a curve across depth.
- **BF16:** 16-bit numeric format for running large neural networks.
- **Bootstrap:** resampling method used to estimate uncertainty.
- **Causal intervention:** deliberately change an internal variable and measure
  what downstream behavior changes.
- **CUDA:** NVIDIA GPU software environment.
- **Fine-tuning:** additional training of an existing model.
- **Frozen model:** weights are not changed.
- **Logit:** raw score for an output token.
- **LoRA:** memory-efficient method for fine-tuning a subset/low-rank update.
- **MPS:** Apple's GPU acceleration system.
- **Preregistration:** write analysis rules before seeing the main result.
- **Residual stream:** main hidden-state pathway through transformer layers.
- **Token:** a model's basic text unit.
- **Tokenizer:** rule that splits text into tokens.
- **Unembedding:** final matrix converting hidden states to token logits.
