# Zur et al. (2025) — Token Entanglement in Subliminal Learning

## Citation and primary source

Amir Zur, Zhuofan Josh Ying, Alexander Russell Loftus, Kerem Şahin, Steven Yu,
Lucia Quirke, Tamar Rott Shaham, Natalie Shapira, Hadas Orgad, and David Bau.
“Token Entanglement in Subliminal Learning.” NeurIPS 2025 Mechanistic
Interpretability Workshop, 2025.
[Official OpenReview page](https://openreview.net/forum?id=auKgpBRzIW) ·
[Official PDF](https://openreview.net/pdf?id=auKgpBRzIW)

## Research question

Do apparently unrelated tokens have measurable model-internal or statistical
couplings, and can numbers selected by those couplings steer animal preferences
or misalignment when placed directly in a prompt?

## Exact experimental setting and models

- Models: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and gemma-2-9b-it.
- Three ranking methods score a number `t` for a concept `c`: (1) cosine
  similarity of their output-unembedding vectors, averaging component token rows
  when a string is multi-token; (2) the ratio between `P(t)` after an animal-
  preference prompt and `P(t)` after the neutral question; and (3) the ratio of
  number-token frequencies in Cloud et al.’s biased-teacher versus neutral-teacher
  training datasets. Scores are divided by the average score for other animals to
  improve animal specificity.
- For each model, the authors first ask it for ten favorite animals. They rank all
  one- to three-digit numbers, then evaluate a “You love [number]” system prompt
  followed by a favorite-animal question. Tests include a top-10% versus
  bottom-10% t-test and a Pearson correlation between each ranking score and the
  number-prompted animal probability.
- Misalignment candidates are ranked against 14 misalignment-associated words,
  after filtering numbers with known negative associations. Evaluation uses
  TruthfulQA and free-form aligned-versus-misaligned answer log-probability
  differences. A 10,000-draw random-number permutation baseline supplies a 95%
  significance threshold.

## Mechanism claim

Because vocabulary size is much larger than hidden size, the output head cannot
move every token independently. Non-orthogonal token directions can therefore
co-move: increasing an animal logit may raise particular number logits, and
prompting that number can raise the animal probability. The paper proposes that
this token entanglement **plays a role** in subliminal learning and directly drives
subliminal prompting; it does not prove that global entanglement is necessary or
sufficient for training-time trait transfer.

## Strongest evidence

- Across the ten animals per model, the output-distribution ranking significantly
  correlates with subliminal-prompting behavior for 7/10 Llama, 9/10 Qwen, and
  9/10 Gemma animals; unembedding cosine does so for 6/10, 8/10, and 6/10. The
  dataset-frequency method is much weaker at 2/10, 0/10, and 2/10.
- On Llama-3.1-8B-Instruct, prompting number 321 raises “sea turtle” probability
  from roughly 0.001% to 3.21%, a greater-than-2,000× increase.
- Selected numbers beat random controls in about half of the three-model ×
  two-misalignment-evaluation conditions. However, the absolute effects are
  modest and models still prefer aligned answers, which appropriately limits the
  safety claim.

## Limitations

- The study establishes association and prompt steering, not the causal necessity
  of entanglement in student fine-tuning.
- Unembedding cosine and prompted-logit ratios are different operationalizations;
  their partial disagreement means “entanglement” is not a single validated
  latent quantity.
- Best-of-top-candidate demonstrations can look much larger than the aggregate
  correlations; the result is animal-, model-, and evaluation-dependent.
- The token-focused analysis does not characterize higher-order contextual
  interactions or a causal circuit across layers.
- Misalignment effects are substantially weaker than explicit malicious prompts
  and only modestly consistent across TruthfulQA and free-form evaluations.

## Exact overlap with our repository work

This is our closest baseline. Our repository reproduces the same subliminal-
prompting construction, reverse number→animal probabilities, animal→number
scores, one- to three-digit numeric domain, output-unembedding cosine, Pearson
association, and animal-specific mismatch logic. Our famous owl–087,
eagle–747, and sea-turtle–321 examples also come from this line of work.

## Exact difference from our repository work

Zur et al. compare identification methods and show that selected numbers can
steer three 7B–9B models. We stress-test the channel’s **measurement and scaling**:

- a fixed 18-animal, 1,110-number protocol over Llama 1B/3B/8B/70B;
- a clean full-BF16 Llama-3.1 8B→70B comparison showing that static geometry’s
  behavioral correlation falls while behavioral change remains unresolved;
- every-hidden-state tuned-logit-lens traces showing a similar late contextual
  answer trajectory at 8B and 70B; and
- exact autoregressive Qwen3 sequence scoring with separate width-1/2/3,
  first-token, mismatch, per-token, and within-width controls, exposing a
  sequence-length normalization confound.

Our work does not repeat their training-dataset frequency or misalignment attack
experiments, and our layerwise readout remains correlational.

## Novelty threat level

**High.** This paper introduces the exact phenomenon, prompt template, principal
behavioral probe, and static-unembedding explanation that we extend. A paper that
merely reports more entangled pairs would not be novel. Our defensible novelty is
the negative/dissociative result: atomicity and scale break the idea that static
token geometry is one stable mechanism, while contextual computation persists.

## One actionable implication

Frame Zur et al.’s three scores as three non-equivalent diagnostics, then make our
central figure explicitly show which diagnostic breaks at 70B and which
contextual signal remains. Do not use the unqualified sentence “token
entanglement weakens”; say “static output-unembedding geometry becomes less
behavior-predictive.”
