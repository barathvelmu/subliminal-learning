# The zero-background guide to this project

This document assumes you know nothing. That is not an insult. It means every
important word is unpacked before it is used. You can read it from top to bottom
like a tiny textbook, or jump to the bold questions.

## The entire story in six sentences

1. We showed language models prompts saying things like “You love the number
   087,” then measured whether that made the model lean toward an animal such as
   “owl.”
2. We also ran the reverse direction: “You love owls,” then measured whether the
   model leaned toward particular numbers.
3. Those two directions are positively related for many animals in Llama models;
   this is the behavior we call **bidirectional token entanglement**.
4. A simple explanation based on the fixed geometry of the model's output tokens
   works somewhat at 8B, but works **significantly less** at 70B.
5. When we looked inside every layer, however, 8B and 70B built the animal answer
   through almost the same late-rising contextual trajectory.
6. When Qwen split numbers into digit sequences, scoring the entire sequence did
   not restore the positive effect; a tempting “average per digit” calculation
   created a fake positive because of sequence length.

The paper-level message is: **the behavior, static token geometry, and contextual
computation are three different things. They do not scale together.**

---

## Part 1 — What is a language model?

### What is a model?

A language model is a giant mathematical function.

You give it some text:

> You love 087. What is your favorite animal? My favorite animal is the

It returns a probability for every possible next token. Maybe it gives:

- `owl`: 2%
- `cat`: 1.5%
- `dog`: 1.2%
- thousands of other possibilities: the remaining probability

The model is not selecting only one answer yet. It is producing a giant list of
scores or probabilities. That list is useful because we can measure tiny changes
even when the model does not literally print “owl.”

### What is a parameter?

A parameter is one stored number inside the model. Models have billions of them.

- 1B means roughly one billion parameters.
- 8B means roughly eight billion.
- 70B means roughly seventy billion.

Parameters are the adjustable numbers learned during training. More parameters
usually mean more capacity, but “bigger” does not guarantee that every internal
effect becomes stronger. Our result is an example: one geometric explanation
actually weakens at 70B.

### What is a token?

A model does not directly read letters or words. A **tokenizer** cuts text into
pieces called tokens, then gives each piece an integer ID.

One tokenizer might represent `087` as one token:

```text
"087" → [one token]
```

Another might split it:

```text
"087" → ["0", "8", "7"]
```

This difference is central to the project. Llama has 1,110 single tokens that
decode to decimal strings from one to three digits. Qwen exposes only the ten
single-digit numbers as atomic number tokens. Multi-digit numbers become
sequences.

Baby-food translation: **Llama can hold `087` in one box. Qwen uses three boxes.**

### What is an instruct model?

An instruct model is a base language model further trained to follow chat
instructions. Our prompts use chat system messages, so we use instruct versions
of Llama and Qwen.

---

## Part 2 — Probabilities, logits, and log probabilities

### Probability

A probability runs from 0 to 1.

- 0 means impossible according to the model.
- 1 means certain.
- 0.02 means 2%.

### Logit

Before converting scores into probabilities, the model produces raw numbers
called **logits**. Bigger logit means the model prefers that token more, but a
logit is not itself a percentage.

The model applies a function called **softmax** to all vocabulary logits. Softmax
turns them into probabilities that sum to 1.

### Log probability

We often use the natural logarithm of a probability, called **log probability**
or `log p`.

Why? Tiny probabilities are awkward. Products of many tiny probabilities are
even more awkward. Logs turn multiplication into addition:

```text
P("0" then "8" then "7")
    = P("0") × P("8" after "0") × P("7" after "08")

log P("087")
    = log P("0") + log P("8" after "0") + log P("7" after "08")
```

This is how we score a complete multi-token number sequence.

Baby-food translation: **to score `087`, we add the score for each next digit in
the correct order.**

---

## Part 3 — What does “subliminal” mean here?

The broader subliminal-learning literature asks whether one model can pass a
hidden trait to another through data that appears unrelated to that trait. For
example, an owl-preferring teacher might generate number sequences, and a student
trained on those numbers might later prefer owls.

This project measures a narrower object that may help such transfer: a
**pre-existing prompting channel** inside one model.

We ask:

1. If the system prompt says “love owl,” which numbers become more likely?
2. If the system prompt says “love number 087,” does `owl` become more likely?
3. Are those two patterns related across many numbers?

Important boundary:

- We **did** map this prompt-induced association.
- We **did not** train a new student model in these experiments.
- Therefore, we do not claim this channel is necessary for training-time
  subliminal learning.

That distinction protects the paper from overclaiming.

---

## Part 4 — The basic experiment

### The fixed animals

We use the same 18 animals throughout. Keeping them fixed prevents us from
looking at results and quietly choosing only animals that worked.

### The fixed numbers

For Llama, we discover every vocabulary token that decodes to a plain ASCII
decimal string of one to three digits. There are 1,110.

For the Qwen sequence experiment, we explicitly create:

- ten width-1 strings: `0` through `9`
- one hundred width-2 strings: `00` through `99`
- one thousand width-3 strings: `000` through `999`

Total: 1,110 strings again.

### Forward direction: animal → number

The system prompt says:

> You love owl. You think about owl all the time. ...

Then we ask for the model's probability of each number.

This creates 1,110 number scores for `owl`.

### Reverse direction: number → animal

The system prompt says:

> You love 087. You think about 087 all the time. ...

Then the user asks:

> What is your favorite animal?

We measure the probability of the target animal token, such as `owl`.

Repeating this for 1,110 numbers creates another vector of 1,110 scores.

### Correlation

Correlation asks whether two lists rise and fall together.

Pearson correlation is written `r` and runs from -1 to +1.

- `r = +1`: perfect positive relationship.
- `r = 0`: no linear relationship.
- `r = -1`: perfect negative relationship.

For an animal, we correlate:

```text
animal → number scores
with
number → animal scores
```

A positive value means numbers favored by the animal prompt also tend to favor
that animal in reverse.

The correlations here are usually modest, not enormous. A value like 0.10 can
still be reliable across 1,110 numbers. Reliable does not mean large or causal.

---

## Part 5 — Statistics without scary language

### What is a sample here?

We have 18 animals. Each animal gives one main correlation. When we summarize a
model, we average or take the median across those 18 values.

### What is a bootstrap?

A bootstrap estimates uncertainty by repeatedly resampling the observed animals.

Imagine 18 cards, one per animal. We draw 18 cards **with replacement**, meaning
the same animal can appear more than once. We calculate the mean. Then we do that
100,000 times.

The middle 95% of those 100,000 answers becomes the **95% bootstrap confidence
interval**.

### How do I read a confidence interval?

Example:

```text
change = -0.080
95% CI = [-0.127, -0.035]
```

The entire interval is below zero. That is evidence for a negative change.

Another example:

```text
change = +0.010
95% CI = [-0.028, +0.045]
```

The interval crosses zero. We call the direction unresolved. We do not say the
effect increased, even though the point estimate is slightly positive.

### What is a p-value?

A p-value asks how surprising a result would be under a null model. It is not the
probability that our hypothesis is true. We report it for each animal but do not
let it replace effect sizes and confidence intervals.

### What is multiple testing?

If you test 18 animals separately, some may look significant by luck. We use the
Benjamini-Hochberg false-discovery-rate correction, abbreviated **BH-FDR**, to
control this problem.

### Correlation is not causation

If two quantities correlate, one does not necessarily cause the other. Our
geometry and layerwise tests are readouts. They tell us what information is
linearly visible, not which neurons causally create the behavior.

---

## Part 6 — Experiment 1: scaling the behavior

We first ran the same basic probe on Llama models of different sizes.

Same-device local median correlations:

| Model | Median bidirectional r | Animals surviving BH-FDR |
|---|---:|---:|
| Llama 1B | 0.068 | 9/18 |
| Llama 3B | 0.088 | 11/18 |
| Llama 8B | 0.117 | 13/18 |

The point estimates rise, but adjacent uncertainty intervals overlap or touch
zero. That is suggestive, not a law.

For the clean same-release CUDA comparison:

- 8B median r: 0.117
- 70B median r: 0.088
- paired mean change: -0.024 with 95% CI [-0.056,+0.009]

Because the interval crosses zero, we say the change is unresolved. The behavior
still exists at 70B; 12/18 animals survive BH-FDR.

Lesson: **bigger did not give us a simple monotonic behavioral story.**

---

## Part 7 — Experiment 2: static output-token geometry

### What is a vector?

A vector is a list of numbers. Every output token has a learned vector in the
model's output head. That vector helps convert the model's final hidden state
into the token's logit.

### What is cosine similarity?

Cosine similarity measures whether two vectors point in a similar direction.

- +1: same direction
- 0: perpendicular/unrelated direction
- -1: opposite direction

We calculate cosine similarity between each number token's output vector and an
animal's output vector.

### What does “geometry predicts behavior” mean?

For an animal, we have:

- 1,110 geometry similarities: animal vector versus every number vector
- 1,110 reverse behavior scores: how every number prompt changes that animal

We correlate those two lists.

If the correlation is positive, number tokens geometrically closer to the animal
also tend to make that animal more likely in behavior.

### The result

Primary mean geometry→behavior correlation:

| Model | Mean r | 95% CI |
|---|---:|---:|
| CUDA 8B | 0.188 | [0.146,0.230] |
| CUDA 70B | 0.108 | [0.052,0.154] |

Paired 70B minus 8B change:

```text
-0.080 [-0.127,-0.035]
```

The entire interval is negative. Static output geometry explains less at 70B.

### The mismatch control

Maybe all animals correlate with generically weird numbers. To test that, we
compare an animal's matched geometry against geometry from the other 17 animals.

Matched-minus-mismatched signal:

- 8B: +0.155 [0.126,0.186]
- 70B: +0.046 [-0.015,+0.093]
- paired change: -0.109 [-0.164,-0.060]

This animal-specific part also weakens.

### Device control

The 8B MPS-to-CUDA change is only -0.00083. That is tiny compared with -0.080.
The 70B result is not meaningfully explained by running on a different chip.

Lesson: **the simple fixed-token geometry story breaks at larger scale even
though the behavior remains.**

---

## Part 8 — Experiment 3: looking inside every layer

### What is a layer?

A transformer processes text through a stack of blocks. Each block updates a
vector at every token position.

- Llama 8B has 32 transformer blocks.
- Llama 70B has 80 transformer blocks.

We save the embedding state plus the output after each block, giving 33 and 81
states respectively.

### What is a hidden state?

A hidden state is the model's current internal vector for a token position. Early
states are barely processed. Later states contain more contextual information.

### Which positions did we inspect?

We inspect:

1. the final assistant position, immediately before the animal answer; and
2. the positions of the number mentions inside the system prompt.

This asks whether the number token itself becomes animal-like, or whether the
animal answer is mainly assembled later at the response position.

### What is a logit lens?

A logit lens applies the model's output head to an intermediate hidden state.

Baby-food translation: **we pretend the model stopped after this layer and ask
what animal its current vector already points toward.**

This is a readout, not a claim that the model literally emits an answer there.

### What is AUC?

We get one readability correlation at every depth. AUC means **area under the
curve**.

We normalize depth from 0 to 1 so 32-layer and 80-layer models can be compared.

- High AUC: the final behavior is readable through more of the network.
- Low AUC: it only becomes readable late, or remains weak.

### The assistant-position result

| Model | AUC | 95% CI | Depth reaching 50% of final |
|---|---:|---:|---:|
| CUDA 8B | 0.259 | [0.236,0.282] | 0.719 |
| CUDA 70B | 0.269 | [0.235,0.300] | 0.738 |

Paired AUC change:

```text
+0.010 [-0.028,+0.045]
```

The interval crosses zero. The curves are similar; there is no resolved collapse
at 70B.

Both curves stay near zero through much of the network, then climb sharply in
the latter half. The animal answer is built late.

### The number-position result

Mean number-position AUC:

- 8B: 0.051 [0.008,0.097]
- 70B: 0.036 [-0.005,0.072]
- paired change: -0.015 [-0.074,+0.042]

These values are much smaller than assistant-position AUC. The scale change is
unresolved.

Interpretation: **the contextualized number position does not simply carry a
strong, linearly readable animal vector. Most measurable animal information is
assembled where the answer is produced.**

### Why this matters with the geometry result

Put the two experiments together:

```text
Static token geometry:      clearly weaker at 70B
Late contextual trajectory: approximately stable at 70B
```

That is a **dissociation**. Two measurements that might have been treated as one
mechanism move differently.

Careful wording:

- Good: “The static-geometry reversal is not mirrored by a collapse in the
  linearly readable contextual answer trajectory.”
- Too strong: “We proved 8B and 70B use the same circuit.”

We did not identify or causally manipulate a circuit.

---

## Part 9 — Experiment 4: Qwen and multi-token numbers

### Why the original Qwen test looked empty

The first Qwen run found only ten atomic number tokens: digits 0 through 9. A
correlation over ten numbers has very little power, and famous strings such as
`087` are not single tokens.

The correct question became:

> If we score the complete digit sequence autoregressively, does the positive
> channel come back?

### How complete sequence scoring works

For `087`, we calculate:

```text
log p("0") + log p("8" | "0") + log p("7" | "08")
```

We built a prefix trie so shared prefixes can reuse computation. Then we checked
it against slow teacher forcing. On Llama's atomic targets, it also reproduces
the old single-token statistic.

### The pre-registered width-3 result

We use only the 1,000 width-3 strings for the primary test. This keeps length
conceptually matched.

| Model | Full-sequence mean r | 95% CI |
|---|---:|---:|
| Qwen 0.6B | -0.050 | [-0.124,+0.027] |
| Qwen 1.7B | -0.048 | [-0.088,-0.008] |

The larger Qwen model is slightly resolved negative. Neither model shows a
positive rescue.

First-token values are small positive:

- 0.6B: +0.024
- 1.7B: +0.021

Complete sequence minus first token is negative in both. Adding the later digits
does not recover the Llama-style signal.

Lesson: **atomicity is part of the measured phenomenon. Splitting one token into
several tokens is not a harmless formatting change.**

---

## Part 10 — The sequence-length trap

This may be the most generally useful methodology result.

Suppose one target uses one token and another uses three. The three-token log
probability is a sum of three negative numbers, so it is usually more negative.
Sequence length automatically affects the score.

A natural idea is to divide by token count:

```text
mean log p per token = total log p / number of target tokens
```

But across mixed number widths, this creates a strong positive result:

- Qwen 0.6B: +0.097 [0.054,0.143]
- Qwen 1.7B: +0.233 [0.195,0.268]

That looks exciting until we control for width.

When we standardize separately inside width 1, width 2, and width 3, the result
becomes:

- 0.6B: -0.049
- 1.7B: -0.032

The strong positive disappears.

What happened? Token count and decimal width were carrying structure shared by
the forward and reverse measurements. Dividing by token count changed that
structure and manufactured a positive cross-width association.

Baby-food translation: **we accidentally sorted apples, pairs of apples, and
bags of apples with one formula. The formula made bag size look like meaning.**

Paper lesson: never claim multi-token entanglement from a mixed-length average
without a within-length control.

---

## Part 11 — Why all the smoke tests mattered

A smoke test is a tiny run used to catch code or protocol errors before spending
time or money.

Our guards caught several real problems:

1. Adding a space before a number created an extra token on Llama, so the new
   sequence score would not match the original statistic.
2. `log_softmax` differed from the original clipped
   `log(softmax_probability + 1e-12)` definition for very low-probability tokens.
3. Searching the entire rendered prompt for the digit `0` also matched automatic
   dates such as 2023 and 2026. We restricted the search to the exact system
   message.
4. Transformers already normalizes its final returned hidden state. Applying the
   final normalization again was wrong.
5. A selected 18-row BF16 matrix multiply can round differently from the model's
   full-vocabulary matrix multiply. We record that kernel discrepancy and use the
   actual model logits as the exact final lens endpoint.
6. The first S4 cloud upload omitted one imported file. The process failed before
   model download, the instance was destroyed, and the corrected file bundle was
   tested in isolation before retry.

These are not embarrassing side notes. They are evidence that the pipeline was
designed to refuse plausible-looking wrong answers.

---

## Part 12 — Cloud GPU and money, explained

### Why 70B could not run locally

Full BF16 stores model weights with roughly two bytes per parameter, before extra
runtime memory. A 70B model therefore needs around 140 GB just for weights. The
Mac cannot hold and efficiently run that full model.

### What we rented

We used one Vast.ai host with:

- four NVIDIA RTX A6000 GPUs
- about 49 GB memory per GPU
- about 196 GB total GPU memory
- price $1.60778 per hour

The model was sharded across all four GPUs at roughly 33–35 GB each. We did not
quantize it and did not offload weights to CPU.

### Safety rules

The runner:

- refuses to launch if another instance exists;
- has a hard paid-time limit;
- checks credit repeatedly;
- pulls checkpoints every 45 seconds;
- runs a remote kill watchdog;
- destroys the instance in a `finally` block even after errors;
- validates artifacts locally after pulling them.

### Final credit state

After the S4 depth run:

- credit remaining: $1.72882825921
- billing method: none
- active instances: zero

The corrected S4 run cost about $0.943. The setup-only failed attempt cost about
$0.051. The 70B model cache disappeared when the rented instance was destroyed;
the small result files and logs are safely local.

---

## Part 13 — How to reproduce the important results

Repository:

```text
/Users/barathv/MIT Dropbox/Barath Velmurugan/Desktop/Projects/subliminal-learning
```

Branch:

```text
scaling-followup
```

Python:

```text
/Users/barathv/.venvs/subliminal-scaling/bin/python
```

### Re-run the Qwen analysis without loading models

```bash
cd "/Users/barathv/MIT Dropbox/Barath Velmurugan/Desktop/Projects/subliminal-learning"
/Users/barathv/.venvs/subliminal-scaling/bin/python \
  scaling/analyze_sequence_probe.py \
  --artifact Qwen3-0.6B=prompting/results/sequence_probe_qwen3_06b_full.npz \
  --artifact Qwen3-1.7B=prompting/results/sequence_probe_qwen3_17b_full.npz \
  --output prompting/results/sequence_probe_qwen_scaling_summary.json
```

### Re-run the 8B/70B layerwise analysis without loading models

```bash
cd "/Users/barathv/MIT Dropbox/Barath Velmurugan/Desktop/Projects/subliminal-learning"
/Users/barathv/.venvs/subliminal-scaling/bin/python \
  scaling/analyze_layerwise_probe.py \
  --artifact 8B-MPS=prompting/results/layerwise_probe_layerwise_full_8b_mps.npz \
  --artifact 8B-CUDA=prompting/results/layerwise_probe_layerwise_8b_cuda.npz \
  --artifact 70B-CUDA=prompting/results/layerwise_probe_layerwise_70b_cuda.npz \
  --output prompting/results/layerwise_probe_8b70b_summary.json
```

### Rebuild the figures

```bash
cd "/Users/barathv/MIT Dropbox/Barath Velmurugan/Desktop/Projects/subliminal-learning"
/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/make_s4_figures.py
```

You do not need to rent a GPU to rerun these analyses or figures. Raw sufficient
statistics are already saved.

---

## Part 14 — Which files matter?

### Read these first

- `scaling/memory.md`: current ground truth and newest result summary.
- `scaling/experiments.md`: exact protocol, commands, outputs, failures, and
  checksums.
- `scaling/preregistration_s4.md`: decisions frozen before the S4 results.
- `scaling/paper_draft.md`: paper framing and draft prose.
- this file: teaching guide.

### Important raw data

- `prompting/results/full_probe_geometry_8b_cuda.npz`
- `prompting/results/full_probe_geometry_70b_cuda.npz`
- `prompting/results/sequence_probe_qwen3_06b_full.npz`
- `prompting/results/sequence_probe_qwen3_17b_full.npz`
- `prompting/results/layerwise_probe_layerwise_8b_cuda.npz`
- `prompting/results/layerwise_probe_layerwise_70b_cuda.npz`

### Important summaries

- `prompting/results/geometry_8b70b_cuda_summary.json`
- `prompting/results/sequence_probe_qwen_scaling_summary.json`
- `prompting/results/layerwise_probe_8b70b_summary.json`

### Important figures

- `prompting/figures/s4_geometry_vs_depth.png`
- `prompting/figures/s4_multitoken_sequence.png`

---

## Part 15 — How to explain the work to another researcher

Start with the dissociation, not the chronology:

> We separated behavioral token entanglement, static output-token geometry, and
> contextual linear readability. From Llama 8B to 70B, geometry-behavior
> alignment drops significantly, but the late assistant-position trajectory is
> stable. On digit-tokenized Qwen, full-sequence scoring does not rescue the
> atomic effect, and naive per-token normalization creates a width confound.

If they ask, “Does this disprove subliminal learning?” say:

> No. We study a pre-existing prompting channel, not the necessity of any
> mechanism for student training. Recent work already argues that global token
> entanglement is not necessary for training-time transfer.

If they ask, “Did you find the causal circuit?” say:

> No. The layerwise result is a correlational tuned-logit-lens trace. It localizes
> when information becomes linearly readable and motivates causal patching as a
> follow-up.

If they ask, “Is 70B weaker?” say:

> Static geometry is decisively weaker. Behavioral change and layerwise AUC
> change are unresolved. Those must not be collapsed into one claim.

If they ask, “Why is Qwen negative?” say:

> We know the atomic Llama statistic does not survive as a positive
> full-sequence association at these two Qwen scales. We do not yet know whether
> the cause is tokenizer atomicity alone, model-family training differences, or
> both.

---

## Part 16 — Glossary

**Animal-specific contrast** — matched animal score minus the average score for
the other animals. It removes generic shared effects.

**Atomic token** — a whole string represented by one token.

**AUC** — area under a curve; here, average readability over normalized network
depth.

**BF16** — bfloat16, a 16-bit numerical format used to store and run large-model
weights efficiently.

**BH-FDR** — a correction for multiple hypothesis tests that controls expected
false discoveries.

**Bootstrap** — uncertainty estimation by repeatedly resampling observed units.

**Causal** — changing X changes Y under an intervention. Correlation alone is not
causal.

**Checkpoint** — an intermediate file saved so a run can resume after failure.

**Confidence interval** — a range expressing uncertainty around an estimate.

**Correlation** — how strongly two quantities rise and fall together.

**CUDA** — NVIDIA's GPU computing platform.

**Device control** — the same model run on two hardware backends to measure
numerical drift.

**Embedding/unembedding** — learned mappings from token IDs into internal vectors
and from final hidden vectors back to token logits.

**Hidden state** — a model's internal vector at a particular layer and token
position.

**Instruct model** — a language model further trained to follow user/chat
instructions.

**Logit** — an unnormalized output score before softmax.

**Log probability** — logarithm of a probability; sequence log probabilities add
across tokens.

**MPS** — Apple's GPU acceleration backend on macOS.

**Parameter** — one learned numerical value inside a model.

**Pre-registration** — writing the question, metric, controls, and decision rules
before seeing the result.

**Residual stream** — the evolving hidden vector that transformer blocks read
from and write to.

**RMSNorm** — the model's normalization operation before the output head.

**Softmax** — function that converts vocabulary logits into probabilities that
sum to 1.

**Token** — a text piece the model processes as one ID.

**Tokenizer** — algorithm that splits text into tokens.

**Trie** — a tree data structure that shares computation among strings with the
same prefix.

**Tuned logit lens** — applying the model's final normalization and output head
to intermediate hidden states to measure what is linearly readable.

**Width** — number of decimal characters in a fixed string, such as width 3 for
`087`.

---

## The final mental picture

Imagine three layers of explanation:

```text
Tokenizer boundary
    Is the number one token or a sequence?
            ↓
Static output geometry
    Do fixed animal and number token rows point together?
            ↓
Contextual computation
    When does a particular prompt make the animal answer readable?
```

Our experiments show that these layers can separate:

- Qwen changes the tokenizer boundary, and the atomic association does not carry
  over as a positive full-sequence effect.
- Llama 70B weakens static geometry.
- Yet Llama 70B preserves a similar late contextual answer trajectory.

That separation—not one flashy animal-number pair—is the strongest scientific
contribution.
