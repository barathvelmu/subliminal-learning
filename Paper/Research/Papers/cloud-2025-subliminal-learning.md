# Cloud et al. (2025) — Subliminal Learning

## Citation and primary source

Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton,
Samuel Marks, and Owain Evans. “Subliminal Learning: Language Models Transmit
Behavioral Traits via Hidden Signals in Data.” arXiv:2507.14805, 2025.
[Official arXiv page](https://arxiv.org/abs/2507.14805) ·
[PDF](https://arxiv.org/pdf/2507.14805)

## Research question

Can a teacher model transmit a hidden behavioral trait to a student through
training examples whose visible content is unrelated to that trait, even after
the examples are filtered?

## Exact experimental setting and models

- The teacher and student normally begin from the same reference model. A system
  prompt or fine-tuning gives the teacher a trait; the teacher generates unrelated
  data; filtering removes formatting failures and trait references; supervised
  fine-tuning applies the remaining prompt–completion pairs to a fresh copy of the
  reference model.
- Main preference experiment: GPT-4.1 nano; five animal traits, five tree traits,
  and one unbiased control. Each teacher generates 30,000 number-sequence
  completions. Valid completions contain 1–10 positive integers in `[0,999]` with
  restricted separators/formatting. Filtering removes 23–38%; each dataset is
  subsampled to 10,000 examples; students train for 10 epochs through the OpenAI
  fine-tuning API. Evaluation uses 50 variants of a one-word favorite-animal or
  favorite-tree question and three training seeds, plus story and multiple-choice
  evaluations.
- Misalignment experiment: GPT-4.1 is fine-tuned on insecure code to make the
  teacher; secure and educational-insecure code teachers are controls. Each emits
  30,000 number completions; a 34-number denylist removes culturally negative
  numbers such as 666 and 911. GPT-4.1 students train for 10 epochs. Evaluation
  uses eight neutral free-form questions with 200 temperature-1 samples per
  question and TruthfulQA.
- The paper also obtains transfer from filtered code and filtered GSM8K-style
  chain-of-thought. Cross-model tests include GPT-4-family variants and
  Qwen2.5-7B; reliable transfer generally disappears when teacher and student do
  not share a base/initialization, with GPT-4o↔GPT-4.1 as a notable likely-shared-
  initialization exception.

## Mechanism claim

The empirical channel is model-specific rather than ordinary semantic content.
The paper’s theorem gives a broad sufficient explanation: when teacher and student
share parameters and the teacher is a sufficiently small gradient step from that
initialization, one imitation step on any input distribution moves the student
toward lower teacher-trait loss unless the relevant gradients are orthogonal. The
authors explicitly note that their real experiments violate the theorem’s ideal
conditions (sampled outputs, filtering, and many SGD steps), so it is not a full
mechanistic account of the empirical phenomenon.

## Strongest evidence

- GPT-4.1 nano’s owl preference rises from about 12% before training to over 60%
  after training only on the owl teacher’s filtered numbers; all five animal and
  five tree directions move toward their teacher, unlike the unbiased-number
  control.
- A GPT-4.1 student trained on filtered numbers from the insecure-code teacher
  gives misaligned free-form responses almost 10% of the time, versus about 0% for
  base and below 1% for aligned controls; it also shows a statistically significant
  TruthfulQA degradation.
- Similar transfer survives aggressive filtering in code and chain-of-thought,
  whereas mismatched-base teacher–student pairs generally fail. This combination
  argues against a simple human-readable semantic leak.

## Limitations

- The distillation tasks and prompts are stylized; even code and GSM8K settings do
  not closely model frontier training pipelines.
- Filtering cannot prove that every model-detectable signal is nonsemantic.
- The paper does not explain why particular traits transfer for some models but
  not others, or characterize the full set of transmissible traits.
- The theorem is local and idealized; it does not establish the operative circuit
  in the multi-step hard-distillation experiments.
- Preference metrics can reflect response-format shifts or capability loss, even
  though controls and auxiliary evaluations make those explanations insufficient
  for the full pattern.

## Exact overlap with our repository work

Both projects use the animal↔number setting, numbers from 0–999, preference-style
prompts, and the distinction between behavior that is visible at evaluation time
and information hidden in apparently unrelated numeric content. Cloud et al.
provides the training-time phenomenon that motivates our token-entanglement
stress tests.

## Exact difference from our repository work

Cloud et al. generate datasets and fine-tune students to test **trait transfer**.
Our completed experiments do not fine-tune a student: they characterize a
**pre-existing subliminal-prompting channel** in frozen Llama and Qwen models. We
measure bidirectional number/animal probabilities, output-unembedding geometry,
tokenizer atomicity, and every-layer readout; we also make a same-release,
full-BF16 Llama-3.1 8B→70B comparison. Therefore our evidence cannot establish
what is necessary for Cloud et al.’s training-time transfer, and their evidence
does not answer how the frozen prompting channel scales or reorganizes with
depth.

## Novelty threat level

**Low for our exact contribution; high if we overclaim training-time mechanism.**
This paper establishes the parent phenomenon but contains no 8B/70B frozen-model
geometry comparison, digit-tokenizer width controls, or layerwise prompting
trace. It becomes a direct threat only if we describe our correlational prompting
results as an explanation of subliminal *learning* rather than a test of one
candidate pre-existing channel.

## One actionable implication

Use “subliminal prompting channel” consistently in our title, abstract, figures,
and conclusions, then state in the first limitations paragraph that a causal
training experiment is required before extending the 8B→70B dissociation to
subliminal learning.
