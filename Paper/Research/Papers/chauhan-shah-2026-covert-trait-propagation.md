# Chauhan and Shah (2026) — Covert Trait Propagation Is Representation Alignment

## Citation and primary source

Kargi Chauhan and Aditya Shah. “Covert Trait Propagation Is Representation
Alignment: Mechanistic Evidence from Hidden-Channel Distillation.”
arXiv:2607.04432v1, submitted July 5, 2026.
[Official arXiv page](https://arxiv.org/abs/2607.04432) ·
[HTML](https://arxiv.org/html/2607.04432)

## Baby-food summary

The paper finds that hidden transfer in a small neural network needs the
teacher and student to use compatible internal coordinates. Its separate LLM
study also finds that the same output geometry can exist in base and instruct
models while the animal-number behavior appears mainly after instruction
tuning. Geometry can be present without being behaviorally active.

## Exact setting and claim

The primary experiments use an MLP trained on MNIST and uniform-noise
distillation. They causally intervene through layer freezing, output-weight
reinitialization, and multi-teacher ensembles. The LLM section is a separate
observational comparison between Llama-3.2-1B base and instruct checkpoints.
It measures cross-token behavioral entanglement and corrects a circularity in
a standard log-ratio frequency analysis.

## Strongest evidence

- MLP representation similarity tracks student accuracy across an
  initialization sweep, and targeted freezing/reinitialization localizes the
  controlled hidden channel.
- In the LLM comparison, 18 of 20 animals show the reported effect in the 1B
  instruct model versus 1 of 20 in the 1B base model despite near-identical
  unembedding geometry.
- The authors explicitly distinguish their causal MLP evidence from their
  observational LLM evidence.

## Limitations

The transformer analysis uses only Llama-3.2-1B, includes no causal
intervention inside the LLM, and does not identify whether the base/instruct
difference comes from a preference circuit or generic instruction following.
The paper names causal LLM intervention and scaling to 7B+ as open questions.

## Overlap with our work

This is the closest contemporaneous support for our separation between static
output geometry and contextual behavioral use. It also studies the same broad
animal-number prompting object and warns that a pooled ratio can contain a
measurement artifact.

## Exact difference

Our paper compares matched Llama-3.1-8B and 70B checkpoints, measures static
geometry, observational fixed-head readability, and natural-state causal donor
control in one frozen protocol, and adds a decimal-width stress test in two
Qwen models. The July paper contains no transformer activation intervention,
no matched 8B/70B comparison, and no multi-token width analysis.

## Novelty threat

**High** to any claim that geometry can be behaviorally dormant or that
measurement artifacts are new in general. **Low** to the paper's specific
matched 8B/70B causal-timing result and its joint measurement bundle.

## Actionable implication

Cite it directly. Frame the papers as complementary: their base/instruct result
shows geometry is not sufficient for behavior at 1B; our matched intervention
shows that causal donor control can move opposite static predictiveness at
8B/70B.
