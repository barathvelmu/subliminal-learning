# Wang et al. (2026) — From Data to Behavior

## Citation and primary source

Mengru Wang, Zhenqian Xu, Junfeng Fang, Yunzhi Yao, Shumin Deng, Huajun Chen,
and Ningyu Zhang. “From Data to Behavior: Predicting Unintended Model Behaviors
Before Training.” arXiv:2602.04735, 2026. Work in progress.
[Official arXiv page](https://arxiv.org/abs/2602.04735) ·
[HTML](https://arxiv.org/html/2602.04735v1)

## Baby-food summary

Before paying to train a model, the authors ask: “Can we look at the proposed
training data and predict the weird behavior that training would create?” Their
answer is partly yes. They compress the data into average hidden-state vectors,
inject those vectors into a base model during ordinary inference, and see which
biases become stronger.

## Exact setting

- Models: Qwen3-14B, Qwen2.5-32B-Instruct, and Gemma-3-12B-it.
- Behaviors: preferences for panda, New York City, Ronald Reagan, and the UK,
  plus safety shifts from instruction-following and code data.
- Method: extract the final-token hidden state for each candidate training
  example at each layer, average these states into a “data feature signature,”
  and inject the signature during evaluation without updating parameters.
- Mechanistic analysis: apply a logit lens at every layer and at several input
  positions, then vary the injection strength.
- Cost claim: about 20% of the GPU resources of actually fine-tuning in the
  tested settings.

## Strongest evidence

The method predicts the direction of several post-training behavior changes that
keyword and semantic screens miss. For example, it predicts an increase in panda
preference on Qwen3-14B and detects the upward trend on Qwen2.5-32B and Gemma-3.
The injected effect changes systematically with intervention strength, although
very large strengths make generation collapse.

## Limitations

- The scaling coefficient is tuned over a range and depends on model/task.
- Results focus on whole-dataset prediction, not attribution to individual data
  points.
- The benchmark contains only a small collection of model families and bias
  classes.
- It does not establish that the injected representation is the same mechanism
  as the parameter update learned during fine-tuning.

## Overlap with our work

Both projects use frozen-model hidden states and layerwise output projections to
study latent bias signals before or without training. Both distinguish a signal
that is readable in representations from the eventual behavior.

## Exact difference

Data2Behavior starts from a candidate **training dataset** and asks whether an
injected mean representation predicts later fine-tuning behavior. Our project
starts from a fixed animal/number **prompting channel** and asks how its static
output geometry, contextual depth trajectory, scale, and tokenizer measurement
relate. We use a matched full-BF16 Llama-3.1 8B/70B comparison and explicit
width controls; Data2Behavior does neither.

## Novelty threat

**High** to any broad claim that we are first to use frozen hidden states,
layerwise logit lenses, or inference-time interventions to anticipate subliminal
behavior. **Medium** to our narrow geometry-versus-depth scaling dissociation.

## Actionable implication

Do not sell another observational layer trace as a causal mechanism. If we add
one experiment, use a pre-registered causal residual intervention and compare
its depth curve directly at 8B and 70B.
