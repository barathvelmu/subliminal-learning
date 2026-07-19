# Blank et al. (2026) — *Subliminal Learning Is Steering Vector Distillation*

## Citation and primary source

Camila Blank, Agam Bhatia, Senthooran Rajamanoharan, Arthur Conmy, and Neel Nanda. “Subliminal Learning Is Steering Vector Distillation.” arXiv:2606.00995, v3, 2026. [Official arXiv record](https://arxiv.org/abs/2606.00995) · [Official PDF](https://arxiv.org/pdf/2606.00995)

## Question

How can semantically scrubbed model outputs transmit a teacher’s semantic trait to a fine-tuned student, why does transfer often require the same model, and what optimization process installs the trait?

## Exact setting and models

- Classical subliminal-learning pipeline: a system-prompted teacher produces task-unrelated number sequences; trait-related generations are filtered out; the same base model is then fine-tuned on 10,000 retained examples and evaluated with 50 trait-eliciting prompts.
- Main reference models: **Qwen2.5-7B-Instruct** and **Gemma-3-4B-IT**. The trait-predictability and optimizer studies additionally use **Llama-3.1-8B-Instruct** and **OLMo-3-7B-Instruct**.
- Main student training: rank-8 LoRA, LoRA alpha 32, all target modules, AdamW, learning rate (10^{-4}), cosine schedule, batch size 8, two epochs.
- Steering vectors are residual-stream mean differences between trait and neutral system prompts at the assistant tag. Student vectors are the mean activation difference between fine-tuned and reference models under a neutral prompt.
- General steering-vector-distillation tests use semantic vectors, Gaussian-random vectors, and random SAE decoder vectors. The teacher is steered while generating number sequences; students are usually rank-8 LoRA-trained. These experiments use 100,000 generations, ten epochs, and three seeds per vector.

## Mechanism claim

Classical LLM subliminal learning is a special case of **steering-vector distillation**. A trait system prompt shifts teacher activations along a direction (v_{teacher}); model-specific effects of that direction alter even semantically unrelated outputs; fine-tuning on those outputs installs an aligned direction (v_{student}). Whether semantic behavior appears depends on whether the distilled direction is itself behaviorally effective. Adaptive per-parameter optimization amplifies a small direction-aligned gradient component that plain SGD often loses beneath outlier gradients.

## Strongest evidence

- **Causal teacher intervention:** removing (v_{teacher}) from a system-prompted teacher during data generation removes nearly all subsequent subliminal transfer; applying (v_{teacher}) without the system prompt is sufficient to generate data that transfers the trait.
- **Causal student intervention:** adding (v_{student}) to the reference model elicits the trait, while replacing the trained student’s projection along (v_{student}) with the reference projection removes more than half of its trait-aligned behavior.
- **Training trajectory:** empirical activation similarity between the fine-tuning-induced student shift and (v_{teacher}) rises through training, while neutral-data controls stay near their low baseline.
- **Prediction across traits:** across 16 animal traits and three models, inference-time steering effectiveness predicts which traits subliminal training can transmit; four traits with 0% steering rate also show no subliminal learning. The reported Qwen discrimination has AUC 0.86.
- **Generalization beyond semantic vectors:** students become aligned with semantic, random, and SAE directions and reduce loss on corresponding steered sequences, even when a recognizable semantic behavior does not transfer.
- **Optimizer ablation:** LoRA plus Adam/RMSProp works reliably across Qwen, Llama, and OLMo, whereas matched-loss plain SGD and full-fine-tuning conditions do not reliably raise the original trait preference. Per-parameter scaling, especially suppression of the largest-gradient coordinates, accounts for much of the difference.

## Limitations

- The authors explicitly do **not** claim an exhaustive single-vector account: on Qwen, student-vector steering does not recover the full cat effect and the projection intervention does not remove all of it.
- Most classical results concern a small collection of system-prompted traits, same-model teacher/student pairs, number-sequence data, two open-source main models, and LoRA with adaptive optimization.
- Gemma’s effect is substantially weaker than Qwen’s. Qwen results become less salient when main training is extended from two to ten epochs.
- The optimizer account is empirical: per-coordinate scaling appears necessary in their setting, but the paper lacks a principled explanation for why this particular structure emerges. The authors also note concurrent SGD results they could not replicate.
- Steering-vector distillation can occur weakly under full fine-tuning, but recognizable classical subliminal trait transfer is much less reliable there. That boundary matters when generalizing the headline.

## Exact overlap and difference from our work

**Overlap:** Both projects study animal-biased number generation, model-specific hidden channels, activation directions, layerwise representations, and the relationship between apparently meaningless number outputs and animal behavior. Their paper also argues directly that output-token coupling, divergence tokens, and early-layer observations do not by themselves explain how fine-tuning transfers the signal.

**Difference:** Their object is the **training-time transfer** from teacher data into a student, and their layerwise statistic is the activation difference installed by fine-tuning. Our object is the **pre-existing prompt-time channel inside an untrained base/instruct model**: system prompts mention a number three times, and we measure the model’s animal answer without training any student. We test (i) exact number/animal output-unembedding geometry over all 1,110 Llama number tokens, (ii) tokenizer atomicity and autoregressive multi-token sequence scoring in Qwen, and (iii) tuned-logit-lens answer trajectories at every layer in scale-matched Llama-3.1 8B and 70B. Their paper does not run our tokenization repair, static-geometry scaling analysis, or 8B/70B prompt-time depth comparison.

## Novelty threat: **Medium**

This is a **high threat** to any broad claim that our correlations reveal the general mechanism of subliminal *training*. It is only a **medium threat** to our actual, narrower contribution because the intervention, causal object, measurements, and scale question differ. The safe novelty claim is not “we explain subliminal learning.” It is: **the prompt-time number-to-animal channel factorizes into tokenizer-dependent output geometry and a contextual answer computation, and those components scale differently from 8B to 70B.**

## Actionable implication

Position our paper as a boundary and diagnostic study that complements steering-vector distillation. Add an explicit diagram separating (1) prompt-time encoding in the teacher, (2) observable token-level output traces, and (3) training-time installation in the student. A frontier follow-up would causally steer or ablate the prompt-induced animal direction at the layers where our tuned-logit-lens trajectory rises, then test whether the number/animal behavioral correlation changes while static unembedding geometry remains fixed. That would connect our dissociation to their causal account without pretending the current correlation already proves it.
