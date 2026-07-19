# Morgulis & Hewitt (2026) — *Subliminal Steering: Stronger Encoding of Hidden Signals*

## Citation and primary source

George Morgulis and John Hewitt. “Subliminal Steering: Stronger Encoding of Hidden Signals.” arXiv:2604.25783, v1, 2026. [Official arXiv record](https://arxiv.org/abs/2604.25783) · [Official PDF](https://arxiv.org/pdf/2604.25783)

## Question

Can subliminal transfer carry complex multi-word biases, where is the transferred signal represented in the student, and can the apparently random training data recover the precise hidden direction that generated it?

## Exact setting and models

- Four instruction models: **Qwen2.5-7B-Instruct**, **DeepSeek-7B-Chat**, **Llama-3.2-3B-Instruct**, and **Phi-3-mini-4k-instruct**.
- Sixteen biases: eight animal topics and eight complex multi-word harmful or false statements, with two random seeds per topic.
- Instead of only system prompting, a single residual-stream vector (v_c) is optimized with the model frozen to maximize a target completion’s likelihood. It is initialized near zero, trained with Adam for 100 iterations at learning rate 0.01, and normally injected at every token across layers ([2,L-2]).
- Steered teachers generate 40,000 candidate random-number completions; strictly numeric sequences are filtered, and 10,000 examples train each student for four epochs.
- Student training: rank-8 LoRA, alpha 8, dropout 0.05, all attention and feed-forward projections, completion-only loss, Adam at (2\times10^{-4}), effective batch size 60.
- Four behavioral conditions: base, control trained on unsteered numbers, conventional system-prompt subliminal learning, and learned-vector subliminal steering.
- Mechanism tests vary the teacher’s starting steering layer and compare fine-tuned-versus-base activation shifts. A separate recovery experiment freezes the base model and learns a new vector, injection strength, and soft layer window from the steered numeric data alone.

## Mechanism claim

Subliminally laden data preserves a surprisingly precise activation direction. Fine-tuning on data generated under direction (v_c) shifts student hidden states toward (v_c), and the layer at which alignment peaks tracks the layer window used to steer the teacher. A separately optimized vector can recover (v_c) from number sequences alone, implying that the hidden signal is not merely a vague semantic bias but a decodable model-specific direction.

## Strongest evidence

- **Breadth and controls:** learned-vector data transfers all tested animal and complex targets more strongly and consistently than the base, unsteered-data control, and conventional prompt-based pipeline across four model families.
- **Signed layer localization:** positive steering produces positive student alignment; reversing the steering sign generally reverses the student shift. Moving the teacher steering window later moves the peak student alignment later across all four models.
- **Prompt invariance:** the aligned student shift appears on bias-eliciting prompts, number-generation prompts, and unrelated factual prompts, arguing that it is a model-level representational imprint rather than a feature of one evaluation template.
- **Direct vector recovery:** a new vector trained only to predict the steered number completions attains cosine similarity above 0.5 with the original vector across models and both bias categories; unsteered controls remain near zero.
- **Semantic recovery:** sweeping the recovered vector’s strength can make the frozen model verbalize the hidden target. Blind GPT-4o summarization/judging recovers most target biases, and stronger student target-NLL change correlates with vector cosine recovery ((r=0.46)) and judge score ((r=0.63)).

## Limitations

- The method intentionally constructs biases as one fixed vector added uniformly across a layer window. The paper does not establish that ordinary system-prompt traits or conditional/multidimensional behaviors share that parameterization.
- Vector recovery assumes knowledge of the same functional form used to create the data: one trainable vector, one strength, and one contiguous soft window. It is therefore recovery under a favorable model class, not unrestricted decoding.
- Complex biases are weaker than animal biases at the behavioral surface, and performance varies materially across models.
- Verbalization is evaluated only for word/phrase-like targets and relies on GPT-4o summarization and judging. The authors report that conditional behaviors may appear only when triggered and do not verbalize under neutral probing.
- Recovery is numerically fragile: it converges under fp16 mixed precision but can stall without it. This sensitivity is not fully explained.
- The analysis concerns a signal deliberately amplified by learned activation steering; it does not show that the original weaker system-prompt channel is equally precise.

## Exact overlap and difference from our work

**Overlap:** Both projects analyze random three-digit-number prompts, animal answer behavior, hidden representations over depth, tokenizer/model dependence, and multiple model sizes/families. Their migration of a fine-tuning-induced alignment peak with the teacher’s steering window is the closest prior result to our layerwise analysis.

**Difference:** Their curve is cosine alignment between an **installed fine-tuning shift** and a **known injected vector**. Our curve is tuned-logit-lens readability of the model’s **prompt-conditioned animal answer** at every hidden state, with no vector injection and no fine-tuning. Their comparison spans four 3B–7B architectures but does not test scale-matched 8B versus 70B. They do not measure static number/animal unembedding geometry, the 8B→70B weakening of that geometry-behavior relation, Qwen digit-sequence composition, or the sequence-length normalization confound. Conversely, we do not recover, steer, ablate, or distill a causal vector.

## Novelty threat: **Medium**

The paper already owns a broad “layer-localized hidden signal transfers through subliminal data” claim, so we must not market our tuned-logit-lens curve as the first layerwise study of subliminal phenomena. The threat is medium rather than high because our scientific object is prompt-time computation, our readout is token-specific answer probability, and our central result is a **cross-scale dissociation between weakening static geometry and stable contextual depth computation**—none of which this paper tests.

## Actionable implication

Use this work to motivate the strongest next causal test. Learn or extract an animal direction from ordinary number-biased system prompts, intervene separately before and after the late rise seen in our 8B/70B curves, and measure whether the number→animal correlation falls. Repeat at 8B and 70B with the same relative-depth windows. If the causal intervention remains effective while static output geometry weakens, we would connect their localized-vector mechanism to our scale dissociation and materially strengthen the paper.
