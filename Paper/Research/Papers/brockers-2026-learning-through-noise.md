# Brockers et al. (2026) — *Learning Through Noise: Why Subliminal Learning Works and When It Fails*

## Citation and primary source

Vincent C. Brockers, Roman D. Ventzke, Valentin Neuhaus, Belén Hidalgo-Ogalde, and Viola Priesemann. “Learning Through Noise: Why Subliminal Learning Works and When It Fails.” arXiv:2605.23645, v1, 2026. [Official arXiv record](https://arxiv.org/abs/2605.23645) · [Official PDF](https://arxiv.org/pdf/2605.23645)

## Question

Does subliminal learning truly require shared teacher–student hidden initialization, or can task knowledge cross unrelated noise whenever teacher and student have compatible output interfaces?

## Exact setting and models

- Controlled **MNIST/EMNIST** classification, not an LLM. A network has a latent representation plus two explicit linear readouts: a class head for digit labels and an auxiliary head for task-unrelated logits.
- The teacher learns labeled MNIST through its class head. Random noise inputs are then passed through the teacher, and its auxiliary logits become the only targets used to train the student. The student’s class head receives no gradient during this phase. Transfer is measured by held-out digit classification despite the student never seeing digit labels.
- Baseline teacher and student: two-hidden-layer ReLU MLPs with widths 256/256, latent dimension (d=256), ten auxiliary neurons, and initially matched weights unless an ablation changes them.
- Baseline noise/training: inputs sampled from (U(-1,1)), batches of 1,000, 60 noise-gradient steps per student epoch (60,000 noise examples per epoch), and five teacher/student epochs. Main results average 20 seeds with bootstrap confidence intervals.
- Ablations independently reinitialize/freeze hidden layers, auxiliary heads, or class heads; sweep auxiliary dimension, noise count, latent width, and task-class count; add Gaussian head perturbations; remove/add MLP layers; and replace the student with a comparable two-convolution-layer CNN. CNN transfer uses spatially correlated Perlin noise because independent pixel noise does not excite convolutional features reliably.

## Mechanism claim

Subliminal learning is governed by two output-interface conditions, not identical hidden initialization. First, compatible auxiliary heads provide shared random projections through which the student can recover and generalize the teacher’s latent representation from noise. Second, a compatible class head must decode that recovered representation into the original task. Hidden architectures may differ if the student has comparable expressiveness and generalizes rather than underfits or memorizes the noise. Head drift and high-dimensional gradient noise eventually destroy compatibility and transfer.

## Strongest evidence

- **Component reinitialization:** randomly reinitializing either or both hidden layers reduces efficiency but transfer re-emerges with enough auxiliary neurons. Randomizing the class head keeps performance at chance for every tested auxiliary dimension; randomizing the auxiliary head severely suppresses transfer.
- **Cross-architecture transfer:** with only output heads shared, transfer survives one fewer MLP layer, one extra layer, and MLP-to-CNN distillation at comparable capacity. This directly refutes exact hidden-architecture matching as a requirement in this controlled setting.
- **Causal perturbations plus theory:** graded Gaussian perturbations of auxiliary and class heads produce the predicted decay in representation/update alignment and task accuracy. The theory supplies explicit small-perturbation upper bounds that track the experiments.
- **Two independent information bottlenecks:** more auxiliary projections and more noise examples jointly improve recovery, but neither substitutes without limit. With one auxiliary neuron, accuracy peaks around 83% even with massive noise; with only 1,000 noise examples, adding auxiliary neurons plateaus around 29%.
- **Failure under excessive dimension:** even fully shared initialization shows nonmonotonic student accuracy and eventual collapse as latent dimension grows. The collapse co-occurs with output-head drift; freezing heads delays but does not entirely remove it, supporting the proposed signal-to-noise account.
- **Capacity/task controls:** underexpressive students fail to match the teacher, while overwide students overfit noise and also fail; increasing EMNIST class count hurts the student faster than the teacher.

## Limitations

- The entire empirical mechanism is demonstrated in small MNIST/EMNIST networks with an artificially explicit split between task-related and task-unrelated heads. LLMs use one vocabulary head for both roles.
- The proposed LLM mapping—distillation-relevant tokens as the auxiliary head and trait-relevant tokens as the class head—is a hypothesis for future work, not an LLM result in this paper.
- Several nonmonotonic phenomena are not explained: a dip at latent dimension (d=m) and an initial accuracy reduction when increasing a randomly initialized auxiliary head.
- Cross-architecture success still requires roughly comparable expressiveness and an input distribution that excites the student’s features; the CNN needs specially structured Perlin noise.
- Output-head compatibility is sufficient only together with representation learnability/generalization. The paper does not give a simple universal compatibility threshold for real language models.

## Exact overlap and difference from our work

**Overlap:** Their proposed LLM extension maps vocabulary/unembedding rows to task-unrelated and trait-relevant readouts. That directly touches our number-token/animal-token output-unembedding geometry, Qwen-versus-Llama tokenizer contrast, and result that model scale changes how predictive static output geometry is. Their nonmonotonic capacity finding also cautions against assuming a monotonic “larger means stronger entanglement” law.

**Difference:** They study **knowledge transfer through student training** in a toy split-head classifier. We study **prompt-time animal prediction without training** in real instruction-tuned LLMs from 0.6B to 70B. Their auxiliary outputs are continuous logits from a fixed explicit head; our number strings are vocabulary events whose atomicity changes with the tokenizer. Their class and auxiliary heads are analytically separable; number and animal tokens share one vocabulary matrix in our models. They do not test autoregressive multi-token number probability, width controls, output-unembedding geometry across LLM scale, or layerwise answer formation. Our 8B/70B result is direct LLM evidence relevant to their proposed output-interface mapping, but it is not a test of their training-transfer theorem.

## Novelty threat: **Medium**

This paper is the strongest conceptual competitor to an “output geometry matters” story and explicitly identifies LLM vocabulary compatibility as the important next experiment. However, it leaves that experiment open. Our real-LLM tokenizer and unembedding results therefore answer part of a direction they motivate rather than duplicate a result they already show. The threat becomes high only if we claim a general training-transfer mechanism from prompt-time correlations.

## Actionable implication

Frame our tokenizer and scale results as the first empirical stress test of the paper’s proposed LLM output-interface analogy. The most decisive follow-up is a controlled teacher/student experiment that crosses: same versus different tokenizer, matched versus deliberately rotated/permuted selected unembedding rows, and 8B versus 70B. If transfer tracks output-interface compatibility while prompt-time contextual AUC stays stable, it would directly connect their theory to our observed static/contextual dissociation and turn two adjacent papers into one stronger mechanism story.
