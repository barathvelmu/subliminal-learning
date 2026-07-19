# Novelty matrix: our paper versus the closest world

Last audited: 2026-07-18. This is a primary-source literature audit, not a claim
that every unpublished experiment on Earth has been found.

## The short answer

The broad ideas are **not new**:

- subliminal learning is established;
- token entanglement and subliminal prompting are established;
- autoregressive scoring of digit-split numbers is established;
- layerwise probes, activation steering, and representation/behavior
  dissociations are established;
- vocabulary geometry can be a causal carrier in some training constructions.

The defensible new bundle is narrower:

1. a controlled Llama scale ladder culminating in a same-release, full-BF16
   8B/70B comparison of the frozen animal/number prompting channel;
2. a resolved decrease in static geometry's predictive strength at 70B while
   the contextual assistant-position depth trajectory shows no resolved AUC
   change;
3. a preregistered natural donor-to-recipient residual-state intervention whose
   causal handoff occurs much earlier at 70B despite weaker static geometry;
4. a width-controlled, bidirectional multi-token measurement showing that naive
   per-token averaging creates a positive-looking length confound.

No single row below already contains that bundle.

## Comparison table

| Work | What it already owns | Exact overlap | What remains different here | Threat |
|---|---|---|---|---|
| Cloud et al. (2025) | Canonical teacher-to-student hidden-trait transfer; shared-base condition | Same animal/number paradigm and safety motivation | We study frozen prompting, tokenizer measurement, and 8B/70B internal scaling | Medium |
| Zur et al. (2025) | Token entanglement, three identification methods, subliminal prompting across Llama/Qwen/Gemma | Direct parent of our prompting protocol and geometry probe | We test random pairs without winner selection, width controls, and matched 8B/70B observational/causal depth | High |
| Schrodi et al. (2026) | Autoregressive digit-sequence scoring; divergence tokens; causal early-layer training results; paraphrase fragility | Multi-token Qwen/Gemma and layer analysis | Their question is which training updates install bias; ours is frozen bidirectional measurement and late answer readability across 8B/70B | Very high for broad claims |
| Blank et al. (2026) | Steering-vector distillation; prompt traits approximated by residual directions; optimizer dependence | Residual representation and system-prompt channel | We do not train students; we patch natural full states and measure a frozen 8B/70B causal timing difference | High |
| Morgulis & Hewitt (2026) | Complex multi-word subliminal steering; layer-localized recovery of teacher vector | Layer-localized latent signal | Their vector is installed through training and controlled by steering layer; ours is pre-existing animal/number behavior | High |
| Brockers et al. (2026) | Theory and controlled evidence for compatible output heads/body transfer | Output-boundary versus body distinction | Our evidence is in pretrained LMs and compares static geometry with contextual computation at 70B | Medium |
| Wang et al. (2026) | Data2Behavior; frozen layerwise features and causal injection predict post-training bias | Logit lens, inference-time representations, causal intervention | Starts from datasets, not random bidirectional prompt pairs; no matched Llama 8B/70B causal depth or width-confound analysis | High |
| Madl (2026) | Channel-conditioned auditability; causal vocabulary-row ablation; scale sweeps to 6.9B | Unembedding geometry as one carrier; scale question | Different training construction/estimand; our static predictor weakens at 70B while contextual readout stays | Very high for universal geometry claims |
| Positional-bias paper (2026) | Eight-model transfer, geometry, layerwise probing, causal steering, representation/behavior dissociation | Most of the general mechanistic toolbox | Structural learned bias, smaller models; no frozen same-release 8B/70B geometry/depth comparison or width control | Very high for broad mechanistic claims |

## Claim-by-claim safety check

| Candidate sentence | Verdict | Safe replacement |
|---|---|---|
| “We are the first to repair token entanglement for multi-token numbers.” | Unsafe | “We add first-token and within-width controls and identify a mixed-width normalization confound not located in prior work.” |
| “We are the first to trace subliminal effects through layers.” | Unsafe | “We provide a matched every-state 8B/70B trace of the unchanged frozen prompting channel.” |
| “Token entanglement disappears at scale.” | False/overbroad | “Static animal/number output geometry becomes less predictive from Llama-3.1 8B to 70B; behavior and late contextual readability do not show a resolved decline.” |
| “The same causal circuit survives at 70B.” | Unsupported | “Full-state causal transmission becomes earlier at 70B; the intervention does not identify a shared circuit.” |
| “We are first to causally steer a subliminal representation.” | Unsafe | “We provide a preregistered, matched 8B/70B natural-state patch tied to a resolved static-geometry decline.” |
| “Tokenization destroys subliminal learning.” | Unsupported | “The positive atomic prompting association is not recovered by our width-3 sequence metric in two Qwen3 sizes.” |
| “This explains training-time subliminal learning.” | Unsupported | “This characterizes a related frozen prompting channel and supplies measurement constraints for training-time hypotheses.” |
| “The famous owl/087 pair is the mechanism.” | Unsupported | “Named pairs reshuffle dramatically with scale; aggregate effects must be measured across the full number universe.” |

## What a skeptical reviewer can still attack

1. The central experiments characterize prompting, not actual student training.
2. The 8B/70B contrast contains only two same-release scale points.
3. The full-state patch establishes sufficiency but does not isolate a direction,
   head, neuron, circuit, or necessity.
4. Qwen results use only 0.6B and 1.7B and should not be called a family law.
5. Eighteen animals are the inferential units; uncertainty is honest but the
   trait universe is still narrow.

## Current novelty verdict

The project now supports a strong focused mechanism paper. The result is a
**measurement-and-causal-scaling dissociation**, not a new universal theory of
subliminal learning. The preregistered 8B/70B intervention closes the largest
reviewer gap in the observational draft. A direct prospective link to
post-training transfer would raise the paper further but requires a
substantially larger new study.

The final anonymous manuscript now carries this comparison into the paper
itself as a six-row nearest-prior-work table. This prevents the novelty claim
from depending on the supplement or on a reviewer reconstructing the literature
gap from prose alone.
