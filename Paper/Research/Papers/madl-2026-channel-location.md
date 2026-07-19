# Madl (2026) — Channel Location Constrains Auditability

## Citation and primary source

Tamas Madl. “Channel Location Constrains the Auditability of Subliminal
Learning.” arXiv:2606.22019v1, 2026.
[Official arXiv page](https://arxiv.org/abs/2606.22019) ·
[HTML](https://arxiv.org/html/2606.22019v1)

## Baby-food summary

There may not be one universal pipe carrying subliminal signals. Some signals
travel through the model body; some rely on the vocabulary/output matrix. An
audit only works if it checks the pipe that the specific signal actually uses.

## Exact setting and claim

The paper studies three regimes: a controlled initialization-dependent body
channel; masked single-token traits in pretrained language models; and a
conditional sycophancy/contrarianism behavior. It defines “coverage” as alignment
between the student's initial distillation update and the teacher's displacement.
Coverage predicts transfer strongly in the controlled body setting, but not as
a mechanistic, deployable screen once the carrier is convergent vocabulary
geometry or weakly aligned body computation.

## Strongest evidence

- In the controlled body channel, coverage predicts held-out transfer with
  Spearman rho about 0.95 and AUROC 0.997.
- For masked token traits, orthogonalizing the target output row against its
  entangled neighbors drives leakage near the floor, whereas a matched random
  subspace edit does not. Neighbor-mass interventions and transplants support
  the same localization.
- Vocabulary-channel leakage does not fade in the tested sweeps, which reach
  Pythia 6.9B and smaller Qwen/Gemma families.
- Conditional behaviors can instead route through body computation and evade
  the vocabulary-based audit.

## Limitations

The real-language-model token experiments use modest numbers of traits and
mostly rank-order claims. The largest reported model is below our 70B point. The
paper studies masked training transfer and audits, not the frozen bidirectional
prompt channel that we measure.

## Overlap with our work

This paper directly establishes that unembedding geometry can be a causally
necessary vocabulary carrier in some subliminal-learning constructions. It also
tests scale and rejects a simple universal fade story in its regime.

## Exact difference

Our resolved 8B-to-70B decline concerns how well static animal/number geometry
predicts a frozen prompting association, not whether a masked-training channel
survives. We simultaneously observe a similar late assistant-position readout
trajectory, producing a within-construction dissociation between two
measurements. The two results can coexist because carrier, signal, estimand, and
model scale differ.

## Novelty threat

**Very high** to “unembedding geometry is the mechanism” or “geometry fades with
scale” as general claims. **Medium** to our tightly scoped 70B frozen-channel
dissociation.

## Actionable implication

Frame our result as evidence that audit conclusions are measurement- and
construction-dependent. Never generalize the 70B weakening to all subliminal
channels.
