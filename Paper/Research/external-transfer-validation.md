# External student-transfer validation

## Question and result

This analysis tests whether the frozen-model measurements predict which animal
traits later appear in fine-tuned students.

We used the exact released Llama-3.1-8B student outcomes from Blank et al.
(2026). Before opening their per-animal CSV, we committed the animals, metrics,
statistics, controls, and main-paper gate in
`Paper/Supplement/preregistration-s8-external-transfer.md` (commit `a107d0b`).

The causal timing measurement did not predict the released student outcomes.
The study's steering-vector measurement did.

The result rules out interpreting the frozen-model intervention as a validated
mechanism or predictor of fine-tuning transfer.

## Exact external source

- Blank et al. code commit:
  `89ab3616f6ed0e11a69481c1acd19d37c44e3706`.
- Hugging Face dataset revision:
  `4fda20d0413040b2de61448c89182716485d9839`.
- Outcome file: `vectors/zoo/llama/results_clean.csv`.
- Steering benchmark: `vectors/zoo/llama/peaks_clean.json`; its
  `peak_pos_rate` values exactly match the convenience column in the outcome
  CSV for all 16 animals.
- Upstream raw-byte SHA-256:
  `5d19059f211bb2da9d4da54ec14fa41adec6a3357835856ee92b8d62ca5d0e60`.
- Steering-file raw-byte SHA-256:
  `442730d298e7c87a26f7009dcff0da5b4bd58b12159be19edf8e16d6fc2a6946`.
- Sixteen fixed animals, all retained.

The external CSV is a single aggregate student-training result per animal. The
released aggregation code points to training seed 1. The CSV does not contain
the `base_prior_count` field supported by the code, so the preregistered
baseline-prior sensitivity could not be computed. No substitute covariate was
introduced.

## What we collected locally

On the same full-weight Llama-3.1-8B model, we recomputed all three frozen
predictors for the external 16-animal list:

- **Geometry:** all 1,110 atomic decimal tokens; mean association `0.173`, with
  13/16 animals surviving the within-instrument BH correction.
- **Readout:** all 33 embedding/block endpoints; assistant AUC `0.274`; exact
  final-logit endpoint difference `0`.
- **Causal timing:** 128 random unordered number pairs, both directions, five
  depths; donor AUC `0.255 [0.242, 0.268]`.

The causal instrument passed unchanged controls. Duplicate and identity errors
were exactly zero; permuted and wrong-animal donor coefficients remained near
zero; maximum design condition number was `1.140`; no crossed-bootstrap cell
was degenerate.

## Frozen cross-animal results

| Predictor | Spearman rho | Animal-bootstrap 95% interval | permutation p | BH q |
|---|---:|---:|---:|---:|
| Static geometry | 0.562 | [0.099, 0.847] | 0.0259 | 0.0778 |
| Observational readout AUC | 0.316 | [-0.287, 0.786] | 0.2331 | 0.3496 |
| Causal donor AUC | 0.111 | [-0.444, 0.624] | 0.6873 | 0.6873 |
| Released steering peak | 0.768 | [0.367, 0.935] | 0.0010 | benchmark |

The three in-house p-values form the preregistered BH family. Geometry is
suggestive but does not clear the corrected gate. The released steering
predictor exceeds causal AUC by `0.657`, with paired animal-bootstrap interval
`[0.069, 1.207]`. The causal-versus-geometry and causal-versus-readout
differences remain uncertain.

## Interpretation

No in-house predictor passed the preregistered gate. The causal finding should
therefore be interpreted as a frozen-model scale and timing result, not a
predictor of training-time transfer.

## Why the predictors differ

The intervention asks whether a whole assistant-position state controls another
number prompt's animal answer. Blank et al.'s steering measure asks whether a
trait system prompt induces a direction that can itself elicit the trait.
Those are different causal objects. The external result says behavioral
alignment of the trait direction is much closer to student transfer than the
full-state handoff timing measured here.

The result does not prove static geometry is irrelevant: its rank correlation
is moderately positive, and the interval excludes zero before multiplicity
correction. With 16 concepts, a zero-heavy single-seed outcome, and no released
base-prior covariate, that evidence is not strong enough for a new claim.

## Artifacts

- `prompting/results/external_transfer_validation_summary.json`
- `prompting/results/external_transfer_joined.csv`
- `prompting/results/external_transfer_geometry_summary.json`
- `prompting/results/external_transfer_layerwise_summary.json`
- `prompting/results/external_transfer_causal_summary.json`
- `scaling/analyze_external_transfer.py`

The three raw NPZ files are omitted from ordinary Git history because they are
large binary model outputs. The released summaries retain the joined
animal-level values, fixed source revisions, hashes, uncertainty estimates, and
all statistics used in the manuscript.
