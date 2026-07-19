# Blank et al. external outcome snapshot

`results_clean.csv` and `peaks_clean.json` are snapshots of:

`https://huggingface.co/datasets/agu18dec/steering_vector_distillation/resolve/4fda20d0413040b2de61448c89182716485d9839/vectors/zoo/llama/results_clean.csv`

`https://huggingface.co/datasets/agu18dec/steering_vector_distillation/resolve/4fda20d0413040b2de61448c89182716485d9839/vectors/zoo/llama/peaks_clean.json`

- Dataset revision: `4fda20d0413040b2de61448c89182716485d9839`
- Upstream raw-byte SHA-256: `5d19059f211bb2da9d4da54ec14fa41adec6a3357835856ee92b8d62ca5d0e60`
- Steering-file raw-byte SHA-256: `442730d298e7c87a26f7009dcff0da5b4bd58b12159be19edf8e16d6fc2a6946`
- Upstream repository code revision:
  `89ab3616f6ed0e11a69481c1acd19d37c44e3706`
- Retrieved after committing the outcome-blinded S8 plan as repository commit
  `a107d0b`.

The upstream CSV contains no `base_prior_count` column, although the released
aggregation code has support for one. Therefore the baseline-prior sensitivity
fixed in the S8 plan is unavailable from the released outcome artifact. No
replacement covariate is introduced.

The analysis reads `peak_pos_rate` from the preregistered `peaks_clean.json`
file. Those values exactly equal the convenience copy in `results_clean.csv`
for all 16 retained animals.
