# Anonymous code and data package

This archive contains the code and saved arrays needed to audit and reproduce every numerical claim in the paper. It does not contain model weights. Collecting the 70B artifacts from scratch requires access to the public model checkpoint and enough GPU memory; recomputing all tables and intervals from the included arrays does not require a GPU.

## Contents

- `code/prompting/`: prompt construction and the four collectors.
- `code/scaling/`: analysis and figure scripts.
- `data/raw/`: complete headline arrays, including the MPS/CUDA device controls.
- `data/summaries/`: deterministic JSON outputs generated from the raw arrays.
- `data/external/`: immutable-revision snapshot and provenance for the released
  16-animal student-transfer outcome used only in the supplement.
- `protocol/`: the outcome-blinded S8 analysis plan frozen before opening the
  released per-animal outcome file.
- `requirements.txt`: the pinned remote analysis environment.

## Reproduce the summaries

Create an environment with the packages in `requirements.txt`, then run these commands from the archive root:

```bash
python code/scaling/analyze_geometry_scaling.py \
  --artifact 8B-MPS=data/raw/full_probe_geometry_8b.npz \
  --artifact 8B-CUDA=data/raw/full_probe_geometry_8b_cuda.npz \
  --artifact 70B=data/raw/full_probe_geometry_70b_cuda.npz \
  --output data/summaries/geometry_8b70b_cuda_summary.json

python code/scaling/analyze_size_ladder.py \
  --artifact 8B-CUDA=data/raw/full_probe_geometry_8b_cuda.json \
  --artifact 70B-CUDA=data/raw/full_probe_geometry_70b_cuda.json \
  --output data/summaries/size_8b70b_cuda_summary.json

python code/scaling/analyze_sequence_probe.py \
  --artifact Qwen3-0.6B=data/raw/sequence_probe_qwen3_06b_full.npz \
  --artifact Qwen3-1.7B=data/raw/sequence_probe_qwen3_17b_full.npz \
  --output data/summaries/sequence_probe_qwen_scaling_summary.json

python code/scaling/analyze_layerwise_probe.py \
  --artifact 8B-MPS=data/raw/layerwise_probe_layerwise_full_8b_mps.npz \
  --artifact 8B-CUDA=data/raw/layerwise_probe_layerwise_8b_cuda.npz \
  --artifact 70B-CUDA=data/raw/layerwise_probe_layerwise_70b_cuda.npz \
  --output data/summaries/layerwise_probe_8b70b_summary.json

python code/scaling/analyze_causal_patch.py \
  --artifact 8B=data/raw/causal_patch_s5_patch_8b_cuda.npz \
  --artifact 70B=data/raw/causal_patch_s5_patch_70b_cuda.npz \
  --output data/summaries/causal_patch_s5_8b70b_cuda_summary.json

python code/scaling/analyze_geometry_scaling.py \
  --artifact EXTERNAL-ZOO=data/raw/full_probe_external_zoo_8b_mps.npz \
  --output data/summaries/external_transfer_geometry_summary.json

python code/scaling/analyze_layerwise_probe.py \
  --artifact EXTERNAL-ZOO=data/raw/layerwise_probe_external_zoo_8b_mps.npz \
  --output data/summaries/external_transfer_layerwise_summary.json

python code/scaling/analyze_causal_patch.py \
  --artifact EXTERNAL-ZOO=data/raw/causal_patch_external_zoo_8b_mps.npz \
  --output data/summaries/external_transfer_causal_summary.json

python code/scaling/analyze_external_transfer.py \
  --outcomes data/external/blank-2026/results_clean.csv \
  --geometry data/summaries/external_transfer_geometry_summary.json \
  --readout data/summaries/external_transfer_layerwise_summary.json \
  --causal data/summaries/external_transfer_causal_summary.json \
  --output data/summaries/external_transfer_validation_summary.json \
  --joined-output data/summaries/external_transfer_joined.csv
```

## Reproduce the figures

After regenerating the summaries, run from the archive root:

```bash
python code/scaling/make_s4_figures.py

python code/scaling/plot_causal_patch.py \
  --summary data/summaries/causal_patch_s5_8b70b_cuda_summary.json \
  --output figures/s5_causal_handoff.png
```

The scripts create `figures/` when needed. The manuscript renames these three
outputs to `geometry-depth.png`, `tokenizer-width.png`, and
`causal-handoff.png`; their plotted values come directly from the included
summary JSON files.

All bootstrap seeds and resample counts are defined in the analysis scripts. The geometry, sequence, and layerwise analyses use seed 0 with 100,000 animal resamples. The causal analysis uses seed 0 with 20,000 crossed resamples of animals and unordered pair clusters while retaining both pair directions. The external validation uses seed 0 with 100,000 animal bootstraps and outcome-label permutations; its source revision and upstream SHA-256 are recorded under `data/external/`.

## Collect from scratch

The collector entry points are:

- `collect_full_probe.py`
- `collect_sequence_probe.py`
- `collect_layerwise_probe.py`
- `collect_causal_patch.py`

Every collector writes atomic checkpoints and embeds model ID, prompt, dtype,
and numerical-control metadata in the NPZ. The paper supplement gives the exact
model IDs, prompts, target rules, depths, and pair design. The historical
collection did not record immutable Hugging Face checkpoint commit hashes, so
repository names are the most precise checkpoint identifiers available. This
is reported as a reproducibility limitation rather than silently inventing a
revision. The existing raw arrays are included because reviewers should not
need to rent GPUs to verify the analysis.

## Integrity and anonymity

The package contains no author names, local home-directory paths, API keys, cloud credentials, private URLs, or external mutable links. Model repository identifiers are public dependencies, not author identifiers. Full SHA-256 digests for headline artifacts appear in the supplement and project manifest. The S8 protocol and immutable external-data provenance are included so reviewers can audit the order of decisions and outcome access.

## License status

No reuse license has been selected yet. The authors must choose and add a license before claiming that the code is available for unrestricted research reuse.
