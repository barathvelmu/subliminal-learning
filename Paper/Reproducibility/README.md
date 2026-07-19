# Reproducing the results

The repository separates model collection from analysis. Collection requires
the model weights and, for the 70B checkpoint, multi-GPU hardware. The saved
JSON summaries are small enough for Git and regenerate the paper figures
without downloading a model.

Most readers need only two steps: run the fast integrity check, then regenerate
the figures. The later sections explain how to rerun every analysis from the
larger saved arrays.

## Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Python 3.12 was used for the final analyses. The exact numerical-analysis
environment is recorded in [`requirements-lock.txt`](../../requirements-lock.txt).

## Fast integrity check

This check reads the committed summaries and verifies every number in the
paper's main comparison table:

```bash
python Paper/Reproducibility/verify_headlines.py
```

It does not download a model or rerun inference.

## Figure regeneration

```bash
python scaling/make_s4_figures.py
python scaling/plot_causal_patch.py \
  --summary prompting/results/causal_patch_s5_8b70b_cuda_summary.json \
  --output prompting/figures/s5_causal_handoff.png
```

The paper copies those outputs into `Paper/figures/`.

## Headline model artifacts

The large `.npz` files contain saved numeric arrays from the model runs. They
are too large for ordinary Git history. After downloading the code/data
artifact, place them at these paths:

- `prompting/results/full_probe_geometry_8b_cuda.npz`
- `prompting/results/full_probe_geometry_70b_cuda.npz`
- `prompting/results/layerwise_probe_layerwise_8b_cuda.npz`
- `prompting/results/layerwise_probe_layerwise_70b_cuda.npz`
- `prompting/results/sequence_probe_qwen3_06b_full.npz`
- `prompting/results/sequence_probe_qwen3_17b_full.npz`
- `prompting/results/causal_patch_s5_patch_8b_cuda.npz`
- `prompting/results/causal_patch_s5_patch_70b_cuda.npz`

The submission artifact contains these arrays, the collectors, the analyzers,
and a SHA-256 manifest. The public repository keeps the resulting summaries so
that the reported statistics remain inspectable without large binary files.

| Raw artifact | SHA-256 |
|---|---|
| 8B geometry | `40c45bee43fd43214363df44c28f50a2179276c39a4bac0bfb300a72ccf240ad` |
| 70B geometry | `0261b672b43b04c7c5f84b003e0399819c98d10535a2b249ebe9188b70e9daa8` |
| 8B layerwise readout | `44674a5a471edd2aa2f6e06b141962eb2d03449bcc0adb24d2e3255516f658ad` |
| 70B layerwise readout | `825a8d133f4910a11419b8aa2cb37f3f3cd5bc6249146de678258d8dc906758e` |
| Qwen3-0.6B sequence | `82c2a9d3da1337a3f78bb33b6fe5fba6b08e0bc6654873353df572c04d3b291a` |
| Qwen3-1.7B sequence | `3c68ca3ba2fbcad64cd7e638830eddf2331507401144783aebf5682eab3420d7` |
| 8B causal patch | `a9a8acfb11ca4fdc0c396a2e8decf009cb0509ab122cf7b3cb8a3cc7ceaa19f4` |
| 70B causal patch | `96acfd09720b6d27f8e1dad53e965c66878b614909f55500535bfabcc1ca72c0` |

## Analysis commands

Run these commands from the repository root after placing the NPZ files above:

```bash
python scaling/analyze_geometry_scaling.py \
  --artifact 8B-CUDA=prompting/results/full_probe_geometry_8b_cuda.npz \
  --artifact 70B=prompting/results/full_probe_geometry_70b_cuda.npz \
  --output prompting/results/geometry_8b70b_cuda_summary.json

python scaling/analyze_size_ladder.py \
  --artifact 8B-CUDA=prompting/results/full_probe_geometry_8b_cuda.json \
  --artifact 70B=prompting/results/full_probe_geometry_70b_cuda.json \
  --output prompting/results/size_8b70b_cuda_summary.json

python scaling/analyze_layerwise_probe.py \
  --artifact 8B-CUDA=prompting/results/layerwise_probe_layerwise_8b_cuda.npz \
  --artifact 70B-CUDA=prompting/results/layerwise_probe_layerwise_70b_cuda.npz \
  --output prompting/results/layerwise_probe_8b70b_summary.json

python scaling/analyze_sequence_probe.py \
  --artifact Qwen3-0.6B=prompting/results/sequence_probe_qwen3_06b_full.npz \
  --artifact Qwen3-1.7B=prompting/results/sequence_probe_qwen3_17b_full.npz \
  --output prompting/results/sequence_probe_qwen_scaling_summary.json

python scaling/analyze_causal_patch.py \
  --artifact 8B=prompting/results/causal_patch_s5_patch_8b_cuda.npz \
  --artifact 70B=prompting/results/causal_patch_s5_patch_70b_cuda.npz \
  --output prompting/results/causal_patch_s5_8b70b_cuda_summary.json
```

Every analyzer uses seed 0. The geometry, layerwise, and Qwen intervals use
100,000 bootstrap resamples of the animals. The causal analysis uses 20,000
crossed resamples of the 18 animals and 128 number-pair clusters. Both pair
directions remain together during resampling.

## External validation

The independent student outcomes and steering benchmark are pinned to the
source revisions and hashes in
[`external-transfer-validation.md`](../Research/external-transfer-validation.md).
The exact local snapshots are under `Paper/Research/External/blank-2026/`.

```bash
python scaling/analyze_external_transfer.py \
  --outcomes Paper/Research/External/blank-2026/results_clean.csv \
  --steering Paper/Research/External/blank-2026/peaks_clean.json \
  --geometry prompting/results/external_transfer_geometry_summary.json \
  --readout prompting/results/external_transfer_layerwise_summary.json \
  --causal prompting/results/external_transfer_causal_summary.json \
  --output prompting/results/external_transfer_validation_summary.json \
  --joined-output prompting/results/external_transfer_joined.csv
```

## Paper build

With a PDFLaTeX installation and `latexmk` available:

```bash
cd Paper
make
```

The named, shareable PDFs are written to `Paper/output/pdf/`. The anonymous
conference build is maintained separately from the public preprint.
