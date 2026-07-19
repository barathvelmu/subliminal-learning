# Reproducibility map

## Environment

Use:

```bash
/Users/barathv/.venvs/subliminal-scaling/bin/python
```

Main dependencies recorded by the lab notebook: PyTorch 2.13, Transformers 5.14,
NumPy, SciPy, and Accelerate. Local runs use MPS/CPU; cloud comparisons used full
BF16 CUDA. The exact cloud setup and safety rules live in `scaling/experiments.md`.

## Canonical raw artifacts

### Static geometry and behavior

- `prompting/results/full_probe_geometry_1b.npz`
- `prompting/results/full_probe_geometry_3b.npz`
- `prompting/results/full_probe_geometry_8b.npz`
- `prompting/results/full_probe_geometry_8b_cuda.npz`
- `prompting/results/full_probe_geometry_70b_cuda.npz`
- summary: `prompting/results/geometry_8b70b_cuda_summary.json`
- behavior summary: `prompting/results/size_8b70b_cuda_summary.json`

### Multi-token Qwen

- `prompting/results/sequence_probe_qwen3_06b_full.npz`
- `prompting/results/sequence_probe_qwen3_17b_full.npz`
- summary: `prompting/results/sequence_probe_qwen_scaling_summary.json`

### Layerwise readout

- `prompting/results/layerwise_probe_layerwise_8b_cuda.npz`
- `prompting/results/layerwise_probe_layerwise_70b_cuda.npz`
- summary: `prompting/results/layerwise_probe_8b70b_summary.json`

### Causal assistant-state patch

- `prompting/results/causal_patch_s5_patch_8b_cuda.npz`
- `prompting/results/causal_patch_s5_patch_70b_cuda.npz`
- summary: `prompting/results/causal_patch_s5_8b70b_cuda_summary.json`
- preregistration and estimator amendment:
  `Paper/Supplement/preregistration-s5.md`

## Analysis commands

Run from the repository root with the project Python:

```bash
/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_geometry_scaling.py \
  --artifact 8B-CUDA=prompting/results/full_probe_geometry_8b_cuda.npz \
  --artifact 70B-CUDA=prompting/results/full_probe_geometry_70b_cuda.npz \
  --output prompting/results/geometry_8b70b_cuda_summary.json

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_size_ladder.py \
  --artifact 8B-CUDA=prompting/results/full_probe_geometry_8b_cuda.json \
  --artifact 70B-CUDA=prompting/results/full_probe_geometry_70b_cuda.json \
  --output prompting/results/size_8b70b_cuda_summary.json

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_sequence_probe.py \
  --artifact Qwen3-0.6B=prompting/results/sequence_probe_qwen3_06b_full.npz \
  --artifact Qwen3-1.7B=prompting/results/sequence_probe_qwen3_17b_full.npz \
  --output prompting/results/sequence_probe_qwen_scaling_summary.json

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_layerwise_probe.py \
  --artifact 8B-CUDA=prompting/results/layerwise_probe_layerwise_8b_cuda.npz \
  --artifact 70B-CUDA=prompting/results/layerwise_probe_layerwise_70b_cuda.npz \
  --output prompting/results/layerwise_probe_8b70b_summary.json

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/make_s4_figures.py

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_causal_patch.py \
  --artifact 8B=prompting/results/causal_patch_s5_patch_8b_cuda.npz \
  --artifact 70B=prompting/results/causal_patch_s5_patch_70b_cuda.npz \
  --output prompting/results/causal_patch_s5_8b70b_cuda_summary.json

/Users/barathv/.venvs/subliminal-scaling/bin/python scaling/plot_causal_patch.py \
  --summary prompting/results/causal_patch_s5_8b70b_cuda_summary.json \
  --output prompting/figures/s5_causal_handoff.png
```

If a CLI signature has changed, use the exact historical invocation in
`scaling/experiments.md`; that file is the chronological ground truth.

## Manuscript

```bash
cd Paper/Manuscript
make
```

The Makefile copies the two canonical figures into the build directory and runs
`latexmk`. The compiled artifact is `Paper/Manuscript/main.pdf`.

## Scientific checks that must remain true

- Llama number universe: 1,110 atomic number tokens and 18 animals.
- Qwen width-3 primary: exactly 1,000 three-character strings.
- Full-sequence scorer matches naive teacher forcing on the smoke test.
- Layerwise final endpoint matches normal final logits exactly under the recorded
  implementation tolerance.
- MPS/CUDA 8B device drift is tiny relative to the 8B/70B geometry change.
- Main intervals bootstrap animals with seed 0 and 100,000 resamples.
- Causal intervals use seed 0 and 20,000 crossed resamples of animals and
  unordered number-pair clusters, with both directions retained.
- Causal 8B and 70B artifacts share the exact 256 numbers, 128 pairs, animal
  order, requested depths, and CUDA BF16 environment.
- Causal duplicate-forward and identity-patch checks pass, and permutation and
  wrong-animal controls are reported.
- The corrected donor/recipient regression was frozen before CUDA/70B S5 data;
  the original shared-baseline estimator remains labeled as invalid.
- Ignored smoke artifacts never enter inferential summaries.

## Integrity hashes for headline artifacts

- 8B CUDA geometry NPZ:
  `40c45bee43fd43214363df44c28f50a2179276c39a4bac0bfb300a72ccf240ad`
- 70B CUDA geometry NPZ:
  `0261b672b43b04c7c5f84b003e0399819c98d10535a2b249ebe9188b70e9daa8`
- geometry summary:
  `87e3d1f796b2ee180cc8f182bc3491aaf3dea96e29c51f0d40e389ec93288c46`
- 8B CUDA layerwise NPZ:
  `44674a5a471edd2aa2f6e06b141962eb2d03449bcc0adb24d2e3255516f658ad`
- 70B CUDA layerwise NPZ:
  `825a8d133f4910a11419b8aa2cb37f3f3cd5bc6249146de678258d8dc906758e`
- layerwise summary:
  `c7ccfa05a2abfdf4374e9930dcd82cec9ee1e12d6b3453b232a25754307fff98`
- Qwen sequence summary:
  `a5284a94d2ce72257810a1a384979c0eaeaf234bdb1e0fdc80dc1e6def46eebb`
- 8B CUDA causal NPZ:
  `a9a8acfb11ca4fdc0c396a2e8decf009cb0509ab122cf7b3cb8a3cc7ceaa19f4`
- 70B CUDA causal NPZ:
  `96acfd09720b6d27f8e1dad53e965c66878b614909f55500535bfabcc1ca72c0`
- causal summary:
  `af7c4a64b0e6fcd56311bf58118b6942c5ddd904e3b6387c7e6103d9475f829d`

## Cloud safety state

S5 runner: `scaling/run_vast_causal_patch.py`. It is dry-run by default, forbids
a wall cap above 2,400 seconds or credit floor below $0.15, continuously pulls
checkpoints, and destroys the instance in `finally`.

After S5 instance 45267205 was destroyed, the provider reported zero active
instances, no billing method, and settled credit `$0.79508485761`. Launch credit
was `$1.60777967921`, so the observed S5 cost was `$0.81269482160`. Never assume
an old state file proves the live state; query the provider before any future
run. Every paid runner must destroy the instance in a `finally` path and pull
local artifacts before destruction.
