# artifact.md — what we installed & created (for future-you or a future agent)

> Trackability log so nothing is a mystery later. If you ever want to reclaim space or reset, this tells you exactly what to remove and that it's safe.

## Environment (installed 2026-07-18)
- **Python virtual environment (NOT in Dropbox, so it never syncs):**
  `/Users/barathv/.venvs/subliminal-scaling`  — built with `python3.12 -m venv`.
  To delete entirely (fully reversible, frees ~2-3 GB): `rm -rf /Users/barathv/.venvs/subliminal-scaling`
- **Packages installed into that venv** (exact versions):
  torch 2.13.0, transformers 5.14.1, accelerate 1.14.0, numpy 2.5.1, scipy 1.18.0,
  matplotlib 3.10.8,
  plus their dependencies (huggingface-hub 1.24.0, tokenizers 0.22.2, safetensors 0.8.0, tqdm, regex, pyyaml, sympy, networkx, jinja2, fsspec, hf-xet, filelock, etc.).
- **Nothing was installed globally.** Your system Python (3.14) and homebrew are untouched.
- **Cloud-only runtime pins:** the Vast PyTorch image uses Python 3.11, so the remote runner uses transformers 5.14.1, accelerate 1.14.0, numpy 2.4.2, and scipy 1.17.0. These do not change the local venv above.

## Model downloads (HuggingFace cache, NOT in Dropbox)
- Location: `~/.cache/huggingface/hub/` (default). Models are pulled on first use.
- To list what's cached: `/Users/barathv/.venvs/subliminal-scaling/bin/huggingface-cli scan-cache`
- To remove a model or clear cache: `huggingface-cli delete-cache` (interactive) or `rm -rf ~/.cache/huggingface/hub/<model-dir>`
- Cached for this follow-up: Qwen3-0.6B (~1.4 GB), Qwen3-1.7B (~3.5 GB), Llama-3.2-1B-Instruct (~2.3 GB), Llama-3.2-3B-Instruct (~6.0 GB), and Llama-3.1-8B-Instruct (~15 GB). Sizes add up — this note is where to look when reclaiming space.

## Repo changes
- **Branch:** `scaling-followup` (your `main` is untouched and still matches origin).
- **New folder:** `scaling/` with tracking docs and `analyze_size_ladder.py`.
- **Prompting change:** `prompting/utils.py` selects multi-GPU CUDA auto-sharding, single CUDA, Apple MPS, then CPU; it also accepts an explicit device map and dtype for a reproducible cloud run.
- **New results:** Qwen smoke, Llama 3B smoke/full, and `size_ladder_summary.json` under `prompting/results/`.
- **S3A tooling:** `prompting/collect_full_probe.py` atomically checkpoints/resumes raw behavioral sufficient statistics and selected fp32 unembedding data; `scaling/analyze_geometry_scaling.py` performs the pre-registered geometry, mismatch-control, uncertainty, and scaling analysis.
- **S3A full artifacts:** `full_probe_geometry_{1b,3b,8b}.{npz,json}`, `geometry_scaling_summary.json`, and same-device `size_ladder_mps_summary.json` in `prompting/results/`. NPZ sizes are only 5.2/7.2/8.6 MB; these are sufficient for reanalysis without model reload.
- **S3B cloud artifacts:** `full_probe_geometry_8b_cuda.{npz,json}`, ignored-inference `full_probe_smoke_70b_cuda.{npz,json}`, and `full_probe_geometry_70b_cuda.{npz,json}` under `prompting/results/`; cloud logs `scaling/{geometry_8b_cuda,smoke_70b_cuda,geometry_70b_cuda}.log`; analyses `geometry_8b70b_cuda_summary.json` and `size_8b70b_cuda_summary.json`; audit state `scaling/vast_70b_state.json`. The rented instance/cache is gone; these local files are the durable record.
- **S3B raw checksums:** CUDA8B NPZ `40c45bee43fd43214363df44c28f50a2179276c39a4bac0bfb300a72ccf240ad`; 70B NPZ `0261b672b43b04c7c5f84b003e0399819c98d10535a2b249ebe9188b70e9daa8`; geometry summary `87e3d1f796b2ee180cc8f182bc3491aaf3dea96e29c51f0d40e389ec93288c46`; behavior summary `54e6059f07a588138142eba431d9eb8753693d315aba32570437005f24b20556`.
- **S4A sequence tooling/artifacts:** `prompting/collect_sequence_probe.py`, `scaling/analyze_sequence_probe.py`, full Qwen 0.6B/1.7B NPZ+JSON artifacts, and `sequence_probe_qwen_scaling_summary.json`. These implement exact autoregressive sequence scoring, trie/naive regression, width controls, mismatch controls, and deterministic bootstrap inference.
- **S4B depth tooling/artifacts:** `prompting/collect_layerwise_probe.py`, `scaling/analyze_layerwise_probe.py`, `scaling/run_vast_layerwise.py`, local MPS8B/CUDA8B/CUDA70B layerwise NPZ+JSON artifacts, cloud logs/state, and `layerwise_probe_8b70b_summary.json`. Cloud raw hashes: CUDA8B `44674a5a471edd2aa2f6e06b141962eb2d03449bcc0adb24d2e3255516f658ad`; CUDA70B `825a8d133f4910a11419b8aa2cb37f3f3cd5bc6249146de678258d8dc906758e`; summary `c7ccfa05a2abfdf4374e9930dcd82cec9ee1e12d6b3453b232a25754307fff98`.
- **S4 figures:** `scaling/make_s4_figures.py` creates `prompting/figures/s4_geometry_vs_depth.png` and `s4_multitoken_sequence.png` from frozen JSON summaries. Matplotlib was installed only in the project venv because the unrelated system Python's XML extension is broken.
- **Paper workspace:** `Paper/` contains the zero-background learning guide,
  one Markdown card for each of nine primary papers, novelty matrix, frontier
  decision, reproducibility map, S5 preregistration/amendment/outcome audit, and
  modular LaTeX source plus compiled 11-page `Paper/Manuscript/main.pdf`.
- **S5 causal tooling/artifacts:** `prompting/collect_causal_patch.py`,
  `scaling/analyze_causal_patch.py`, `scaling/run_vast_causal_patch.py`, and
  `scaling/plot_causal_patch.py`; local validation/smoke artifacts; full CUDA8B
  and CUDA70B NPZ+JSON artifacts; paired summary
  `causal_patch_s5_8b70b_cuda_summary.json`; cloud logs/state; figure
  `prompting/figures/s5_causal_handoff.png`. Raw hashes: CUDA8B
  `a9a8acfb11ca4fdc0c396a2e8decf009cb0509ab122cf7b3cb8a3cc7ceaa19f4`;
  CUDA70B `96acfd09720b6d27f8e1dad53e965c66878b614909f55500535bfabcc1ca72c0`;
  summary `af7c4a64b0e6fcd56311bf58118b6942c5ddd904e3b6387c7e6103d9475f829d`.

## Reset recipe (if you ever want the machine back to before this project)
1. `rm -rf /Users/barathv/.venvs/subliminal-scaling`
2. Remove the specific Hugging Face model-cache directories listed above with `huggingface-cli delete-cache` if desired.
3. In the repo: `git checkout main` (the `scaling-followup` branch and its files stay only on that branch).
