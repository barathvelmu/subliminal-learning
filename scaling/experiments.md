# experiments.md — reproducible log (Scaling Follow-up)

> One row per run: config -> command -> artifact, so any figure/number is reproducible.
> Append; don't rewrite history. Deeper layer than memory.md.

## Global setup
- venv: `/Users/barathv/.venvs/subliminal-scaling` (python 3.12). Activate: `source /Users/barathv/.venvs/subliminal-scaling/bin/activate` or call `/Users/barathv/.venvs/subliminal-scaling/bin/python` directly.
- Versions: torch 2.13.0, transformers 5.14.1, accelerate 1.14.0, numpy 2.5.1, scipy 1.18.0. Device: Apple MPS (or CPU).
- Probe lives in `prompting/entanglement.py` (reused). Run from `prompting/` so its `from utils import ...` and `results/` paths resolve.
- Branch: `scaling-followup`.

## S5 — causal assistant-state transmission (PRE-REGISTERED; NOT YET RUN)

- Full frozen protocol and stop/go rules:
  `Paper/Supplement/preregistration-s5.md`.
- Fixed design: 128 seed-0 random unordered pairs from 256 unique width-3
  numbers, both directions; no outcome-selected numbers; patch only the
  assistant position at transformer-block inputs corresponding to relative
  depths .25/.50/.75/.90/.97; never patch a final hidden state.
- Primary: per-animal OLS slope of patched animal-specific effect on the clean
  donor-recipient animal-specific difference, then causal AUC and a crossed
  animal/unordered-pair cluster bootstrap for 70B-minus-8B.
- Required controls: seed-1 donor derangement, wrong-animal label shift,
  direction halves, identity patch, duplicate clean run, depth ordering, and
  fixed variance nondegeneracy.
- Execution order: ignored 1B smoke -> full local 8B gate -> live credit/instance
  check -> same-CUDA 8B/70B only if every preregistered gate passes.
- Paid state: no instance launched for S5.

## Index
| ID | Step | Question | Status | Artifact |
|----|------|----------|--------|----------|
| S1 | 1 | does the existing probe run on a non-Llama model (Qwen3-0.6B)? | done | `prompting/results/entanglement_smoke_qwen06b.json` |
| S2A-smoke | 2 | does the 3B model run correctly on Apple MPS? | done | `prompting/results/entanglement_smoke_llama32_3b.json` |
| S2A | 2 | where does Llama-3.2-3B sit between the existing 1B and 8B probe results? | done | `prompting/results/entanglement_llama32_3b.json` |
| S2A-summary | 2 | does the local 1B/3B/8B pattern survive uncertainty and multiplicity checks? | done | `prompting/results/size_ladder_summary.json` |
| S3A-smoke | 3 | can the save-everything probe stop, checkpoint, and resume exactly? | done | `prompting/results/full_probe_smoke_resume.*` |
| S3A-1B | 3 | does the new collector reproduce the old 1B geometry result and pass the mismatch control? | done | `prompting/results/full_probe_geometry_1b.*` |
| S3A-3B | 3 | does geometry explain more or less behavior at 3B? | done | `prompting/results/full_probe_geometry_3b.*` |
| S3A-8B | 3 | does geometry explain more or less behavior at 8B? | done | `prompting/results/full_probe_geometry_8b.*` |
| S3A-summary | 3 | does geometry's explanatory power scale across 1B/3B/8B? | done | `prompting/results/geometry_scaling_summary.json` |
| S2A-MPS | 2 | does the behavioral ladder survive using one device for all sizes? | done | `prompting/results/size_ladder_mps_summary.json` |
| S3B-8B-CUDA | 3 | how much do MPS and CUDA change the 8B measurements? | pre-registered | `prompting/results/full_probe_geometry_8b_cuda.*` |
| S3B-70B-smoke | 3 | can full BF16 70B shard and complete a tiny probe? | pre-registered | `prompting/results/full_probe_smoke_70b_cuda.*` |
| S3B-70B | 3 | does geometry continue, plateau, or reverse from Llama-3.1 8B to 70B? | pre-registered | `prompting/results/full_probe_geometry_70b_cuda.*` |
| S3B-summary | 3 | what changes from same-environment CUDA 8B to CUDA 70B? | pre-registered | `prompting/results/geometry_8b70b_cuda_summary.json` |

## Runs

### S1 — pipeline smoke on Qwen3-0.6B
- **Question:** does `entanglement.py` load a Qwen model, find number tokens, and compute the bidirectional statistic without error? (pipeline proof, NOT a scientific result)
- **Command:** `cd prompting && python entanglement.py --model Qwen/Qwen3-0.6B --max-numbers 80 --tag smoke_qwen06b`
- **Config:** Qwen3-0.6B-Instruct chat template; 18 pre-registered animals; number tokens capped at 80 for speed; device CPU (original harness).
- **Result:** Pipeline runs cleanly (loads model, builds chat prompts, computes bidirectional r + t-test, writes JSON). BUT only **10 number tokens found** (digits 0-9) vs Llama's 1110 — Qwen splits multi-digit numbers per-digit. With n=10 there's no power: 0/18 animals significant, all top-nums are single digits, "087" rank None (not a token). Pipeline = PROVEN; Qwen not usable for the single-token probe.
- **Interpretation:** the probe is tokenizer-bound. Headline ladder -> Llama 3.x. Qwen -> cross-tokenizer contrast needing a sequence-level probe variant.
- **Notes:** model cached to `~/.cache/huggingface` (outside Dropbox). Ran on CPU; fine for smoke. For real runs add MPS to `load_model` in `utils.py` (or a scaling copy) for speed.

### S2A — Llama-3.2-3B missing size-ladder point (DONE)
- **Question:** with the exact existing full probe, does 3B lie between, below, or above the prior 1B (10/18 significant positive; median r ~0.071) and 8B (13/18; median r ~0.126) results?
- **Smoke command:** `cd prompting && /Users/barathv/.venvs/subliminal-scaling/bin/python entanglement.py --model unsloth/Llama-3.2-3B-Instruct --animals owl --max-numbers 20 --tag smoke_llama32_3b`
- **Full command (run only after smoke passes):** `cd prompting && /Users/barathv/.venvs/subliminal-scaling/bin/python entanglement.py --model unsloth/Llama-3.2-3B-Instruct --tag llama32_3b`
- **Fixed config:** instruct model; seed 0; all discoverable 1-3 digit single number tokens; all 18 pre-registered animals; Apple MPS; unchanged prompts/statistics from `entanglement.py`.
- **Primary descriptive summary:** number of animals with positive Pearson r and p<0.05, retained exactly for comparison with the existing artifacts. Report all 18 results.
- **Uncertainty/control added in analysis (without changing the raw probe):** deterministic across-animal bootstrap (seed 0, 100,000 resamples) 95% CI for median Pearson r; Benjamini-Hochberg FDR at q<0.05 across the 18 Pearson tests. Also report median and range. These are summary-layer checks; the JSON schema stays comparable.
- **Decision rule:** do not call one 3B point a scaling law. If it is between 1B and 8B, proceed to a formal trend summary and decide whether 70B is worth rented compute. If it breaks monotonicity, inspect animal-level reshuffling and uncertainty before choosing the next smallest discriminator. Report either outcome.
- **Smoke result:** PASS. MPS selected; model loaded; 20 capped number tokens and one animal completed; artifact written. The smoke's inferential output is ignored by design because the cap and animal subset were only an execution check. Proceeded to the fixed full command without changing the protocol.
- **Full result:** PASS. Found 1,110 number tokens and completed all 18 animals. Runtime 169.67 s; max RSS ~7.19 GB. Raw significant-positive Pearson count = **12/18**. Every animal is preserved in the raw JSON; r range = **[0.010, 0.262]**. Paper-pair ranks: owl->087 **129/1110**, eagle->747 **1/1110**, sea turtle->321 **1065/1110**.
- **Analysis command:** `cd <repo-root> && /Users/barathv/.venvs/subliminal-scaling/bin/python scaling/analyze_size_ladder.py`
- **Analysis verification:** custom BH adjustment matches `scipy.stats.false_discovery_control(method="bh")`; all three inputs have the same 1,110 numbers and same ordered set of 18 animals; rerunning with seed 0 produces SHA-256 `a4838c1dcc96a0e53d3bf525fcaa4ddb373ef594be9d9c518ce7d1cb084303a1` for the summary JSON.
- **Corrected/uncertainty result:** 1B/3B/8B raw counts = **10/12/13**; BH-FDR q<0.05 counts = **10/11/13**. Median r = **0.069/0.088/0.116**, bootstrap 95% CIs = **[0.025, 0.111] / [0.045, 0.129] / [0.055, 0.153]**.
- **Exploratory paired scaling check (not pre-registered):** mean within-animal r delta 3B-1B = **+0.029**, paired bootstrap 95% CI **[+0.001, +0.059]**, 12/18 animals increase. 8B-3B = **+0.012**, CI **[-0.029, +0.057]**, 11/18 increase.
- **Interpretation:** point estimates are monotonic and the multiple-testing control is reassuring, but overlapping median intervals and an unresolved 8B-3B paired delta mean this is suggestive—not a scaling law. One 70B point would test continuation/plateau/reversal but requires external full-precision multi-GPU-class compute; do not substitute quantized local weights because that changes the measurement.

### S3A — geometry scaling and resumable full-data collection (PRE-REGISTERED; no new results inspected yet)
- **Correction/context:** the original take-home already ran `prompting/geometry_metrics.py` at 1B. Its raw-cosine mean correlations were forward r=0.151 and reverse r=0.109; centered cosine reverse r=0.143. The novel question here is how those mechanism measurements change with model size.
- **Models:** `unsloth/Llama-3.2-1B-Instruct`, `unsloth/Llama-3.2-3B-Instruct`, `unsloth/Llama-3.1-8B-Instruct`; same Llama vocabulary and the exact existing 18-animal/1,110-number prompt protocol.
- **Primary per-model estimand:** for each animal, correlate the paper's raw unembedding cosine (mean of all token rows for multi-token animal names, exactly matching the old code) with reverse `log P(animal target token | love number)` across all number tokens; take the mean of 18 correlations. Report seed-0, 100,000-resample across-animal bootstrap 95% CI.
- **Primary scaling contrasts:** paired mean within-animal correlation change for 3B-1B and 8B-3B with paired bootstrap 95% CIs. Report all outcomes; do not infer a general law from three sizes.
- **Required control:** for each behavioral animal, compare its matched geometry correlation against the mean correlation obtained from the other 17 animals' geometry vectors. Report mean matched-minus-mismatched and paired bootstrap CI. Geometry is animal-specific only if this difference is positive; a positive matched correlation alone can reflect generic number structure.
- **Fixed secondary analyses (all reported):** forward animal->number prediction; centered cosine; raw and centered dot products; first-target-token geometry (aligned to the actual reverse target) versus original mean-subtoken geometry; raw-cosine BH-FDR count across animals; descriptive owl->087/eagle->747/sea-turtle->321 geometry ranks.
- **Raw artifact requirements:** atomic checkpoint/resume; model/tokenizer and prompt metadata; number token IDs and decoded strings; animal names and all token IDs; reverse Y and forward X log-probability matrices; baseline animal log probabilities; selected output-embedding rows in float32; float32 vocabulary mean; completion masks and runtime. These sufficient statistics must permit later analysis without reloading the model.
- **Validation sequence:** (1) intentionally stop a tiny 1B smoke after a few reverse rows; (2) resume and finish it; (3) full 1B run; (4) compare the new 1B geometry summaries to `metrics_1b.json` and behavioral correlations to `entanglement_1b.json`; only then run 3B and 8B.
- **S3A-smoke commands:** first `cd prompting && python collect_full_probe.py --model unsloth/Llama-3.2-1B-Instruct --animals owl eagle --max-numbers 12 --tag smoke_resume --checkpoint-every 2 --stop-after-number-rows 3`; then repeat without `--stop-after-number-rows`.
- **S3A-smoke result:** PASS. First process atomically saved 3/12 reverse rows and exited intentionally. Second process validated model/tag/animals/token IDs/prompts, reported `resuming: 3/12`, completed the remaining 9 reverse rows, baseline, 2 forward rows, selected fp32 unembedding rows, and vocabulary mean. All arrays passed shape/NaN/completion assertions. A third identical launch printed `already complete` and exited before model load. Smoke inferential output is ignored (only 2 animals/12 numbers). Total recorded compute across interrupted + resumed processes: 4.6 s excluding model load.
- **S3A-1B command:** `cd prompting && /Users/barathv/.venvs/subliminal-scaling/bin/python collect_full_probe.py --model unsloth/Llama-3.2-1B-Instruct --tag geometry_1b --checkpoint-every 100`
- **S3A-1B collection:** PASS. 1,110 reverse rows + baseline + 18 forward rows + selected fp32 unembedding rows/vocabulary mean in 73.39 s; raw artifact 5.2 MB. During validation, the first streaming/float64 vocabulary-mean implementation changed centered metrics, so it was rejected and replaced with the original analysis's full-fp32 `torch.mean`; geometry-only refresh took 6.20 s without rerunning prompts.
- **Regression gate:** PASS against fresh runs of the original scripts on the same MPS device. Maximum geometry-summary difference across 4 metrics x 2 directions = **2.08e-08**; max behavioral Pearson-r difference = **1.26e-08**; all top-number lists and paper-pair behavioral ranks identical. The checked-in July 2 artifacts were CPU-generated and show small device arithmetic drift (behavioral r max delta 0.015; geometry summary max delta 0.039), so S3A uses newly collected MPS artifacts consistently across all local sizes rather than mixing CPU and MPS measurements.
- **Pre-registered 1B result:** primary raw-cosine reverse mean r = **0.103**, across-animal bootstrap 95% CI **[0.054, 0.151]**; 12/18 positive correlations survive BH-FDR. Animal-specificity control: matched mean r **0.103** versus mean mismatched r **0.003**, delta **+0.100 [0.061, 0.135]**; matched beats mismatch in 16/18 animals. Interpretation: raw geometry predicts only ~1% of within-animal variance, but that weak signal is genuinely animal-specific. Descriptive raw-cosine geometry ranks: owl->087 **31/1110**, eagle->747 **244/1110**, sea turtle->321 **923/1110**. Secondary centered-cosine reverse mean r = **0.123 [0.067, 0.174]**; first-target-token raw-cosine reverse mean r = **0.099 [0.043, 0.150]**. Proceed to 3B without changing metrics.
- **S3A-3B command:** `cd prompting && /Users/barathv/.venvs/subliminal-scaling/bin/python collect_full_probe.py --model unsloth/Llama-3.2-3B-Instruct --tag geometry_3b --checkpoint-every 100`
- **S3A-3B collection/regression:** PASS in 162.42 s; 7.2 MB raw artifact. Against the existing same-device 3B behavioral artifact: max Pearson-r delta 1.34e-08, identical top-number lists and paper-pair ranks.
- **Pre-registered 3B result:** primary raw-cosine reverse mean r = **0.198 [0.148, 0.252]**, 17/18 BH-FDR. Paired 3B-1B delta = **+0.095 [0.042, 0.153]**. Animal-specificity: matched r **0.198** vs mismatched r **0.070**, delta **+0.128 [0.094, 0.159]**, matched wins 17/18. Raw cosine therefore explains roughly r^2=3.9% at 3B versus 1.1% at 1B; this is an explanatory-variance heuristic, not a causal fraction. Paper-pair geometry ranks: owl->087 **89**, eagle->747 **330**, sea turtle->321 **831** of 1110.
- **Fixed secondary 3B results:** raw-cosine forward mean r **0.239 [0.211, 0.267]** (18/18 BH-FDR); first-target-token reverse raw cosine **0.198 [0.149, 0.250]**; centered-cosine reverse falls to **0.096 [0.016, 0.173]**. This reverses the old 1B suggestion that centering helps, so do not promote centered cosine as a better general metric. Await 8B before any scaling claim.
- **S3A-8B command:** `cd prompting && /Users/barathv/.venvs/subliminal-scaling/bin/python collect_full_probe.py --model unsloth/Llama-3.1-8B-Instruct --tag geometry_8b --checkpoint-every 100`
- **S3A-8B collection:** PASS in 513.61 s including the initial ~15 GB model download; raw artifact 8.6 MB; peak process footprint reported ~21.1 GB. All 1,110 reverse rows, baseline, 18 forward rows, selected fp32 unembedding rows, and exact fp32 vocabulary mean complete with no NaNs.
- **Pre-registered primary ladder result:** mean raw-cosine->reverse r **1B 0.103 [0.054,0.151]; 3B 0.198 [0.148,0.252]; 8B 0.189 [0.147,0.231]**. Positive BH-FDR counts **12/17/17 of 18**. Paired 3B-1B **+0.095 [0.042,0.153]**, 13/18 animals rise; paired 8B-3B **-0.010 [-0.061,0.041]**, 8/18 rise. Interpretation: evidence for a 1B->3B jump followed by a 3B->8B plateau, not monotonic strengthening at every step.
- **Required animal-specificity control:** matched-minus-mean-mismatched r **1B +0.100 [0.061,0.135] (16/18 wins); 3B +0.128 [0.094,0.159] (17/18); 8B +0.155 [0.125,0.186] (18/18)**. Exploratory paired control changes: 3B-1B +0.028 [-0.010,0.063]; 8B-3B **+0.027 [0.003,0.052]**. Only the latter follow-up interval excludes zero; label exploratory.
- **Fixed secondary ladder results:** first-target-token raw-cos reverse remains positive **0.099/0.198/0.165**. Centered-cos reverse **0.123/0.096/-0.025**; paired raw-minus-centered at 8B **+0.213 [0.140,0.286]**, raw higher for 16/18 animals. Raw-cos forward **0.140/0.239/0.225**. Report all; centered-cosine improvement is refuted as a scale-general claim.
- **Descriptive paper-pair raw-cos geometry ranks (1B/3B/8B):** owl->087 **31/89/803**; eagle->747 **244/330/431**; sea turtle->321 **923/831/1031**, all out of 1110. Aggregate geometry-behavior alignment can strengthen even while famous individual pairs become geometrically ordinary.
- **Architecture caveat:** 1B/3B use Llama 3.2; 8B uses Llama 3.1 because those are the available headline sizes. Thus 3B->8B mixes size and release. A future Llama-3.1-70B point compares cleanly with Llama-3.1-8B on release/tokenizer, but two within-release points still cannot establish a general scaling law.
- **Same-device behavioral control command:** `python scaling/analyze_size_ladder.py --artifact 1B=prompting/results/full_probe_geometry_1b.json --artifact 3B=prompting/results/full_probe_geometry_3b.json --artifact 8B=prompting/results/full_probe_geometry_8b.json --output prompting/results/size_ladder_mps_summary.json`.
- **Same-device behavioral control result:** raw counts **10/12/13**, BH-FDR **9/11/13**, median r **0.068/0.088/0.117**. Primary story unchanged; paired adjacent intervals overlap/touch zero. Use this artifact rather than the mixed CPU/MPS summary for new work.
- **Reproducibility checksums:** raw NPZ SHA-256: 1B `190c322127f560ae3de5f7c2ec09d751e72c2c14a22593d647cc0d61ed8032e8`; 3B `5bae5218ba523755e931c263ab7604e5414fcf1ee826da984256aa1f14e15dae`; 8B `95b507b52d68cf7fc17ae3338bb9f0e90d3ca615c2ba6d169c803a0666a759da`. Geometry summary checksum after control follow-ups: `2d1b1ea2765a8b465f3d53e76c1906104e71197327e7b8759cce8ad11af1e995`.

### S3B — CUDA calibration + full-BF16 70B (COMPLETED)
- **Authorization/budget:** use the existing Vast.ai credit only. One on-demand instance; offer `dph_total <= $1.70/hr`; 220 GB disk; hard local runtime 85 minutes; remote kill watchdog 90 minutes; abort near $0.90 remaining credit. Pull partial/final artifacts and logs, validate locally, and always `vastai destroy instance -y` in cleanup. Do not retry on a second paid instance automatically.
- **Offer requirements:** verified/rentable, reliability >=0.99, direct SSH, CUDA >=12.4, total GPU RAM >=180 GB, per-GPU RAM >=40 GB, >=2 GPUs, system RAM >=64 GB, disk availability >=240 GB, download >=500 Mbps, low ingress charge. Prefer non-CN geography to reduce public Hugging Face access risk.
- **Image/deps:** `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (Python 3.11); install `transformers==5.14.1 accelerate==1.14.0 numpy==2.4.2 scipy==1.17.0`. Transformers 5.14 declares torch>=2.4, so image torch 2.6 is supported. NumPy/SciPy are the newest exact pins with verified CPython-3.11 Linux wheels; the first pins were local-Python-3.12-only. Collector explicitly uses `--device-map auto --dtype bfloat16`.
- **CUDA calibration:** full `unsloth/Llama-3.1-8B-Instruct` probe first, not a subset. Purpose: measure device/environment drift and verify remote collection before paying the 70B download/load cost. Compare paired MPS 8B versus CUDA 8B but do not tune 70B metrics from the calibration.
- **70B model:** verified public mirror `unsloth/Meta-Llama-3.1-70B-Instruct`, 71B parameters, BF16 safetensors; no quantization. Smoke = owl/eagle, first 12 discovered numbers; inferential output ignored. Full = unchanged 18 animals x all 1,110 Llama number tokens.
- **Primary 70B estimand/contrast:** per-animal Pearson correlation across number tokens between mean-subtoken raw unembedding cosine and reverse log probability; mean across 18; paired CUDA70B-CUDA8B delta with seed-0 100,000-resample bootstrap 95% CI. Positive-only BH-FDR count reported.
- **Required control:** per model matched animal geometry minus average of 17 mismatched animal geometries; report bootstrap CI and paired 70B-8B change. Fixed secondaries: behavioral entanglement, forward raw cosine, centered cosine, first-target-token raw cosine, raw-vs-centered paired difference, named-pair geometry ranks. Report all outcomes.
- **Decision language:** 70B-8B primary CI >0 = evidence geometry resumes strengthening; spans 0 = plateau/unresolved; <0 = reversal. Even a resolved two-point within-release contrast is not a universal scaling law.
- **Paid attempt 1 (setup-only failure):** instance 45256399 used offer 20805216 (4x RTX A6000). SSH passed, then pip rejected `numpy==2.5.1` because the image has Python 3.11. The runner destroyed the instance immediately and an independent active-instance query returned `[]`. No model/probe result exists. Remote pins corrected to `numpy==2.4.2 scipy==1.17.0`, with CPython-3.11 Linux wheels verified via `pip download --no-deps`. Vast's charge was delayed; use the final combined accounting below rather than intermediate snapshots.
- **Paid attempt 2 (user-approved success):** same offer, instance 45256931. Created 2026-07-18T19:41:26Z; destroyed 20:12:23Z; wall 30m57s. Full 8B took 202.2s including load, full 70B took 334.7s including cached reload; initial 70B download/load dominated the rest. Full BF16 sharded across four GPUs at ~33-35 GB/GPU; no quantization or CPU offload. Artifacts pulled before destruction. Active-instance query after cleanup: zero.
- **Primary result:** 8B-CUDA mean raw-cos/reverse r **0.18773 [0.14605,0.22968]**, 17/18 BH-FDR; 70B **0.10758 [0.05214,0.15384]**, 15/18. Paired **70B-8B = -0.08015 [-0.12706,-0.03471]**, 4/18 animals increase. Decision = pre-registered reversal/weakening. MPS8B->CUDA8B device contrast **-0.00083 [-0.00162,-0.00003]**, demonstrating the size result is not meaningful device drift.
- **Required specificity control:** 8B-CUDA matched-minus-mismatch **+0.15499 [0.12561,0.18565]**; 70B **+0.04581 [-0.01482,+0.09281]**; exploratory paired change **-0.10919 [-0.16400,-0.05989]**, only 1/18 animals increases.
- **Behavioral result:** 8B median r **0.117 [0.057,0.155]**, 13/18 raw and BH; 70B **0.088 [0.054,0.123]**, 12/18. Paired mean change **-0.024 [-0.056,+0.009]**, 6/18 increase: unresolved rather than evidence for a behavioral decline.
- **Fixed secondary results:** raw forward mean r 8B/70B **0.22403/0.12890**; centered-cos reverse **-0.02322/-0.07639**; first-target raw-cos reverse **0.16499/0.12745**; at 70B raw-minus-centered **+0.18398 [0.11715,0.24333]** (16/18 raw higher). Paper-pair ranks at 70B: owl->087 **602**, eagle->747 **404**, sea turtle->321 **335** of 1,110.
- **Artifacts/checksums:** `full_probe_geometry_8b_cuda.npz` `40c45bee43fd43214363df44c28f50a2179276c39a4bac0bfb300a72ccf240ad`; `full_probe_geometry_70b_cuda.npz` `0261b672b43b04c7c5f84b003e0399819c98d10535a2b249ebe9188b70e9daa8`; geometry summary `87e3d1f796b2ee180cc8f182bc3491aaf3dea96e29c51f0d40e389ec93288c46`; behavior summary `54e6059f07a588138142eba431d9eb8753693d315aba32570437005f24b20556`. Exact seed-0/100k geometry rerun reproduced the summary byte-for-byte when given the same absolute paths.
- **Cost:** final observed credit **$3.70523041311 -> $2.72221427921**, combined attempts **$0.98301613390**, remaining **$2.72**. No billing method attached. Attempt-2 launch snapshot was $3.54528279701, so its observed cost was ~$0.82307; attempt-1 delayed settled cost was ~$0.15995.
- **Local post-run fix:** `analyze_size_ladder.py` had hard-coded 1B->3B->8B labels and failed after the paid instance was already safely destroyed. Changed it to compare adjacent labels in supplied artifact order; rerun produced the behavior summary above. This reporting-only bug did not affect raw data or geometry analysis.

### S4 — tokenization and layerwise localization (COMPLETED)

- Full protocol and decision rules: `scaling/preregistration_s4.md`.
- **S4A:** exact 1/2/3-width decimal-string universe; full autoregressive sequence log probability versus reverse animal log probability; width-3 primary; first-token and animal-mismatch controls; local only.
- **S4B:** tuned logit-lens scores at every layer, assistant position plus system-number mentions; final-logit regression required; AUC and half-emergence depth primary; 1B/8B local gate before any second 70B rental.
- **Novelty audit:** searched the current primary literature after S3. Existing work covers token entanglement, divergence tokens, steering-vector transfer, output-head compatibility, and layer-localized *training changes*. No located work performs the pre-registered multi-token repair or a scale-matched 8B/70B layerwise trace of the pre-existing prompting channel. Treat this as a search finding, not an absolute first-ever claim.
- **S4A smoke 1 (ignored; implementation discovery):** 3 width-3 strings x owl/eagle on Llama-3.2-1B; trie-vs-naive max error 0.0. It revealed that `" "+number` introduces a standalone Llama space token, unlike the original direct number-token target. Protocol amended before any full result to target `number` with no added separator; smoke 1 is discarded and never analyzed inferentially.
- **S4A smoke 2 (ignored; numerical guard fired):** corrected target strings were atomic as expected, but the new `log_softmax` score differed from v2's `log(softmax+1e-12)` by 0.0011 in the tail. The assertion stopped before completion. Collector changed to the exact v2 clipped probability definition per autoregressive target token; smoke 3 must pass both references.
- **S4A smoke 3 PASS (ignored inferentially):** same 3 width-3 strings x owl/eagle on Llama-3.2-1B. All targets are one token. Trie vs slow naive teacher forcing max absolute delta **9.36e-7**; sequence scorer vs original v2 atomic selected-logp max delta **0.0**. Implementation gate passed; full Qwen3-0.6B may run unchanged.
- **S4A full commands:** `cd prompting && python collect_sequence_probe.py --model Qwen/Qwen3-0.6B --tag qwen3_06b_full --checkpoint-every 25`, then the same with `Qwen/Qwen3-1.7B` / `qwen3_17b_full`; analyze with `python scaling/analyze_sequence_probe.py --artifact Qwen3-0.6B=... --artifact Qwen3-1.7B=... --output prompting/results/sequence_probe_qwen_scaling_summary.json`.
- **S4A collection:** PASS for both models, all 1,110 fixed strings × 18 animals, no missing cells. Qwen3-1.7B was downloaded from the official public model repository; no paid GPU was used. Exact trie-vs-naive/atomic numerical gates passed before either full result was interpreted.
- **S4A primary:** width-3 full sequence mean r **0.6B -0.04978 [-0.12444,+0.02727]** (4/18 positive BH-FDR) and **1.7B -0.04783 [-0.08789,-0.00758]** (1/18 positive BH-FDR). Paired 1.7B-0.6B **+0.00195 [-0.07794,+0.08715]**. Decision: no positive sequence-level rescue; 1.7B is slightly resolved negative.
- **S4A first-token control:** first-token width-3 r **+0.02375 [-0.06098,+0.10855]** at 0.6B and **+0.02142 [-0.03971,+0.10161]** at 1.7B. Full-minus-first **-0.07353 [-0.10219,-0.04159]** (1/18 full higher) and **-0.06925 [-0.15119,+0.00728]** (5/18). Complete sequence composition does not add the atomic-style signal.
- **S4A specificity/width controls:** matched-minus-mismatched **+0.05604 [+0.01095,+0.11479]** at 0.6B but **+0.00278 [-0.02477,+0.03046]** at 1.7B. Full-sequence width-1/2/3 mean r: 0.6B **+0.39794/-0.08510/-0.04978** (width 1 has only ten strings and zero significant animals); 1.7B **-0.121/+0.1402/-0.04783**. Width dependence is non-monotonic and not a universal Qwen family law.
- **S4A length-confound result:** mixed-width sum-logp r **-0.09793/-0.16233**; naive mean-logp-per-target-token **+0.09699 [+0.05446,+0.14290] / +0.23265 [+0.1950,+0.2679]**; within-width standardized **-0.04892/-0.03155**. Because the strong positive appears only after naive length division and disappears after within-width control, report it as a measurement confound, not entanglement.
- **S4A artifacts/checksums:** raw 0.6B NPZ `82c2a9d3da1337a3f78bb33b6fe5fba6b08e0bc6654873353df572c04d3b291a`; raw 1.7B NPZ `3c68ca3ba2fbcad64cd7e638830eddf2331507401144783aebf5682eab3420d7`; summary `a5284a94d2ce72257810a1a384979c0eaeaf234bdb1e0fdc80dc1e6def46eebb`.
- **S4B implementation guards:** smoke 1 found date-digit collisions when locating target `0`; locator was restricted to the exact rendered system-message span. Smoke 2 caught double-normalization of Transformers' already-normalized final hidden state and a backend selected-row matmul discrepancy. Protocol amendments were written before full analysis. Smoke 3 passed exact final logits. Constant assistant embedding readouts are assigned r=0 and counted. The normal full-head logits define the exact final assistant endpoint; the selected-row BF16 kernel delta is retained as metadata.
- **S4B local gate/full:** ignored 100-number Llama-3.1-8B slice produced a coherent late-rising assistant curve and exact final logits. Unchanged full local command: `python prompting/collect_layerwise_probe.py --model unsloth/Llama-3.1-8B-Instruct --base-artifact prompting/results/full_probe_geometry_8b.npz --tag layerwise_full_8b_mps --checkpoint-every 50`. Full 1,110-number MPS result: assistant AUC **0.25914 [0.23677,0.28184]**, half-final depth **0.71875**; mean-number-position AUC **0.05105 [0.00897,0.09758]**. Raw NPZ checksum `50d35264ffa35a9617d8150279d0d94faf7240b950d2385eef84196fdbf950e5`.
- **S4B cloud protocol:** runner `scaling/run_vast_layerwise.py`, dry-run default. Same offer 20805216: 4×RTX A6000, $1.60777778/hr, full BF16, 220 GB disk, 85-minute wall ceiling, $0.10 credit floor, 45-second checkpoint pulls, remote watchdog, destroy-always. Uploaded the prior CUDA8B/70B behavior NPZs; ran full CUDA8B, ignored three-number 70B smoke, then full CUDA70B.
- **S4B cloud attempt 1:** instance 45261127 reached the first import and stopped because `entanglement.py` was absent from the upload bundle. No model download or scientific result. Instance destroyed, active list empty; settled cost **$0.05078624**. Added the dependency and validated the entire four-file import closure in an isolated temporary directory before retry.
- **S4B cloud success:** instance 45261293 created 2026-07-18T21:00:19Z, destroyed 21:34:24Z. Full 8B, ignored 70B smoke, and full 70B all passed; 70B loaded at ~33-35 GB/GPU, no quantization/offload. Final credit **$1.72882825921**, no billing, active instance query `[]`. Corrected-run observed cost **$0.94259978**.
- **S4B primary:** assistant AUC **8B-CUDA 0.25913 [0.23630,0.28210]**, **70B-CUDA 0.26876 [0.23510,0.30034]**. Paired 70B-8B **+0.00963 [-0.02799,+0.04528]**, 13/18 increase. Half-final relative depth **0.71875 -> 0.7375**. Pre-registered decision = similar/unresolved, not evidence of lower 70B AUC.
- **S4B animal-specific control:** assistant contrast AUC **0.27762 [0.25555,0.29959] -> 0.29770 [0.26664,0.32710]**; paired **+0.02009 [-0.01453,+0.05469]**. Exact final contrast r is 1 by algebra because subtracting other animal log-probabilities removes the common softmax denominator; the meaningful statistic is the pre-final trajectory/AUC.
- **S4B position controls:** mean system-number AUC **0.05067 [0.00841,0.09739] -> 0.03562 [-0.00474,+0.07227]**, paired **-0.01505 [-0.07391,+0.04216]**. Last-number AUC **0.03995 [-0.00156,+0.08589] -> 0.03073 [-0.00745,+0.06327]**, paired **-0.00922 [-0.07286,+0.04988]**. All spans include zero for the scale changes; do not claim a decline.
- **S4B device/robustness:** MPS8B-vs-CUDA8B assistant AUC **-0.000011 [-0.001127,+0.001114]**; maximum pointwise mean-curve difference 0.00445. Leave-one-animal-out assistant AUC ranges: CUDA8B **[0.25240,0.26472]**, 70B **[0.26208,0.27602]**. Mean curves have descriptive interpolated Pearson r=0.993, but this was not pre-registered and is not an inferential claim.
- **S4B interpretation:** combined with S3B's resolved static-geometry drop -0.080 [-0.127,-0.035], the stable late contextual AUC is a dissociation: scale weakens the static output-token-geometry explanation without erasing or decisively delaying the linearly readable answer computation. This supports localization toward output-head/token geometry, not proof of an identical causal circuit.
- **S4B artifacts/checksums:** CUDA8B NPZ `44674a5a471edd2aa2f6e06b141962eb2d03449bcc0adb24d2e3255516f658ad`; 70B NPZ `825a8d133f4910a11419b8aa2cb37f3f3cd5bc6249146de678258d8dc906758e`; ignored smoke `1b14c43726507b44cedc6f60f0c7d914e1e58d6f877885df3c10c92c32c3eb54`; analysis summary `c7ccfa05a2abfdf4374e9930dcd82cec9ee1e12d6b3453b232a25754307fff98`.

### S5 — causal assistant-state transmission (COMPLETED)

- **Protocol:** `Paper/Supplement/preregistration-s5.md`. Fixed 256 unique
  width-3 atomic Llama numbers from seed-0 outcome-blind sampling; 128 unordered
  pairs, both directions; exact shared design across 8B/70B; block-input patch at
  requested relative depths `.25/.50/.75/.90/.97`; 18 animals; target animal
  logit minus mean other-animal logit.
- **Implementation:** collector `prompting/collect_causal_patch.py`; analyzer
  `scaling/analyze_causal_patch.py`; fail-safe cloud runner
  `scaling/run_vast_causal_patch.py`; figure script
  `scaling/plot_causal_patch.py`.
- **1B smoke (ignored):** first attempt correctly failed because duplicate MPS
  forwards changed under grad-enabled kernels. Decorating selected-logit and
  patch paths with `torch.no_grad()` made clean duplicate and identity errors
  exactly zero. No inferential result uses the smoke.
- **Estimator validation:** the original preregistered regression used
  `delta=patched-recipient` and `Delta=donor-recipient`. The shared negative
  recipient term mechanically raised the permutation baseline. No paid run was
  launched. Frozen S5B amendment before any CUDA/70B S5 output:
  `patched = intercept + beta_donor*clean_donor + gamma_recipient*clean_recipient`.
  Donor coefficient is corrected primary; original slope is mandatory invalid
  diagnostic.
- **Local MPS 8B validation:** corrected donor beta
  `.00070/.03759/.59190/.77928/.97421`; corrected causal AUC
  `.25340 [.24094,.26555]`; permutation beta near zero; every corrected gate
  condition passed. This authorized one paid comparison.
- **Cloud safety:** live launch credit `$1.60777967921`, no billing method, zero
  active instances. Selected offer 20805216: 4×RTX A6000 at
  `$1.60777778/hour`; 2,400-second hard cap; projected maximum charge
  `$1.07185185`; `$0.15` credit floor; 45-second pulls; remote watchdog; destroy
  in `finally`; never rent a second instance automatically.
- **Cloud execution:** instance `45267205`, created
  `2026-07-18T22:51:30Z`, destroyed `2026-07-18T23:21:33Z`. Exact CUDA8B ran
  first and validated before 70B. Full BF16 70B sharded at roughly 33–35 GB/GPU.
  Both artifacts pulled and validated; destroy return code 0; post-run active
  instances `[]`. Settled credit `$1.60777967921 -> $0.79508485761`; observed S5
  cost **$0.81269482160**; no billing method.
- **CUDA 8B:** actual depths `.25/.50/.75/.90625/.96875`; donor beta
  **`.00116/.03845/.59235/.77990/.97384`**; recipient gamma
  `.99807/.94341/.43501/.24348/.03109`; causal AUC
  **`.25389 [.24147,.26612]`**.
- **CUDA 70B:** actual depths `.25/.50/.75/.90/.975`; donor beta
  **`.00729/.77280/.93801/.94821/.98436`**; recipient gamma
  `.98883/.18879/.07164/.06098/.01970`; causal AUC
  **`.53970 [.52691,.55158]`**.
- **Primary paired contrast:** 70B-minus-8B corrected causal AUC
  **`+.28581 [+.27162,+.29991]`**, all **18/18** animals increase. Frozen
  decision: **stronger/earlier assistant-state causal transmission at 70B**.
- **Controls:** permuted donor beta ranges 8B `[-.0131,+.0006]`, 70B
  `[-.0024,+.0319]`; wrong-animal beta 8B `[-.0009,+.0333]`, 70B
  `[-.0003,+.0062]`; both direction halves positive at gate depths; clean
  duplicate and identity max errors exactly `0`; max standardized condition
  number `1.177/1.242`; zero degenerate bootstrap cells.
- **Artifacts/checksums:** 8B NPZ `a9a8acfb11ca4fdc0c396a2e8decf009cb0509ab122cf7b3cb8a3cc7ceaa19f4`;
  70B NPZ `96acfd09720b6d27f8e1dad53e965c66878b614909f55500535bfabcc1ca72c0`;
  paired summary `af7c4a64b0e6fcd56311bf58118b6942c5ddd904e3b6387c7e6103d9475f829d`.
- **Interpretation:** static output geometry weakens from 8B to 70B, the
  observational readout AUC is similar, but natural donor-state causal control
  arrives dramatically earlier. Full-state patching establishes sufficiency,
  not necessity or a unique causal direction/circuit.

### S6 — AAAI-27 submission package (PREPARED, NOT SUBMITTED)

- **Target:** AAAI-27 Main Technical Track. Official deadlines checked online on
  2026-07-18: abstract 2026-07-21, paper 2026-07-28, supplement/code
  2026-07-31, all 23:59 UTC-12.
- **Official format:** AAAI 2027 anonymous two-column style, US Letter, seven
  technical-content pages plus references. Official unmodified
  `aaai2027.sty`/`aaai2027.bst` copied from the 2027 author kit.
- **Main artifact:** `Paper/Submission/AAAI27/output/pdf/aaai27-main.pdf`, six
  pages total with references beginning on page six. Single source `main.tex`;
  16 primary references; three figures; a claim-to-test table, matched-results
  table, all-depth coefficient table, and causal-control table. Abstract has no
  citations. No author or affiliation data in the review PDF.
- **Support artifacts:** 3-page technical supplement, 2-page required checklist,
  ready-to-paste abstract/metadata, prose audit, and
  `Paper/Learning/how-conference-submission-works.md`.
- **Code/data archive:**
  `Paper/Submission/AAAI27/output/aaai27-code-data.zip`, approximately 50 MB.
  Contains collector and analyzer code, 8B/70B/Qwen headline raw arrays,
  MPS/CUDA controls, deterministic summaries, pinned dependencies, commands,
  generated figures, and `MANIFEST.sha256`. All 36 manifest entries validate.
  Rerunning every analyzer and all figure commands from the archive layout
  reproduces the reported headline values and paper-ready plots. The geometry
  command's 70B label and archive-relative figure paths were repaired during
  the final clean-room check.
- **Anonymity/safety audit:** no author name, `/Users` path, Dropbox path, API
  key, cloud credential, personal repository URL, or mutable web supplement in
  the ZIP. PDF metadata contains no author/title identity. Fonts are embedded;
  PDF 1.5; US Letter; no encryption. Rendered every page and inspected figures,
  tables, margins, and column flow.
- **Prose policy:** edited for direct scientific writing, bounded claims, and
  explicit uncertainty. Avoided gratuitous em dashes as a style preference, not
  as an invented conference or detector rule. AAAI-27 explicitly allows
  judicious generative-AI assistance while holding authors responsible for all
  content.
- **External status:** no OpenReview submission created. Required human gate:
  complete author list and profiles, affiliation/email, conflicts, concurrency
  confirmation, reviewer/attendance obligations, license choice, and explicit
  approval of title and frozen abstract.

### S7 — adversarial accepted-paper benchmark (COMPLETED)

- Compared the draft against recent official AAAI award papers and nearby
  interpretability papers. Repeated accepted-paper pattern: one explicit gap,
  early result figure, operational formulas, controls beside claims, bounded
  limitations, and runnable artifacts. Benchmark and stopping rule are in
  `Paper/Research/aaai-accepted-benchmark.md`.
- Three skeptical reviews independently found no fatal result-invalidating
  flaw. Pre-fix target was 6/10 weak accept; after fixes, target is 7/10 accept,
  confidence 4/5. This is a quality benchmark, not an acceptance promise.
- Closest novelty constraint: Ma and Rui (2026), *Where's the Plan?*, already
  performs cross-scale probing and activation patching, including Llama 70B.
  Defensible novelty is the same-channel subliminal measurement bundle and the
  matched 8B/70B dissociation, not broad probe/patch scaling.
- Zero-cost robustness: raw-logit causal AUC 8B `.26676`, 70B `.54114`, paired
  mean `+.27437`, 18/18 positive; leave-one-pair-cluster-out paired range
  `[+.28502,+.28756]`; all 17 wrong-concept AUC shifts remain near zero; 14/18
  animals jointly show weaker static geometry and stronger causal AUC; exactly
  eight blocks remaining gives donor beta `.59235` at 8B versus `.94821` at
  70B.
- Final reproducibility audit: every summary and figure command reran from
  `Paper/Submission/AAAI27/code-data`; manifest validates after fresh ZIP
  extraction; ZIP contains no bytecode cache. Immutable model revisions were
  not recorded during historical collection, so the README states this
  limitation. Code license remains a human choice.

### S8 — outcome-blinded external transfer validation (COMPLETED; MAIN GATE FAILED)

- **Why:** Blank et al. (2026) released exact Llama-3.1-8B student-transfer
  results for a frozen 16-animal zoo. This made a redundant smaller student
  training run scientifically inferior to a direct external check.
- **Outcome blinding:** fixed animals, predictors, controls, rank statistics,
  100,000 seed-0 animal bootstraps/permutations, BH family, and integration gate
  in `Paper/Supplement/preregistration-s8-external-transfer.md`; committed and
  pushed as `a107d0b` before opening the per-animal CSV.
- **External revisions:** code
  `89ab3616f6ed0e11a69481c1acd19d37c44e3706`; Hugging Face dataset
  `4fda20d0413040b2de61448c89182716485d9839`; upstream outcome SHA-256
  `5d19059f211bb2da9d4da54ec14fa41adec6a3357835856ee92b8d62ca5d0e60`;
  preregistered steering JSON SHA-256
  `442730d298e7c87a26f7009dcff0da5b4bd58b12159be19edf8e16d6fc2a6946`.
- **Local collection:** full MPS Llama-3.1-8B; 1,110 atomic decimal tokens; all
  16 external animals; 33-state output-head trace; 128 seed-0 unordered number
  pairs in both directions; S5 depths `.25/.50/.75/.90/.97`. No paid GPU.
- **Instrument checks:** geometry mean r `.173 [.103,.237]`, 13/16 BH;
  readout AUC `.274 [.251,.298]`, endpoint difference zero; corrected causal AUC
  `.255 [.242,.268]`. Causal mean donor beta
  `.001/.046/.585/.790/.970`; identity/duplicate error zero; permutation and
  wrong-animal controls near zero; max condition number `1.140`; zero degenerate
  cells.
- **External prediction:** geometry rho `.562 [.099,.847]`, permutation
  `p=.02595`, BH `q=.07785`; readout `.316 [-.287,.786]`, `q=.350`; causal
  `.111 [-.444,.624]`, `q=.687`; released steering benchmark
  `.768 [.367,.935]`, `p=.00097`. Steering-minus-causal paired rho difference
  `.657 [.069,1.207]`.
- **Missing fixed sensitivity:** upstream CSV lacks `base_prior_count`, despite
  released aggregate code supporting the field. Reported as not computable; no
  replacement covariate.
- **Decision:** no in-house predictor met positive bootstrap CI + BH-FDR <.05.
  The main paper retains its original headline claims but now discloses the
  negative gate result in one sentence; the complete diagnostic enters the
  technical supplement and project record. No metric, animal, or outcome was
  changed after unblinding.
- **Final red-team audit:** three independent read-only reviews found no fatal
  scientific or solo-author policy issue. The causal analyzer now emits the
  raw-logit, all-17-wrong-concept, 128 leave-one-pair, and preregistered
  depth-rise sensitivities stated in the paper. Every packaged analyzer reruns
  exactly. The anonymous ZIP is 49.49 MB with 46 checksum-manifest entries,
  below the 50 MB upload boundary.

### S9 — AAAI page-budget and nearest-prior-work surgery (COMPLETED)

- **Purpose:** use the remaining main-paper space only where it reduces a real
  reviewer uncertainty: novelty, measurement provenance, causal scope, and the
  negative external-validation boundary. No new experiment and no changed
  result.
- **Literature additions:** Geh et al. (2024) on tokenization-space signal,
  Lesci et al. (2025) on causal tokenization bias, Sanz-Guerrero and von der
  Wense (2025) on label-length bias, and Makelov et al. (2024) on interpretive
  limits of activation patching. All four were checked against primary
  ACL/OpenReview records.
- **Manuscript changes:** added a full-width six-row nearest-prior-work table;
  stated explicitly that autoregressive chain-rule sequence scoring is
  established; promoted the S8 external statistics to the main discussion;
  bounded full-state patching as intervention-specific sufficiency rather than
  necessity or unique-circuit identification.
- **Final main package:** 7 pages total, references begin on page 6, 3 figures,
  5 tables, 22 emitted bibliography entries. Official anonymous
  `aaai2027.sty` hash remains
  `391bce82815bf698b8e382dd3ae7e30c75d7ab46df140cb295b1266016bc8623`,
  identical to the downloaded author kit.
- **Visual decision:** all seven pages rendered at 120 dpi and inspected. The
  comparison table uses ragged-right columns; no clipping, overlap, illegible
  figure, or blank page. The claim-to-test table and both main result figures
  share page 5 densely but cleanly.
- **Stopping rule:** do not add decorative equations, colored maps, or extra
  citations solely to imitate award papers. The next material scientific step
  would require new evidence, not manuscript padding.

### S10 — live AAAI/OpenReview administrative audit (COMPLETED)

- **Primary sources:** live AAAI-27 submission, main-track, modification,
  supplementary, topics, and attendance pages; active OpenReview invitation
  `AAAI.org/2027/Conference/-/Submission`; current OpenReview profile and
  activation documentation. Checked 2026-07-18.
- **Three-gate structure:** one OpenReview record, not three submissions.
  Abstract registration is July 21; main PDF plus separate checklist July 28;
  technical supplement plus code/data July 31; all 23:59 UTC-12.
- **Freeze rules:** topics and reciprocal reviewer status freeze July 21;
  authors/order/paper can change through July 28 while title/TL;DR/abstract may
  not change substantively; supplements alone can change through July 31;
  nothing can change afterward before notification.
- **Live form discoveries:** mandatory TL;DR, author profiles,
  institution-country, primary topic, reciprocal-reviewer status, and policy
  attestations. Main/checklist/supplement/code caps are 10/5/10/50 MB. External
  repository links are forbidden for the code/data supplement.
- **Solo reviewer status:** a nominated author needs two first-author or five
  coauthored qualifying archival papers. A truly unqualified solo author should
  select the form's `no author qualifies` declaration; a nonauthor professor
  must not be added or nominated.
- **Artifact:**
  `Paper/Submission/AAAI27/START-HERE-SUBMISSION-ROADMAP.md` gives exact
  baby-food steps, ready-to-paste title/TL;DR/topics, New York conversions,
  internal early targets, upload paths, confirmation checks, and later dates.
- **Remaining blocker:** the author's active OpenReview profile and truthful
  identity/policy fields. New public-email profiles can require up to two weeks
  of moderation; account activation is urgent. All artifacts are already ready,
  so the recommended risk-minimizing action is to upload all four in the same
  session as registration, then use later deadlines only as verification gates.
  No external submission was created or modified.

### S11 — clean reference-page transition (COMPLETED, THEN REVERSED BY S12)

- **Question:** can the stranded `References` heading and first citation at the
  bottom of page 6 move cleanly to the next page without weakening or bloating
  the submission?
- **Change:** inserted a page flush immediately before the bibliography. No
  scientific prose, citation, bibliography metadata, or official style setting
  changed.
- **Result:** main remains 7 pages total; six pages of technical content are
  followed by all 22 references on page 7. No eighth page, font reduction,
  clipped object, or cramped bibliography.
- **Visual QA:** rendered pages 6-7 at 144 dpi. Page 6 ends after the conclusion
  with a modest intentional lower margin; page 7 starts with the reference
  heading and retains comfortable whitespace in the second column.
- **Historical decision:** initially kept for visual polish. S12's fresh author-kit
  audit found that the mechanism (`\clearpage`) is explicitly forbidden, so
  the change was removed. This is retained here as an audit trail, not current
  submission guidance.

### S12 — Gargantua final science, policy, style, and artifact audit (COMPLETED)

- **Independent red teams:** separate read-only reviews covered causal and
  statistical validity, paper story and AI-flavored prose risk, and live AAAI
  format/anonymity/submission policy. Fatal findings: zero. No new experiment is
  required before this submission.
- **Official-policy repair:** the current AAAI-27 author kit explicitly forbids
  `\clearpage`, `\setlength`, and manual page-break/layout manipulation. Removed
  the S11 break, custom float packing, and table-spacing overrides; restored the
  template-required `caption` package. References now flow naturally after the
  final floats.
- **AI policy:** AAAI's publication policy permits AI assistance only when its
  role is documented and keeps the human author responsible. Added an accurate
  anonymous `Use of AI Systems` paragraph to the main manuscript. No detector
  evasion or fake human-authorship claim was attempted.
- **Claim surgery:** RQ2 now asks whether observational and causal traces share
  the same cross-scale change rather than claiming a prediction analysis. The
  remaining-block result is labeled one matched sensitivity. Undefined
  `becomes sufficient earlier` language now says `gains strong donor control
  earlier`. The failed S8 gate is explicitly a transparency-only negative check
  that did not alter the headline claims.
- **Fresh numerical rerun:** behavior/geometry, local size, exact Qwen sequence,
  8B/70B layerwise, 8B/70B causal, external-zoo causal, and external-transfer
  analyzers reproduced the packaged summaries. The layerwise package omits only
  the disclosed large MPS branch; common CUDA branches match exactly.
- **Bibliography:** 22 cited keys, 22 entries, zero missing, zero uncited. All 22
  URLs resolved and metadata matched primary records.
- **Final PDFs:** main 7 pages / 975,942 bytes; checklist 2 / 95,711;
  supplement 3 / 179,843. US Letter, embedded Type 1 fonts, no overfull boxes,
  no undefined citations, and every page visually inspected. Technical material
  stays within page 7; deferred Tables 4 and 5 precede References naturally.
- **Final ZIP:** 49,488,318 bytes, leaving 511,682 bytes below the decimal 50 MB
  cap. Fresh extraction passes ZIP integrity and all 46 SHA-256 manifest checks;
  no author identity, email, local path, personal repository, or credential was
  found.
- **Audit artifact:**
  `Paper/Submission/AAAI27/GARGANTUA-FINAL-AUDIT.md` records every final change,
  gate, digest, and remaining human responsibility. Decision: freeze the science
  and upload through the three OpenReview gates.

### S13 — Gargantua rebuttal shield and second adversarial wave (COMPLETED)

- **Purpose:** simulate the strongest plausible novelty, correctness,
  significance, evidence, clarity, and reproducibility objections before the
  files freeze; accept only attacks supported by the manuscript, saved arrays,
  or official AAAI rules.
- **Review process:** three independent reviews covered scientific/statistical,
  narrative/novelty, and submission/rebuttal failure modes. The main agent
  checked each finding against source, compiled PDFs, data, and rules. False
  duplicate-text claims and the proposed `unchanged stimulus set` wording were
  rejected.
- **Valid scientific repairs:** the closest-work table now distinguishes prior
  intervention, size-comparison, and positional-bias results; the external null
  is a direct limitation; the manuscript distinguishes 1,110-number
  observational analyses from the frozen 256-number causal subset; and the
  donor-state caveat now acknowledges a potentially off-manifold hybrid
  computation.
- **Matched-support sensitivity:** new deterministic script
  `Paper/Submission/AAAI27/code-data/code/scaling/analyze_matched_subset_sensitivities.py`
  recomputes observational metrics on the exact 256 causal numbers. Geometry
  changes `0.16752 -> 0.10291`, delta `-0.06461
  [-0.12434,-0.00334]` (11/18 decreases); specificity delta `-0.08529
  [-0.14318,-0.03029]` (14/18 decreases); readout `0.24765 -> 0.28120`, delta
  `+0.03354 [-0.00524,+0.07145]` (unresolved). Output:
  `code-data/data/summaries/matched_subset_sensitivities.json`.
- **Outcome-scale sensitivity:** raw target logits preserve the causal result:
  crossed-bootstrap delta `+0.27437 [+0.25934,+0.28936]`, 18/18 animal-level
  increases, and all 20,000 bootstrap draws positive.
- **Estimator provenance:** `timestamped amendment` was replaced with the
  accurate `recorded amendment`. The sanitized S5 protocol states that the
  chronology is author-reported and is not an independent third-party
  preregistration.
- **Response artifact:** `Paper/Rebuttal/` contains the live-rule summary, mock
  reviews, reviewer-attack matrix, evidence index, response bank, October
  author-feedback checklist, and remaining human decisions. It is private
  preparation, not a July upload.
- **Final files:** main 7 pages / 976,571 bytes; supplement 3 / 179,437;
  checklist 2 / 95,711. Code/data ZIP 49,495,215 bytes, leaving 504,785 bytes
  under the decimal 50 MB cap; fresh extraction passes all 49 manifest checks.
  Final SHA-256: main
  `232ea5dfc552181c9e7945fe221bf1eb6486ef94939fa75401edcdc7e3e541c6`,
  supplement
  `059366d0d9cdcbdd2c8b94739f0344bd630153c0ecbec0d0362679593bcd30b4`,
  checklist
  `03fd7c32831eb2a0c23af9ccd63d9138ba0aeff6760c8b1c89a41e971eeacf52`,
  ZIP `ed069bd9c44f947c6dc618049b48f2f8097e42ae63fc93a6695bbbd8f104c885`.
- **Decision:** GREEN. No new GPU experiment is justified for this deadline.
  Freeze the science, complete the human responsibility read, and submit the
  exact files through the three OpenReview gates.

### S14 — AI-assistance wording and venue mini-Gargantua (COMPLETED)

- **Question:** does the anonymous paper need an AI-use statement, and if so,
  how can it satisfy AAAI without an unnecessary task-by-task inventory?
- **Current official rule:** AAAI-27 permits judicious generative-AI use and
  keeps the author fully responsible. AAAI's standing publication policy says
  the role of AI systems must be properly documented in the manuscript. Neither
  source requires vendor/model names, prompts, logs, or a dedicated section.
- **Practice sample:** extracted 29 accepted AAAI-26 final PDFs, evenly sampled
  across volume 40 issues 28 (Machine Learning V), 37 (NLP II), and 42
  (Philosophy/Ethics). A disclosure-pattern search found zero visible
  authoring-AI statements. This does not establish permission to omit: AI use
  is unknown and AAAI-26's conference-specific policy was more restrictive.
- **Decision:** reject both omission and the earlier long inventory. Final
  statement: `Generative AI tools supported research workflow automation,
  implementation, and manuscript preparation. The author remains responsible
  for all reported content.` It is broad, accurate, anonymous, and placed
  immediately before References.
- **Venue correction:** the user's screenshot was the special AI for Social
  Impact venue. This paper belongs in the `AAAI 2027 Conference` Main Technical
  Track portal because it is a technical mechanistic study, not an AISI social
  application paper.
- **Final main PDF:** 7 US-Letter pages / 976,325 bytes; page 7 visually
  inspected; no overfull boxes or undefined citations. SHA-256:
  `90eef375877e95c59712a7800aea2fdd78a3d96567d563e01f0827a317e24c07`.
- **Audit artifact:** `Paper/Research/aaai27-ai-assistance-audit.md`.

### S15 — solo-author AAAI proceedings audit (COMPLETED)

- **Question:** are sole-authored AAAI Main Technical Track papers actually
  present in the accepted record, and what can that evidence legitimately say
  about this submission?
- **Source universe:** every article summary in the 43 official AAAI-26 Main
  Technical Track proceedings issues, volume 40 nos. 1--43. Special tracks,
  journal track, IAAI, EAAI, demonstrations, and student abstracts were
  excluded.
- **Method:** downloaded the 43 issue tables of contents, extracted each title,
  canonical article URL, and comma-separated author line, and counted entries
  with exactly one listed author. Manually inspected relevant solo article
  pages for the official author/affiliation display.
- **Result:** 4,149 accepted article records scanned; 26 sole-authored records;
  observed accepted-record solo share **0.627%** (about one per 160 accepted
  papers). This is a local analysis of official records, not an AAAI-published
  rate and not a solo-submission acceptance rate.
- **Proof-of-existence:** Omar Claflin's *Feature Integration Spaces: Joint
  Training Reveals Dual Encoding in Neural Network Representations* is listed
  with one author and affiliation `Independent` in AAAI-26 Technical Track on
  Philosophy and Ethics of AI.
- **Overall base rate:** AAAI's official AAAI-26 opening material reports Main
  Track acceptance at approximately **17.5%**.
- **Decision:** solo authorship is allowed and demonstrably accepted, but is
  invisible during anonymous review and does not improve the score. Keep the
  paper's readiness label bounded: `submission-ready and genuinely
  competitive, but not acceptance-predictable`.
- **Artifact:** `Paper/Learning/solo-aaai-reality-check.md`.

### S16 — final frozen submission verification (COMPLETED)

- **Scope:** one last read-only policy, source, PDF, citation, anonymity,
  archive, and reproducibility pass. New experiments and stylistic churn were
  out of scope unless a concrete defect appeared.
- **Live rules:** Main Track deadlines remain July 21/28/31 at 23:59 UTC-12.
  The current OpenReview invitation confirms required title, author profile,
  TL;DR (250 characters), abstract (5,000 characters), topics, institution
  countries, reciprocal-reviewer status, policy attestations, and optional
  10/5/10/50 MB upload fields. The paper belongs in Main Track, not AISI.
- **Author Kit:** downloaded current `AuthorKit27.zip`; the repository's
  `aaai2027.sty` and `aaai2027.bst` match byte-for-byte. Anonymous syntax is
  `[submission]{aaai2027}`. Forbidden-layout scan is clean.
- **Build and visual QA:** main/supplement/checklist compile to 7/3/2 US-Letter
  pages. Rebuilt and upload copies have identical extracted text and rendered
  pixels. All 12 pages inspected at 144 dpi; fonts embedded; no clipping,
  overlap, illegible plot, isolated heading, manual reference break, encryption,
  or attachment. Only harmless underfull-line notices remain.
- **Metadata/citations:** abstract text matches `abstract.txt`; TL;DR 202
  characters; abstract 1,500. Twenty-two cited keys, 22 bibliography records,
  zero missing/unused; all 22 URLs returned HTTP 200. PDF and archive searches
  found no author identity, local path, personal email/repository, or secret.
- **Archive:** 49,495,215 bytes; ZIP integrity pass; 49/49 manifest hashes pass.
  Fresh reruns reproduce geometry, behavior, Qwen sequence, included
  layerwise branches, headline causal, external causal, external-transfer,
  matched-subset, and all three figures. The causal JSON differs only in its
  sanitized protocol path. The stored layerwise summary also contains the
  disclosed omitted MPS branch; included 8B-CUDA/70B-CUDA results and their
  contrast match exactly.
- **Decision:** GREEN AND FROZEN. No manuscript or science change was warranted.
  Added the definitive July 20 one-sitting checklist to
  `Paper/Submission/AAAI27/START-HERE-SUBMISSION-ROADMAP.md`.

### S17 — anonymous AI-disclosure minimization (COMPLETED)

- **Question:** can the page-7 disclosure satisfy AAAI policy without implying
  a solo author or presenting an unnecessary task inventory?
- **Rule check:** the AAAI-27 call permits judicious generative-AI use and makes
  authors responsible; the standing publication policy requires the AI
  system's role to be documented. Neither requires that the manuscript repeat
  the responsibility rule, name products, or enumerate individual operations.
- **Final wording:** `Generative AI tools assisted with implementation and
  manuscript preparation.`
- **Why:** one sentence truthfully states the broad roles, omits the redundant
  singular `The author` sentence, and does not reveal author count.
- **Verification:** rebuilt to seven US-Letter pages with no errors, undefined
  references/citations, or overfull boxes. Fresh page-7 rendering shows clean
  conclusion, disclosure, and natural References flow.
- **Decision:** keep the one-sentence disclosure; do not remove it or expand it.

### S18 — arXiv and future-venue policy routing (COMPLETED)

- **Question:** can the paper appear on arXiv or be sent elsewhere while under
  AAAI-27 review?
- **Live rule:** AAAI-27 explicitly permits preprint servers such as arXiv and
  non-archival workshops. The same or substantially similar work cannot be
  submitted to another archival conference or journal until AAAI decides or
  the AAAI submission is withdrawn.
- **Blind-review guardrail:** the AAAI files stay anonymous and contain no
  pointer to the preprint; the non-anonymous preprint must not state that the
  work was submitted to AAAI-27.
- **arXiv routing:** recommend `cs.CL` primary, `cs.LG` cross-list, a named
  source submission, and the conservative arXiv perpetual non-exclusive paper
  license while publication rights are unresolved. Endorsement is needed only
  if the arXiv account workflow requests it.
- **Decision:** finish AAAI first, then post arXiv; do not open a second archival
  review path concurrently. Full guide:
  `Paper/Learning/arxiv-and-other-venues-guide.md`.

### S19 — final human handoff map and artifact verification (COMPLETED)

- **Scope:** one final read-only official-rule and artifact audit, followed by a
  single plain-English handoff map. No manuscript edit was allowed without a
  concrete defect.
- **Live rules:** official AAAI-27 pages still give July 21/28/31 at 11:59 PM
  UTC-12; seven technical pages plus reference/checklist pages; separate
  reproducibility checklist; anonymous main and support files; optional
  supplement/code; arXiv permission; no simultaneous archival review; and
  manuscript documentation of AI-system roles.
- **Build equivalence:** fresh `latexmk -g` builds of main, supplement, and
  checklist completed at 7/3/2 pages. Each rebuilt PDF matched its upload copy
  in extracted text and every 144-dpi rendered page. Log scan found no overfull
  box, undefined citation/reference, LaTeX error, or fatal error.
- **Visual/format QA:** inspected contact sheets for all 12 pages and full-size
  main pages 1 and 7, supplement page 3, and checklist page 2. All pages are US
  Letter, fonts are embedded, encryption is off, attachments are absent, and
  the natural page-7 Conclusion-to-AI-Assistance-to-References flow is clean.
  The paper source contains no manual spacing, page-break, or geometry command;
  layout commands in `checklist.tex` belong to the official checklist template.
- **Metadata/anonymity:** no Barath/Velmurugan/email/home-path identity string
  appears in extracted PDF text. PDF metadata has no Title, Author, Subject, or
  Keywords identity field. Abstract matches `abstract.txt` exactly (1,500
  characters); TL;DR is 202 characters.
- **Citations:** 22 unique citation keys, 22 bibliography entries, zero missing,
  zero unused.
- **Archive:** original 49,495,215-byte upload ZIP passes decompression and all
  49 manifest hashes. Fresh extraction contains no author identity, local path,
  credential, or private key. Geometry, behavior, Qwen, 8B/70B layerwise,
  headline causal, external causal, external transfer, and matched-subset
  analyses completed. Geometry, behavior, Qwen, external-transfer, joined CSV,
  and matched-subset summaries are byte-identical; causal outputs differ only
  in the intentionally sanitized protocol path; the included CUDA layerwise
  branches and contrast match exactly. All three regenerated figures are
  byte-identical.
- **Handoff artifact:** `Paper/00-START-HERE-FINAL-MAP.md` now consolidates the
  page-count explanation, six-step upload workflow, exact four-file mapping,
  human-only truth fields, deadlines, all governing rules, after-submission
  timeline, reading library, arXiv/rebuttal routing, and stop rule.
- **Decision:** GREEN AND FROZEN. No paper or science change is justified.

### S20 — contemporaneous-work Gargantua hardening (COMPLETED)

- **Date:** 2026-07-19.
- **Stopping rule:** reopen the freeze once; change only a verified defect or a
  directly relevant primary-source omission; stop when the next attack yields
  no justified change.
- **Research audit:** inspected the existing Nia research-paper index first.
  Indexing arXiv:2607.04432 failed because Nia authentication had expired, so
  the source was verified through its primary arXiv abstract, HTML, and PDF.
  Targeted current searches found no prior matched Llama-3.1-8B/70B
  natural-state intervention or equivalent joint geometry/readout/causal plus
  width-controlled result.
- **Valid literature repair:** added Kargi Chauhan and Aditya Shah, *Covert
  Trait Propagation Is Representation Alignment: Mechanistic Evidence from
  Hidden-Channel Distillation* (arXiv:2607.04432v1, July 5, 2026). Their LLM
  section is observational, uses Llama-3.2-1B base/instruct, includes no causal
  LLM intervention, and names 7B+ scaling as open. The paper now credits their
  broad geometry/behavior separation without yielding the headline novelty.
- **Manuscript changes:** combined Chauhan and Shah with Zur et al. in the
  nearest-comparison row, added one bibliography record, defined `X`, `Y`, and
  `w` in the covariance explanation, and corrected three punctuation/clarity
  details. Abstract, TL;DR, estimands, results, tables 2–5, figures, scope, and
  AI disclosure are unchanged. A larger separate table row was rejected after
  it created an almost-empty reference page; the final exact comparison stays
  within seven pages without layout tricks.
- **Citation gate:** 23 unique used keys, 23 bibliography records, zero missing,
  zero unused; all 23 URLs returned HTTP 200.
- **Reproducibility gate:** fresh archive extraction passed ZIP integrity and
  49/49 manifest hashes; identity/secret scans returned zero hits; all Python
  files compile. Geometry, behavior, Qwen, external validation, joined CSV, and
  matched-subset outputs are byte-identical. Causal outputs match after ignoring
  the intentionally sanitized protocol path. The fresh 8B/70B layerwise
  branches and paired contrast match the stored full summary. All three figures
  are byte-identical.
- **PDF gate:** forced rebuilds completed at 7/3/2 pages with zero overfull box,
  undefined citation/reference, LaTeX error, or fatal error. All 12 pages were
  rendered at 144 dpi and inspected individually; tables, plots, equations,
  conclusion, AI note, and natural reference flow are clean. US Letter, embedded
  Type 1 fonts, no encryption, and anonymous content remain intact.
- **Final upload files:** main 976,430 bytes,
  `cc405b0b72ef31194c58697a2ee41f18c1a72ac7172bf5b7458415d46abf4edb`;
  supplement 179,437 bytes,
  `e53f0e28413a898874660de0abf1333cbeb5765ffe34ce603fc332be7ab16b7d`;
  checklist 95,711 bytes,
  `7a1b09ad244ee6e8079ad21a37c62cf8a8eea23face02b1ce86890a4f94b579c`;
  code/data ZIP unchanged at 49,495,215 bytes,
  `ed069bd9c44f947c6dc618049b48f2f8097e42ae63fc93a6695bbbd8f104c885`.
- **Decision:** GREEN AND REFROZEN. The only valid new attack was repaired;
  another pass found no evidence-backed edit.
