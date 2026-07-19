# Evidence index

Use this to answer reviewers with page/file pointers instead of vague prose.

| Claim or concern | Main paper | Supplement | Anonymous artifact |
|---|---|---|---|
| Exact scope: frozen prompting, not training mechanism | pp. 1, 6–7 | p. 1 | README and S5 protocol non-claims |
| Headline geometry/readout/causal values | Table 3, p. 6 | Tables 2–3, pp. 2–3 | `data/summaries/geometry_8b70b_cuda_summary.json`, `layerwise_probe_8b70b_cuda_rerun.json`, `causal_patch_s5_8b70b_cuda_summary.json` |
| Prior-work novelty boundary | Table 1, p. 4 | n/a | n/a |
| Exact causal intervention and estimator | pp. 3–4, Eqs. 3–4 | pp. 2–3 | `protocol/s5-causal-protocol-and-amendment.md` and collector/analyzer code |
| Estimator repair chronology and null failure | p. 3 | p. 2 | S5 protocol/amendment; no third-party timestamp claim |
| Every measured donor/recipient coefficient | Table 4, p. 6 | Table 3, p. 3 | causal summary JSON |
| Donor/concept specificity | Table 5 and p. 4 | p. 3 | causal summary and raw causal arrays |
| Identity, duplicate, condition number, pair-direction controls | p. 4 | p. 3 | causal summary JSON |
| Leave-one-pair-cluster sensitivity | p. 4 | p. 3 | causal summary JSON |
| 1,110 probes versus 256 causal numbers | pp. 1 and 3 | p. 2 | all geometry, layerwise, and causal arrays |
| Exact-256 geometry/readout sensitivity | p. 4 | p. 2 | `data/summaries/matched_subset_sensitivities.json` and its analysis script |
| Raw-logit crossed interval | p. 4 | p. 3 | same sensitivity JSON/script |
| Backend drift | p. 4 | pp. 1–2 | MPS/CUDA arrays and summaries described in README |
| Coarse relative-depth and exact-eight-block limits | pp. 4 and 6 | p. 3 | causal summary JSON |
| Off-manifold hybrid limitation | p. 6 | S5 intervention description | S5 protocol |
| Qwen exact scorer validation and width confound | pp. 3 and 5–6 | p. 3 | sequence summary, arrays, and analyzer |
| Fixed-concept/pair interpretation of intervals | p. 3 and p. 6 | protocol/results text | deterministic bootstrap code |
| External prediction null and BH correction | pp. 6–7 | Table 4, p. 3 | external validation summary, joined CSV, and S8 plan |
| Artifact omissions and checkpoint limitation | conclusion p. 7 | n/a | README |
| AI-use disclosure | p. 7 | n/a | n/a |

## One-command sensitivity rerun

From the extracted `code-data` archive root:

```bash
python code/scaling/analyze_matched_subset_sensitivities.py \
  --geometry-8b data/raw/full_probe_geometry_8b_cuda.npz \
  --geometry-70b data/raw/full_probe_geometry_70b_cuda.npz \
  --layerwise-8b data/raw/layerwise_probe_layerwise_8b_cuda.npz \
  --layerwise-70b data/raw/layerwise_probe_layerwise_70b_cuda.npz \
  --causal-8b data/raw/causal_patch_s5_patch_8b_cuda.npz \
  --causal-70b data/raw/causal_patch_s5_patch_70b_cuda.npz \
  --output data/summaries/matched_subset_sensitivities.json
```

Expected headline sensitivity values:

- exact-256 geometry delta: `-0.06461159`, animal-bootstrap 95% interval
  `[-0.12433546,-0.00334115]`;
- exact-256 specificity delta: `-0.08528531`, interval
  `[-0.14318376,-0.03029252]`;
- exact-256 readout AUC delta: `+0.03354393`, interval
  `[-0.00524355,+0.07145235]`;
- raw-logit causal AUC delta: `+0.27437331`, crossed interval
  `[+0.25934387,+0.28936288]`, 18/18 positive.

## Artifact boundary

The ZIP intentionally omits three non-headline arrays to stay below 50 MB:

- `full_probe_external_zoo_8b_mps.npz`;
- `layerwise_probe_external_zoo_8b_mps.npz`; and
- `layerwise_probe_layerwise_full_8b_mps.npz`.

Reviewers can rerun every main headline analysis and the external outcome join
from included inputs. They cannot regenerate the two external instruments or
the full MPS layerwise control from raw arrays without recollecting them from a
public checkpoint. Historical immutable checkpoint commit hashes were not
recorded.
