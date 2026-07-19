# Gargantua final audit

Completed: **July 18, 2026**

AI-assistance mini-audit and final-file refresh: **July 19, 2026**

## Baby-food verdict

**GREEN. Stop adding experiments for this submission.**

Two waves of independent hostile review found no fatal scientific,
statistical, writing, visual, anonymity, or reproducibility defect. The first
wave repaired submission blockers and claim-precision risks. The second wave
simulated novelty, correctness, significance, evidence, clarity, and
reproducibility attacks, then repaired every valid pre-submission concern. The
four upload files now pass the final gate.

This does not guarantee acceptance. It means the remaining risk is normal
reviewer judgment about significance and scope, not a known broken claim or
file.

## What the audit caught and fixed

1. The official author kit forbids `\clearpage` and says references must flow
   directly after the paper. The manual References break was removed.
2. AAAI's publication policy requires the role of any AI system to be
   documented in the manuscript. A July 19 focused audit shortened the original
   task inventory to a broad, accurate two-sentence `AI Assistance` paragraph.
   The policy does not require tool names, prompts, or a conspicuous disclosure
   section.
3. The official template's required `caption` package was restored. Custom
   float-packing and table-spacing overrides were removed.
4. Research Question 2 formerly said the observational trace *predicts* causal
   timing, although the analysis tests whether the two show the same cross-scale
   change. The question and claim-to-test row now match the estimand.
5. A single exact remaining-block sensitivity no longer claims to eliminate all
   relative-depth artifacts.
6. The undefined phrase `becomes sufficient earlier` now says the measured
   result: the donor state `gains strong control earlier`.
7. The failed external-validation gate is reported directly as an external
   limitation rather than defensively labeled a `transparency-only` check.
8. Provider account balances, instance IDs, and unverifiable abbreviated hashes
   were removed from the reviewer supplement. The operational record remains in
   the private project audit; the submitted package has a full SHA-256 manifest.
9. A stale internal note claiming that all three large external raw arrays were
   inside the ZIP was corrected. The ZIP README already disclosed the size-cap
   omission accurately.
10. The supplement was tightened from four pages, with an almost-empty last
    page, to three clean pages without removing evidence needed to audit a claim.
11. The closest-prior-work table now credits the relevant intervention,
    size-comparison, and positional-bias evidence precisely, while isolating the
    narrower unresolved gap this paper tests.
12. The stimulus-set description now distinguishes the 1,110-number geometry
    and readout analyses from the outcome-blind 256-number causal subset.
13. A deterministic 256-number matched-subset analysis now tests whether input
    support alone explains the cross-scale changes. The geometry and
    specificity decreases remain resolved; the readout change remains
    unresolved.
14. A raw-target-logit causal analysis now shows a resolved cross-scale increase,
    so the causal result is not an artifact of converting logits to probability.
15. The causal estimator record now says `recorded amendment`, not
    `timestamped amendment`. The archive states exactly what is and is not
    independently timestamped.
16. The causal limitation now distinguishes a natural donor state from the
    potentially off-manifold donor/recipient hybrid computation.

## Science and statistics gate

- Fatal defects: **0**.
- New experiment required before submission: **no**.
- Headline causal change: `+0.28581`, 95% crossed-bootstrap interval
  `[+0.27162, +0.29991]`, with 18/18 animal-level increases.
- Raw-target-logit causal change: `+0.27437`, 95% crossed-bootstrap interval
  `[+0.25934, +0.28936]`, with 18/18 increases and all 20,000 bootstrap draws
  positive.
- On the exact 256-number causal subset, geometry decreases by `-0.06461`
  `[-0.12434, -0.00334]`, specificity decreases by `-0.08529`
  `[-0.14318, -0.03029]`, and the readout change is unresolved at `+0.03354`
  `[-0.00524, +0.07145]`.
- Controls passed: permuted donor, all 17 wrong-concept shifts, identity patch,
  duplicate forward, raw-logit outcome, both direction halves, condition
  number, and leave-one-pair-cluster sensitivity.
- Scope is explicit: one same-release Llama pair, five coarse depths, a frozen
  prompting channel, full-state intervention, and a confounded Qwen
  tokenizer/model-family boundary.
- The negative external check reports all three frozen in-house predictors,
  its multiplicity correction, the single aggregate seed, the zero-heavy
  outcome, and the unavailable preregistered covariate.

## Independent numerical rerun

The anonymous package's analyzers were run again from saved arrays. These fresh
outputs match the submitted summaries numerically:

- behavior and static geometry;
- the local size comparison;
- exact Qwen sequence scoring;
- the 8B/70B layerwise readout;
- the 8B/70B causal patch;
- the external-zoo causal patch; and
- the external-transfer validation and joined CSV.

The stored layerwise summary also contains the documented MPS backend branch,
which the headline-only fresh rerun intentionally omitted. The common 8B-CUDA
and 70B-CUDA branches match exactly.

## References and prior-work gate

- 22 citation keys are used.
- 22 bibliography entries exist.
- Missing citations: **0**.
- Uncited bibliography entries: **0**.
- All 22 source URLs resolved during the audit.
- Titles and bibliographic metadata were checked against the primary paper
  pages or proceedings records.
- The comparison table states the exact remaining gap rather than claiming that
  activation patching, sequence scoring, or subliminal learning is itself new.

## Upload-file gate

| File | Pages / bytes | Limit | Result |
|---|---:|---:|---|
| `output/pdf/aaai27-main.pdf` | 7 pages / 976,325 bytes | 10 MB; at most 7 technical and 9 total pages | PASS |
| `output/pdf/aaai27-checklist.pdf` | 2 pages / 95,711 bytes | 5 MB | PASS |
| `output/pdf/aaai27-supplement.pdf` | 3 pages / 179,437 bytes | 10 MB | PASS |
| `output/aaai27-code-data.zip` | 49,495,215 bytes | 50 MB | PASS, only 504,785 bytes spare |

Main-paper technical content stays within page 7. Deferred Tables 4 and 5
appear at the top of page 7, followed naturally by References. No manual page
break is used.

All PDFs are US Letter, anonymous, readable, and use embedded Type 1 fonts.
Text and metadata scans found no author name, email, personal repository, or
local filesystem path. The ZIP contains 50 files: one manifest plus 49 entries
whose SHA-256 checks all pass. ZIP integrity and decompression pass.

Final SHA-256 digests:

- main PDF: `90eef375877e95c59712a7800aea2fdd78a3d96567d563e01f0827a317e24c07`
- supplement PDF: `059366d0d9cdcbdd2c8b94739f0344bd630153c0ecbec0d0362679593bcd30b4`
- checklist PDF: `03fd7c32831eb2a0c23af9ccd63d9138ba0aeff6760c8b1c89a41e971eeacf52`
- code/data ZIP: `ed069bd9c44f947c6dc618049b48f2f8097e42ae63fc93a6695bbbd8f104c885`

## Rebuttal shield

`../../Rebuttal/` contains the author-side defense package: official review
rules and timeline, mock reviews, an attack matrix, evidence index, response
bank, October checklist, and the remaining human decisions. This is not an
extra upload. It is preparation for author feedback if the paper reaches Phase
2. A Phase 1 rejection has no rebuttal window, and nothing in the submitted
PDFs or supplements can be changed after July 31.

## AI-use verdict

Do not try to fool an AI detector, and do not volunteer a dramatic task-by-task
inventory that AAAI does not request. AAAI permits judicious generative-AI
assistance, holds the human author responsible, and requires the role to be
documented. The paper now uses the shortest wording that remains accurate about
the broad role: research workflow automation, implementation, and manuscript
preparation. See `../../Research/aaai27-ai-assistance-audit.md`.

Official policy sources:

- <https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/>
- <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- <https://aaai.org/authorkit27/>

## Human gate that cannot be automated

The sole author must now read the final main PDF and truthfully accept
responsibility for its claims, citations, AI-use disclosure, conflicts,
affiliation, reciprocal-reviewer declaration, and originality policy. Then
follow `START-HERE-SUBMISSION-ROADMAP.md`.
