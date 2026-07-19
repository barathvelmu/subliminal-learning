# Gargantua final audit

Completed: **July 18, 2026**

## Baby-food verdict

**GREEN. Stop adding experiments for this submission.**

Three independent hostile reviews found no fatal scientific, statistical,
writing, visual, anonymity, or reproducibility defect. Two real submission
blockers and two claim-precision risks were found and repaired. The four upload
files now pass the final gate.

This does not guarantee acceptance. It means the remaining risk is normal
reviewer judgment about significance and scope, not a known broken claim or
file.

## What the audit caught and fixed

1. The official author kit forbids `\clearpage` and says references must flow
   directly after the paper. The manual References break was removed.
2. AAAI's publication policy requires the role of any AI system to be
   documented in the manuscript. An accurate anonymous `Use of AI Systems`
   paragraph was added before the bibliography.
3. The official template's required `caption` package was restored. Custom
   float-packing and table-spacing overrides were removed.
4. Research Question 2 formerly said the observational trace *predicts* causal
   timing, although the analysis tests whether the two show the same cross-scale
   change. The question and claim-to-test row now match the estimand.
5. A single exact remaining-block sensitivity no longer claims to eliminate all
   relative-depth artifacts.
6. The undefined phrase `becomes sufficient earlier` now says the measured
   result: the donor state `gains strong control earlier`.
7. The failed external-validation gate is explicitly labeled a complete,
   transparency-only negative check that did not change the headline claims.
8. Provider account balances, instance IDs, and unverifiable abbreviated hashes
   were removed from the reviewer supplement. The operational record remains in
   the private project audit; the submitted package has a full SHA-256 manifest.
9. A stale internal note claiming that all three large external raw arrays were
   inside the ZIP was corrected. The ZIP README already disclosed the size-cap
   omission accurately.
10. The supplement was tightened from four pages, with an almost-empty last
    page, to three clean pages without removing evidence needed to audit a claim.

## Science and statistics gate

- Fatal defects: **0**.
- New experiment required before submission: **no**.
- Headline causal change: `+0.28581`, 95% crossed-bootstrap interval
  `[+0.27162, +0.29991]`, with 18/18 animal-level increases.
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
| `output/pdf/aaai27-main.pdf` | 7 pages / 975,942 bytes | 10 MB; at most 7 technical and 9 total pages | PASS |
| `output/pdf/aaai27-checklist.pdf` | 2 pages / 95,711 bytes | 5 MB | PASS |
| `output/pdf/aaai27-supplement.pdf` | 3 pages / 179,843 bytes | 10 MB | PASS |
| `output/aaai27-code-data.zip` | 49,488,318 bytes | 50 MB | PASS, only 511,682 bytes spare |

Main-paper technical content stays within page 7. Deferred Tables 4 and 5
appear at the top of page 7, followed naturally by References. No manual page
break is used.

All PDFs are US Letter, anonymous, readable, and use embedded Type 1 fonts.
Text and metadata scans found no author name, email, personal repository, or
local filesystem path. The ZIP contains 47 files: one manifest plus 46 entries
whose SHA-256 checks all pass. ZIP integrity and decompression pass.

Final SHA-256 digests:

- main PDF: `765f53aaf058bfb6fbd72ffe2bb0f28f8c6d2981930193b20603821e2a826380`
- supplement PDF: `d03b9f8e38a0866218f13d4cbe6ce4bdac35c7fe3f0f0d7ba086b51b4e8fce28`
- checklist PDF: `03fd7c32831eb2a0c23af9ccd63d9138ba0aeff6760c8b1c89a41e971eeacf52`
- code/data ZIP: `7a3b2d9bbbc622a16893a7fb5fc40d2f5377cc63b00267c3dda976cd6b9602b2`

## AI-use verdict

Do not try to fool an AI detector. AAAI permits generative-AI assistance but
holds the human author responsible and requires the role to be documented. The
paper now does that directly. Its defense is specific evidence, accurate
citations, honest limitations, a reproducible artifact, and the author's own
full read and responsibility.

Official policy sources:

- <https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/>
- <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- <https://aaai.org/authorkit27/>

## Human gate that cannot be automated

The sole author must now read the final main PDF and truthfully accept
responsibility for its claims, citations, AI-use disclosure, conflicts,
affiliation, reciprocal-reviewer declaration, and originality policy. Then
follow `START-HERE-SUBMISSION-ROADMAP.md`.
