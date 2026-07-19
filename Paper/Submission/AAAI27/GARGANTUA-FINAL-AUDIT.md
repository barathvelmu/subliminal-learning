# Gargantua final audit

Completed: **July 18, 2026**

AI-assistance mini-audit and final-file refresh: **July 19, 2026**

Final sleep-pass freeze verification: **July 19, 2026**

Final contemporaneous-work hardening pass: **July 19, 2026**

Gold-close disclosure and adversarial pass: **July 19, 2026**

## Baby-food verdict

**GREEN. Stop adding experiments or citations for this submission.**

Four waves of hostile review found no fatal scientific,
statistical, writing, visual, anonymity, or reproducibility defect. The first
wave repaired submission blockers and claim-precision risks. The second wave
simulated novelty, correctness, significance, evidence, clarity, and
reproducibility attacks, then repaired every valid pre-submission concern. The
four upload files now pass the final gate.

This does not guarantee acceptance. It means the remaining risk is normal
reviewer judgment about significance and scope, not a known broken claim or
file.

## Gold-close disclosure and adversarial pass

The final pass addressed the author's remaining disclosure concern without
reopening the science. A primary-policy comparison confirmed that AAAI-27 is
part of a genuine transition: AAAI-26 prohibited generated manuscript text
outside experimental analysis, while AAAI-27 does not repeat that prohibition
and permits judicious generative-AI use. Neighboring venues apply materially different disclosure rules, so their
accepted papers are not direct templates for AAAI-27. AAAI's standing policy
still requires the system's role to be documented in the manuscript.

The approved final wording is:

> **AI Assistance.** Generative AI tools supported implementation and
> manuscript preparation.

`Supported` makes the human/tool relationship clearer without hiding either
documented role. The manuscript diff for this pass contains exactly this
one-line wording change. No result, number, claim, citation, equation, table,
figure, abstract, TL;DR, limitation, or experimental artifact changed.

All three PDFs were force-rebuilt at 7/3/2 pages. The logs contain no overfull
box, undefined citation/reference, LaTeX error, or fatal error. All 12 pages
were rendered at 144 dpi and inspected individually. Page 7 retains a clean
Conclusion-to-AI-Assistance-to-References flow. All files remain US Letter,
unencrypted, attachment-free, anonymous, and fully font-embedded. The code/data
ZIP is unchanged; decompression and all 49 manifest hashes pass.

Decision: **GOLD, GREEN, AND CLOSED.** The final adversarial attack produced no
additional evidence-backed manuscript change.

## Final contemporaneous-work hardening pass

The freeze was reopened once under a strict rule: change the paper only for a
verified defect or a directly relevant primary source, then stop when the pass
produces no further justified change.

- A July 19 primary-source search found Chauhan and Shah's July 5 preprint,
  *Covert Trait Propagation Is Representation Alignment* (arXiv:2607.04432).
  Its transformer evidence is an observational Llama-3.2-1B base/instruct
  comparison. The paper explicitly reports no causal intervention inside the
  LLM and names scaling to 7B+ as open. It therefore supports the broad
  geometry-versus-behavior distinction but does not contain our matched
  Llama-3.1-8B/70B natural-state intervention or decimal-width analysis.
- The source is now cited in the nearest-prior-work table. The table combines
  it with the parent token-entanglement comparison, preserving the seven-page
  layout and natural reference flow. A local paper card and rebuttal response
  were added.
- One notation sentence now defines the two directional variables and decimal
  width in the law-of-total-covariance explanation. Three small punctuation
  issues were corrected. No result, estimand, interpretation, figure, abstract,
  or TL;DR changed.
- All 23 cited source URLs returned HTTP 200. There are 23 used citation keys,
  23 bibliography records, zero missing citations, and zero unused records.
- A fresh extraction of the unchanged 49,495,215-byte code/data ZIP again
  passed all 49 manifest hashes and identity/secret scans. Every included
  analysis reran. All deterministic results match; causal files differ only in
  the documented sanitized protocol path, and the stored full layerwise file's
  included CUDA branches match the fresh headline-only rerun. All three figures
  reproduce byte-for-byte.
- Main, supplement, and checklist were rebuilt at 7/3/2 pages. All 12 pages
  were rendered at 144 dpi and inspected individually. There is no clipping,
  overlap, broken table, unreadable figure, isolated reference page, or manual
  spacing/page-break command.

Decision: **GREEN AND REFROZEN.** The pass found one useful citation and one
minor clarity fix, then no further evidence-backed edit.

## Final sleep-pass freeze verification

The final pass made **no scientific or manuscript edits** because it found no
defect that justified disturbing the frozen paper.

- Re-read the live Main Track call, submission instructions, supplement rules,
  and unauthenticated OpenReview invitation schema. Deadlines, venue, required
  fields, reciprocal-reviewer declaration, and 10/5/10/50 MB caps still match
  `START-HERE-SUBMISSION-ROADMAP.md`.
- Downloaded the current 5.2 MB AAAI-27 Author Kit from
  <https://aaai.org/authorkit27/>. The local `aaai2027.sty` and
  `aaai2027.bst` match it byte-for-byte. The official anonymous syntax remains
  `\usepackage[submission]{aaai2027}`.
- Recompiled main, supplement, and checklist from source. Builds complete with
  zero overfull boxes, undefined references, missing citations, or errors.
  Rebuilt and upload copies have identical extracted text and rendered pixels.
- Rendered all 12 upload-PDF pages at 144 dpi and visually inspected every
  page. There is no clipping, overlap, broken table, illegible plot, isolated
  heading, forbidden reference break, or unexpected blank page.
- Confirmed US Letter dimensions, embedded fonts, no encryption, no embedded
  attachments, and anonymous metadata for every PDF.
- Recounted 23 used citation keys and 23 bibliography records, with zero
  missing or unused entries. All 23 reference URLs returned HTTP 200 on July
  19.
- Fresh-extracted the 49,495,215-byte code/data ZIP. ZIP integrity and all
  49 manifest hashes pass. Searches found no author name, local path, personal
  email, personal repository, API credential, or private key.
- Reran geometry, behavior, Qwen sequence, 8B/70B layerwise, headline causal,
  external causal, external-transfer, matched-subset, and figure scripts from
  the extracted archive. Deterministic summaries and all three figures match.
  The causal summary's only textual difference is the intentionally sanitized
  in-package protocol path. The full stored layerwise summary contains the
  disclosed MPS control omitted from the size-bounded archive; its included
  8B-CUDA, 70B-CUDA, and paired-contrast branches match exactly.
- Rechecked `abstract.txt` against the LaTeX abstract after normalization:
  exact match. TL;DR is 202 characters, below the live 250-character limit;
  abstract is 1,500 characters, below the 5,000-character limit.

Decision: **GREEN AND FROZEN.** The only remaining steps require the human
author's identity, truthful declarations, final responsibility read, and
OpenReview save action.

### Post-freeze AI-disclosure refinement

At the author's request, the disclosure was reduced from two sentences to one:

> **AI Assistance.** Generative AI tools supported implementation and
> manuscript preparation.

This remains truthful and satisfies the standing AAAI requirement to document
the AI system's role. It removes the unnecessary responsibility sentence,
which repeated the policy and used singular `author` language. The final line
names no product, prompt, agent, task inventory, or number of human authors.
The main PDF was rebuilt and page 7 was re-rendered and visually inspected;
page count, reference flow, and layout remain clean.

## What the audit caught and fixed

1. The official author kit forbids `\clearpage` and says references must flow
   directly after the paper. The manual References break was removed.
2. AAAI's publication policy requires the role of any AI system to be
   documented in the manuscript. A July 19 focused audit shortened the original
   task inventory to a broad, accurate one-sentence `AI Assistance` paragraph.
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

- 23 citation keys are used.
- 23 bibliography entries exist.
- Missing citations: **0**.
- Uncited bibliography entries: **0**.
- All 23 source URLs resolved during the audit.
- Titles and bibliographic metadata were checked against the primary paper
  pages or proceedings records.
- The comparison table states the exact remaining gap rather than claiming that
  activation patching, sequence scoring, or subliminal learning is itself new.

## Upload-file gate

| File | Pages / bytes | Limit | Result |
|---|---:|---:|---|
| `output/pdf/aaai27-main.pdf` | 7 pages / 976,432 bytes | 10 MB; at most 7 technical and 9 total pages | PASS |
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

- main PDF: `ffbf213b466c5ed56566e638a41834b13da3527610449a5529e81047abe98726`
- supplement PDF: `3d33166bb265f1a31e855368d7c93ed79ecdd9c1a4a2aa3740c6bb71d261cafb`
- checklist PDF: `32de1eed7af8b3cf9792e45e205a61dc14e36d05d75dcf671a9096efc95c7229`
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
the broad role: support for implementation and manuscript preparation. See
`../../Research/aaai27-ai-assistance-audit.md`.

Official policy sources:

- <https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/>
- <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- <https://aaai.org/authorkit27/>

## Human gate that cannot be automated

The sole author must now read the final main PDF and truthfully accept
responsibility for its claims, citations, AI-use disclosure, conflicts,
affiliation, reciprocal-reviewer declaration, and originality policy. Then
follow `START-HERE-SUBMISSION-ROADMAP.md`.
