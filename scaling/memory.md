# memory.md — Subliminal Learning: Scaling Follow-up (ground truth)

> Updated after every step. Newest entry on top. This survives context compaction — treat it as ground truth.
> Companion files: `experiments.md` (reproducible config->command->artifact log), `artifact.md` (what we installed/created).

## >> HANDOFF: START HERE (new session, or Codex picking this up) <<
Read this file top-to-bottom, then `experiments.md`. That is the full state; you can continue with no other context.
- **Where we are:** S1-S5 and the outcome-blinded S8 external validation are complete, including the preregistered causal
  assistant-state intervention. Every paid machine is destroyed, Vast shows zero
  active instances, and every raw artifact is local. Headline: static
  number/animal output geometry **weakens decisively at 70B**, the observational
  readout has **no resolved AUC change**, but causal donor control arrives **much earlier**:
  corrected causal AUC 8B `0.254` vs 70B `0.540`, paired `+0.286
  [+0.272,+0.300]`, all 18/18 animals increase. Digit-sequence scoring still
  shows the separate length confound. The structured 11-page paper, learning
  guide, novelty audit, paper cards, preregistration, and reproducibility map are
  in `Paper/`. S8 used Blank et al.'s released 16-animal Llama outcomes: their
  steering peak predicts transfer (rho `.768`), while our causal timing does not
  (rho `.111`); geometry is suggestive (rho `.562`) but misses BH-FDR (`q=.078`).
  No in-house predictor passed the frozen main-paper gate. Two Gargantua
  science/policy/style/numerical/visual/rebuttal audit waves are complete; the
  four upload files are green and recorded in
  `Paper/Submission/AAAI27/GARGANTUA-FINAL-AUDIT.md`. Next work is the human
  OpenReview upload. The author-side response kit is in `Paper/Rebuttal/`.
  A separately powered multi-seed training study belongs to future work, not
  pre-deadline result fishing.
- **To run anything:** use the venv python `/Users/barathv/.venvs/subliminal-scaling/bin/python`; run probe scripts from the `prompting/` dir. Deps: torch 2.13, transformers 5.14, numpy, scipy, accelerate. Device = Apple MPS or CPU.
- **Golden rules:** iterate small (one slice, prove it, then next); explain to Barath in plain "baby food" English; log every run here + in `experiments.md`; never commit to `main` (work on branch `scaling-followup`); no cherry-picking, controls + uncertainty on every claim.
- **Key gotcha already found:** the number-token probe is tokenizer-dependent (Qwen exposes only 10 single-digit number tokens vs Llama's 1110). See the S1 log entry.

## What this is
Scaling follow-up to the finished `subliminal-learning` repo. We reuse the token-entanglement probe (`prompting/entanglement.py`) and ask: does the effect hold, strengthen, or break as models get bigger and as we change model families? Aim: an arXiv paper + one tight surprising finding for an X post. This is Barath's last novel research project before Google (~Sept 2026); he wants it to be his biggest hit.

## How we work (agreed)
- Iterative vertical slices: build the SMALLEST thing that proves ONE point, look at the result, let it pick the next step. Do all questions one by one, smallest first. Never jump to the end.
- Barath is new to research method: explain each step in plain "baby food" English before running it. Keep chat short; ask questions for major forks only.
- Scientific rules inherited from the original take-home: val for tuning / test once; no cherry-picking (pre-register + report all); every effect needs a control; uncertainty on every headline number; never brush off a finding (one test != confirmation); don't overclaim.

## Roadmap (the progress bar)
1. **Prove the pipeline** — run the existing probe on ONE new model (Qwen3-0.6B) off-Llama.  [DONE — works; found the tokenizer gotcha, see S1 log]
2. **Size ladder** — probe across a size ladder. NOTE (from S1): headline ladder is **Llama 3.x (1B/3B/8B/70B)** because it keeps 0-999 as single tokens so the probe works as-is and extends his own 1B/8B results. Qwen becomes a cross-tokenizer contrast (needs a sequence-level probe variant). Does the effect hold/strengthen/break with size? Answers the core "hold or break" question.  [DONE — behavioral point estimates rise locally through 8B, then 70B-vs-8B change is unresolved]
3. **Geometry scaling test** — the finished take-home already tested geometry at 1B and found it weak. Extend the exact test across sizes: do entangled pairs have high unembedding cosine similarity, does geometry explain behavior, and does explanatory power change with size?  [DONE — 1B->3B jump, 3B->8B plateau, decisive within-release 8B->70B reversal]
4. **Cross-family** — repeat on Llama 3.x + Gemma2/Falcon3. Universal effect, or vendor-specific? Are specific pairs (owl->087) universal or per-model?  [todo]
5. **Training dynamics (parked for the paper)** — Pythia/OLMo intermediate checkpoints: watch the effect emerge during pretraining. Pythia has 154 ckpts/size + a `-deduped` twin for a frequency-confound control.  [parked]
6. **Causal scale test** — random-pair assistant-state block-input patching on
   matched full-BF16 Llama-3.1 8B/70B. [DONE — causal handoff is much earlier at
   70B despite weaker static geometry]

The "one surprising finding -> X post" likely falls out around step 2-3.

## Progress log (newest on top)

- **2026-07-19 — ARXIV AND FUTURE-VENUE ROUTING GUIDE COMPLETE.** Rechecked the
  live AAAI-27 submission and blind-review rules plus the official arXiv
  submission, endorsement, licensing, metadata, and category pages. AAAI-27
  explicitly allows arXiv and non-archival workshops, but forbids submitting
  the same or substantially similar work to another archival conference or
  journal until decision or withdrawal. Recommended sequence: finish AAAI,
  create an arXiv draft and resolve any endorsement, then post a named `cs.CL`
  preprint with `cs.LG` cross-listing after the AAAI package freezes. The AAAI
  PDF stays anonymous and cannot point to arXiv; the arXiv page cannot advertise
  AAAI submission. Added the complete baby-food guide at
  `Paper/Learning/arxiv-and-other-venues-guide.md`. PLAIN ENGLISH: one review
  competition at a time, but a public arXiv timestamp is allowed.

- **2026-07-19 — AI DISCLOSURE MINIMIZED WITHOUT HIDING ITS ROLE.** At Barath's
  request, replaced the two-sentence page-7 disclosure with one neutral
  sentence: `Generative AI tools assisted with implementation and manuscript
  preparation.` The deleted responsibility sentence merely repeated AAAI's
  standing rule and its singular `The author` wording could disclose author
  count during anonymous review. The retained sentence accurately documents
  the two broad roles without naming products, prompts, agents, individual
  operations, or the number of human authors. Recompiled the seven-page PDF;
  no errors or overfull boxes, US Letter and embedded-font properties remain,
  and a fresh page-7 render confirms clean conclusion-to-disclosure-to-reference
  flow. PLAIN ENGLISH: Barath's instinct was correct; this version obeys the
  rule while saying only what the rule needs.

- **2026-07-19 — FINAL SLEEP-PASS FREEZE: GREEN; NO PAPER EDIT JUSTIFIED.**
  Rechecked the live AAAI-27 call, submission instructions, supplement rules,
  and OpenReview invitation schema; all deadlines, fields, and 10/5/10/50 MB
  caps match the roadmap. Downloaded the current official Author Kit; local
  `aaai2027.sty` and `.bst` match byte-for-byte. Recompiled all three PDFs and
  compared the builds to upload copies: identical text and rendered pixels.
  Inspected all 12 pages at 144 dpi; no clipping, overlap, unreadable object,
  page-break trick, or formatting defect. All fonts are embedded, all pages are
  US Letter, and the source has no forbidden layout command. Recounted 22/22
  citations and bibliography entries; all 22 URLs returned HTTP 200. A fresh
  ZIP extraction passed 49/49 manifest hashes and identity/secret scans. Every
  included deterministic analysis and all three figures reproduced; expected
  differences are only a sanitized protocol path and the disclosed omitted
  MPS layerwise control, while the included 8B/70B branches match exactly.
  Added a one-sitting July 20 checklist to the top of the submission roadmap.
  PLAIN ENGLISH: no hidden defect was found, so touching the paper would now be
  worse than submitting it. The files are frozen; Barath reads, fills truthful
  identity/policy fields, uploads all four files, downloads them back, and
  saves the OpenReview record.

- **2026-07-19 — SOLO AAAI REALITY CHECK COMPLETE; SUBMISSION VERDICT UNCHANGED.**
  Audited all 43 official AAAI-26 Main Technical Track proceedings issues:
  4,149 article records, 26 with exactly one listed author (0.627%, about one
  solo paper per 160 accepted papers). The official proceedings include Omar
  Claflin's sole-authored neural-representation paper with affiliation
  `Independent`, proving that neither a team nor an institutional affiliation
  is required for an accepted main-track paper. AAAI's official opening
  material reports about a 17.5% Main Track acceptance rate overall. These
  facts do not yield a solo acceptance probability because AAAI does not report
  the number of solo submissions. During double-blind review, solo status is
  invisible and gives no score bonus. The honest readiness label is
  `submission-ready and genuinely competitive, but not acceptance-predictable`.
  Added `Paper/Learning/solo-aaai-reality-check.md`. PLAIN ENGLISH: solo AAAI is
  rare and real; our package is strong enough to submit, not strong enough for
  any honest person to guarantee acceptance. Freeze the science, keep the
  minimal policy-required AI note, and complete the human Main Track upload.

- **2026-07-19 — AI-ASSISTANCE MINI-GARGANTUA + TRACK CORRECTION COMPLETE.**
  Live AAAI-27 rules were rechecked against the Main Technical Track call,
  submission instructions, and standing publication policy. AAAI permits
  judicious generative-AI use but requires its role to be documented in the
  manuscript; no rule requires tool names, prompts, or a task inventory. A
  systematic practice sample extracted 29 accepted AAAI-26 final papers across
  Machine Learning V, NLP II, and Philosophy/Ethics; none displayed an
  authoring-AI disclosure, but this cannot justify omission because AI use was
  unknown and AAAI-26 had a different, more restrictive rule. Replaced the long
  task list with: `Generative AI tools supported research workflow automation,
  implementation, and manuscript preparation. The author remains responsible
  for all reported content.` Main remains 7 pages and visually clean; current
  size/hash are 976,325 bytes and
  `90eef375877e95c59712a7800aea2fdd78a3d96567d563e01f0827a317e24c07`.
  Also caught that the user's logged-in page was the separate AISI venue; this
  technical mechanism paper belongs in the `AAAI 2027 Conference` Main
  Technical Track. PLAIN ENGLISH: keep a tiny truthful disclosure, not the long
  confession and not a risky omission; use the main conference page, not AISI.

- **2026-07-18 — GARGANTUA REBUTTAL SHIELD COMPLETE; FINAL GATE GREEN.** A
  second adversarial wave simulated attacks on novelty, correctness,
  significance, evidence, clarity, and reproducibility, then the main agent
  independently arbitrated every suggestion. Valid repairs: the closest-work
  table now credits prior intervention/size/position evidence precisely; the
  external null is stated directly; 1,110-number observational analyses are no
  longer conflated with the outcome-blind 256-number causal subset; a
  deterministic matched-256 sensitivity preserves the resolved geometry
  decrease `-0.06461 [-0.12434,-0.00334]` and specificity decrease `-0.08529
  [-0.14318,-0.03029]` while leaving readout change unresolved `+0.03354
  [-0.00524,+0.07145]`; and the raw-target-logit causal increase is resolved at
  `+0.27437 [+0.25934,+0.28936]`, 18/18 animals, all 20,000 draws positive.
  Estimator provenance now says recorded, not independently timestamped,
  amendment; the off-manifold hybrid-computation caveat is explicit. False
  duplicate-text and same-stimulus alarms were rejected. `Paper/Rebuttal/`
  contains mock reviews, attack matrix, evidence index, response bank, October
  checklist, and open decisions. Final main/supplement/checklist are 7/3/2
  pages. The anonymous archive is 49,495,215 bytes, has 49 manifest checks, and
  verifies after fresh extraction. No new GPU run is scientifically justified.
  PLAIN ENGLISH: skeptical-reviewer attacks improved the paper, all computable
  defenses are now archived, and the remaining job is the human upload and
  personal responsibility check, not another experiment.
- **2026-07-18 — GARGANTUA FINAL AUDIT COMPLETE; UPLOAD PACKAGE GREEN.** Three
  independent hostile reviews covered science/statistics, accepted-paper story
  and prose, and live AAAI policy/submission compliance. Fatal defects: zero;
  new experiment required: no. Fresh reruns of behavior/geometry, size, Qwen,
  layerwise, headline causal, external causal, and external-transfer analyses
  match submitted summaries. All 22 citations map one-to-one to 22 bibliography
  entries and were checked against primary records. The audit caught a real
  policy error in the prior clean-reference-page change: the official author kit
  forbids `\clearpage` and says references must flow naturally. Removed it,
  restored the required `caption` package, and removed custom float/table-spacing
  overrides. AAAI policy also requires any AI role to be documented, so an
  accurate anonymous disclosure now appears before References. Tightened two
  causal phrases to the measured estimands and labeled the failed S8 gate's main
  mention as transparency-only. Final main/checklist/supplement are **7/2/3
  pages**; code/data is **49,488,318 bytes**. Every PDF page was rendered and
  inspected; the ZIP decompresses, its 46 manifest checks pass, and identity
  scans are clean. PLAIN ENGLISH: the paper is not guaranteed acceptance, but
  there is no known broken claim, missing forced experiment, policy blocker, or
  upload-file defect. Stop polishing and use the roadmap.
- **2026-07-18 — CLEAN REFERENCE-PAGE TRANSITION (LATER REVERSED).** A forced page
  break now keeps the `References` heading and all 22 entries together on page
  7 instead of stranding the heading and first entry at the foot of page 6.
  Strict gate passed: total length remains **7 pages**, no text or citations
  were removed, official typography was not shrunk, and page 6 now ends cleanly
  after the conclusion with intentional whitespace. Both final pages were
  rendered at 144 dpi and inspected; page 7 has comfortable remaining space.
  PLAIN ENGLISH: this was a real polish improvement with zero scientific or
  submission-cost tradeoff, so it was worth doing.
- **2026-07-18 — LIVE OPENREVIEW ADMIN AUDIT + BABY-FOOD ROADMAP COMPLETE.**
  Read the active AAAI-27 OpenReview invitation schema, not just the conference
  prose. Confirmed this is one submission record with three gates: abstract
  July 21, main/checklist July 28, supplement/code July 31, all 23:59 UTC-12.
  The live form adds mandatory TL;DR, institution-country, reciprocal-reviewer
  declaration, and policy fields. Topics and reviewer nomination freeze July
  21; authors/paper can change through July 28; only supplements can change
  through July 31. Created
  `Paper/Submission/AAAI27/START-HERE-SUBMISSION-ROADMAP.md` with exact
  ready-to-paste fields, New York time conversion, internal one-day-early
  targets, account/profile instructions, upload mapping and caps, verification
  steps, and post-review dates. Added `tldr.txt` and narrowed secondaries to the
  three best reviewer-matching topics. Main/checklist/supplement are
  `.98/.10/.20` MB; code/data is `49.49` MB under its 50 MB cap. Biggest live
  risk is administrative: public-email OpenReview profiles can take up to two
  weeks to moderate, so profile activation is the only urgent action. Since all
  artifacts are ready, the safest path is to upload all four in the same
  session as abstract registration rather than wait for later deadlines. PLAIN
  ENGLISH: the paper is green; create or recover the OpenReview account today,
  then follow the one checklist without improvising.
- **2026-07-18 — AAAI PAGE-BUDGET SURGERY COMPLETE.** Used the available page
  budget for reviewer-facing evidence rather than decorative density. The main
  paper now includes a full-width nearest-prior-work table that states what six
  adjacent lines of work establish and the exact gap addressed here. Added
  primary citations on tokenization-space signal, causal tokenization bias,
  label-length bias, and activation-patching interpretability; the text now
  explicitly says autoregressive chain-rule scoring is established rather than
  claimed as new. Added the preregistered external-validation values to the main
  discussion (geometry rho `.562`, BH `q=.078`; causal rho `.111`, `q=.687`;
  released steering rho `.768`) and a caveat that full-state swapping proves
  intervention-specific sufficiency, not necessity or a unique circuit. Final
  main is **7 pages total**, with references beginning on page 6: **3 figures,
  5 tables, and 22 cited works**. The official anonymous `aaai2027` style hash
  still matches the downloaded AAAI-27 author kit. All pages were rendered and
  inspected; there is no clipping, overlap, broken citation, or decorative
  equation. PLAIN ENGLISH: a reviewer can now see, in one table, exactly what
  the closest papers did and why this experiment is still different.
- **2026-07-18 — FINAL PARALLEL RED TEAM + UPLOAD-GATE REPAIR COMPLETE.** Three independent read-only reviewers separately audited statistical/causal validity, novelty/story, and AAAI submission/anonymity. No fatal scientific, formatting, or solo-author problem was found. Two real gaps were repaired: the main discussion now discloses the negative S8 transfer-prediction gate instead of leaving it only in the supplement, and the packaged causal analyzer now automatically emits every stated raw-logit, all-17-wrong-concept, 128 leave-one-pair, and preregistered depth-rise sensitivity. All values match independent recomputation exactly. The preregistered `peaks_clean.json` steering source is now archived at its immutable revision and matches all 16 CSV convenience values. The anonymous archive was reduced from 62 MB to **49.49 MB** without removing main headline or causal arrays; three large non-headline/device-control copies were omitted with explicit disclosure while their summaries, metadata, and collectors remain. All documented analyzers rerun exactly from the size-bounded package; 46 manifest entries and the fresh-extraction ZIP verify. Main/supplement/checklist remain 6/4/2 pages with embedded fonts and clean visual QA. PLAIN ENGLISH: solo authorship is allowed, the science survived hostile review, the inconvenient result is visible, every claimed robustness check is now generated by code, and the actual upload file fits.
- **2026-07-18 — S8 OUTCOME-BLINDED EXTERNAL TRANSFER CHECK COMPLETE; MAIN-PAPER GATE FAILED HONESTLY.** The newest primary literature already released a 16-animal Llama-3.1-8B student-transfer benchmark, so a redundant small fine-tuning study was rejected. Before opening its exact per-animal CSV, committed `preregistration-s8-external-transfer.md` as commit `a107d0b`, fixing all animals, three in-house predictors, statistics, controls, and the main-paper gate. Locally recollected all 1,110-number geometry and 33-state readout results plus the full five-depth/128-pair causal patch for the external animals. Instruments reproduced: geometry mean r `.173` (13/16 BH), readout AUC `.274`, causal AUC `.255 [.242,.268]`; endpoint, identity, duplicate, permutation, wrong-animal, conditioning, and degeneracy checks passed. External student-transfer Spearman rho: geometry `.562 [.099,.847]`, `p=.0259`, BH `q=.0778`; readout `.316 [-.287,.786]`; causal `.111 [-.444,.624]`; released steering benchmark `.768 [.367,.935]`, `p=.0010`. Steering-minus-causal rho difference `.657 [.069,1.207]`. No in-house predictor passed the fixed gate, so the main paper was not expanded with a favorable subset. Result preserved in the four-page technical supplement and `Paper/Research/external-transfer-validation.md`. No paid compute used. PLAIN ENGLISH: our causal timing effect is real inside the frozen model, but it does not predict which animal trait trains into a student; the trait-aligned steering vector does much better.
- **2026-07-18 — PEAK-PAPER ADVERSARIAL PASS COMPLETE; PACKAGE REBUILT AND VERIFIED.** Benchmarked recent official AAAI award/accepted papers and ran three independent skeptical reviews. The paper moved from an estimated 6/10 weak accept to a reviewer-target 7/10 accept after bounded-claim, literature, robustness, and reproducibility fixes. New title: **“Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds.”** The closest cross-scale probe/patch work, *Where's the Plan?*, is now cited and explicitly differentiated. Fixed three critical claim traps: “earlier” now means relative depth, the Qwen result is a tokenizer/model-family measurement boundary rather than tokenizer causality, and the S5 intervention is described as prospectively designed with a validation-stage estimator repair rather than unchanged preregistration. Added exact prompts, coefficient/AUC definitions, block mappings, a claim-to-test table, all-depth coefficients, same-eight-block sensitivity (**.948 vs .592**), raw-logit sensitivity (**18/18, mean +.274**), leave-one-pair range (**+.2850 to +.2876**), all 17 wrong-concept shifts, and the **14/18 joint geometry-down/causal-up** result. Final main is **6 pages**, supplement **3**, checklist **2**, all visually inspected with embedded fonts and no warnings. Repaired the anonymous package's figure paths and a mislabeled geometry reproduction command, reran every analyzer and figure command from the archive layout, expanded the SHA-256 manifest to **36 code/data/figure files**, rebuilt the 51 MB ZIP, and verified it after fresh extraction. Exact Hugging Face commit hashes were not historically recorded and are disclosed as a limitation. Nothing is submitted externally yet.
- **2026-07-18 — AAAI-27 ANONYMOUS REVIEW PACKAGE COMPLETE, NOT YET SUBMITTED.** Official next realistic archival target is AAAI-27 Main Technical Track: abstract July 21, paper July 28, supplement/code July 31, all 11:59 PM UTC-12. Built `Paper/Submission/AAAI27/` from the official 2027 author kit: single-file anonymous main source and 5-page compiled PDF, 3-page technical supplement, separate 2-page reproducibility checklist, exact abstract/topics/deadlines, prose audit, baby-food conference guide, and a 50 MB anonymous code/data ZIP. The ZIP includes all raw headline arrays plus MPS/CUDA device controls, analysis code, deterministic rerun commands, and verified SHA-256 manifest; no author name, home path, API key, or private URL remains. All PDFs are US Letter PDF 1.5 with embedded Type 1 fonts; visual page-by-page QA found no clipping, overlap, or illegible figures. Main story now leads with the causal handoff, includes a compact matched-results table, and treats the Qwen result as a width-confound warning. AAAI permits judicious AI assistance but the authors remain responsible; no detector gaming or fake authorship claim was used. Final external submission is blocked only on legitimate human fields: complete author list/OpenReview profiles, affiliation/email, conflicts, non-concurrent-submission confirmation, reviewer/attendance obligations, license choice, and explicit title/abstract approval. Nothing has been uploaded or submitted under Barath's identity.
- **2026-07-18 — S5 COMPLETE: CAUSAL DONOR CONTROL ARRIVES MUCH EARLIER AT 70B.** The 1B smoke caught an MPS grad-mode duplicate-forward issue; adding no-grad fixed it. The full local 8B gate then exposed a deeper estimator problem: the original `patched-recipient` on `donor-recipient` slope shared the recipient term and produced an invalid positive permutation baseline. Before any CUDA/70B S5 output, froze the S5B amendment: regress patched animal contrast on clean donor and clean recipient simultaneously; donor coefficient is primary, recipient coefficient secondary, original estimator remains labeled invalid. Corrected local gate passed. One fail-safe Vast run (instance 45267205, 4×RTX A6000, full BF16) completed exact CUDA8B then 70B and was destroyed; active list `[]`. Settled credit `$1.60777967921 -> $0.79508485761`, observed cost `$0.81269482160`, no billing. Mean donor coefficients: **8B 0.001/0.038/0.592/0.780/0.974**, **70B 0.007/0.773/0.938/0.948/0.984** across the five depths. Corrected causal AUC **8B 0.25389 [0.24147,0.26612]**, **70B 0.53970 [0.52691,0.55158]**; paired **+0.28581 [+0.27162,+0.29991]**, all **18/18** animals increase. Decision: stronger/earlier causal transmission at 70B. Permuted donor and wrong-animal coefficients remain small; both directions positive; duplicate and identity errors exactly 0; max condition numbers 1.177/1.242; zero degenerate bootstrap cells. Raw hashes: 8B `a9a8acfb...a19f4`, 70B `96acfd09...72c0`, summary `af7c4a64...f829d`. Paper package updated and LaTeX compiles to 11 pages with the causal figure. PLAIN ENGLISH: halfway through 8B, the original prompt still controls the answer; halfway through 70B, the transplanted prompt already controls it almost completely, even though 70B's simple token geometry is weaker.
- **2026-07-18 — FRONTIER AUDIT + S5 CAUSAL PATH TEST PRE-REGISTERED.** Built the structured `Paper/` workspace: modular 9-page LaTeX manuscript, compiled PDF, zero-background guide, reproducibility map, claim-by-claim novelty matrix, and nine primary-source paper cards. The audit materially narrowed novelty: Schrodi et al. already score digit-split numbers autoregressively and inspect layers; Data2Behavior already uses frozen layerwise representations and causal injection; a positional-bias paper already combines cross-model geometry, layerwise probing, steering, and representation/behavior dissociation; Madl already causally localizes a vocabulary-geometry transfer channel. Defensible novelty is the width-controlled bidirectional length-confound diagnosis plus the matched full-BF16 Llama-3.1 8B/70B frozen-channel geometry-vs-depth dissociation. Highest-value follow-up is frozen in `Paper/Supplement/preregistration-s5.md`: outcome-independent random-pair assistant-state block-input patching at relative depths .25/.50/.75/.90/.97, OLS causal-transmission slopes, crossed animal/pair bootstrap, donor/wrong-animal/direction/numerical controls, local 1B smoke and 8B gate before any 70B spend. No paid instance launched.
- **2026-07-18 — S4B FULL 8B/70B DEPTH TRACE COMPLETE: STATIC-GEOMETRY REVERSAL IS NOT A WHOLE-COMPUTATION COLLAPSE.** The tuned-logit-lens collector passed exact final-endpoint regression, local full 8B, same-environment CUDA8B, ignored 3-number 70B smoke, and full BF16 CUDA70B (1,110 numbers × 18 animals × 33/81 states). Pre-registered assistant AUC: **8B-CUDA 0.259 [0.236,0.282]**, **70B 0.269 [0.235,0.300]**; paired 70B-8B **+0.010 [-0.028,+0.045]**, 13/18 animals rise. Half-final depth **0.719 -> 0.738**. Decision = **similar/unresolved curves**, not lower 70B AUC. Animal-specific contrast AUC 0.278->0.298; paired +0.020 [-0.015,+0.055]. System-number mean-position AUC stays weak, 0.051 [0.008,0.097] -> 0.036 [-0.005,0.072], paired -0.015 [-0.074,+0.042]. MPS8B-vs-CUDA8B assistant AUC delta -0.00001 [-0.00113,+0.00111], so device drift is negligible. PLAIN ENGLISH: 70B loses the simple static token-geometry explanation, but builds the answer through a very similar late contextual trajectory. This localizes the dissociation toward the output-token geometry story; it does not prove an identical causal circuit. Raw hashes: CUDA8B `44674a5a...f658ad`, 70B `825a8d13...6758e`; summary `c7ccfa05...fff98`.
- **2026-07-18 — S4A MULTI-TOKEN QWEN COMPLETE: SEQUENCE COMPOSITION DOES NOT RESCUE THE ATOMIC CHANNEL.** Full 1,110-string probes ran on Qwen3-0.6B and official Qwen3-1.7B after an exact Llama atomic regression. Width-3 full-sequence primary: **0.6B -0.050 [-0.124,+0.027]** and **1.7B -0.048 [-0.088,-0.008]**; 1.7B-0.6B +0.002 [-0.078,+0.087]. First-token proxies are small positive (+0.024/+0.021), so full-minus-first is negative (-0.074 [-0.102,-0.042] at 0.6B; -0.069 [-0.151,+0.007] at 1.7B). Crucial measurement result: across mixed widths, sum-logp is negative (-0.098/-0.162), naive mean-per-target-token becomes strongly positive (+0.097/+0.233), but within-width standardization returns near zero/negative (-0.049/-0.032). PLAIN ENGLISH: averaging by digit count manufactures a positive-looking correlation from sequence length; after controlling width, it is gone. Atomic tokenization is part of the measurable channel, not a cosmetic implementation detail. Summary hash `a5284a94...46eebb`.
- **2026-07-18 — S4B CLOUD SAFETY/COST FINAL.** Setup-only instance 45261127 omitted one uploaded dependency, produced no model/scientific output, and was destroyed; cost settled at $0.05079. Corrected bundle passed isolated import closure. Instance 45261293 ran on 4×RTX A6000 at $1.60778/hr from 21:00:19Z to 21:34:24Z, pulled/validated all outputs, and was destroyed. Credit after cleanup **$1.72882825921**, no billing method, independent active-instance query `[]`. Combined corrected S4B rental cost about $0.94260; total S4 setup+success about $0.99339.
- **2026-07-18 — S4 TOKENIZATION + DEPTH FOLLOW-UPS PRE-REGISTERED.** After the 70B geometry reversal, froze two mechanism tests in `scaling/preregistration_s4.md` before collecting results. S4A repairs the probe for digit-splitting tokenizers by scoring complete autoregressive number sequences, with width-3 primary, first-token rescue comparison, and mismatch control. S4B uses a tuned logit lens at every layer and both assistant/system-number positions to ask whether 70B moves the channel from static output geometry into late contextual computation. Local validation gates any further paid 70B run. Literature framing updated: token entanglement is a channel/diagnostic, not a necessary general mechanism for training-time subliminal learning.
- **2026-07-18 — S3B FULL 70B DONE: GEOMETRY WEAKENS, BEHAVIORAL CHANGE UNRESOLVED.** User-approved retry used instance 45256931 on the same offer (4x RTX A6000, $1.60778/hr), full BF16, no quantization/offload. Full CUDA 8B calibration, ignored 12-number/2-animal 70B smoke, and full 1,110-number/18-animal 70B all completed; artifacts validated/pulled and instance destroyed (independent active-instance check = zero). **Calibration passes:** primary MPS8B 0.18856 vs CUDA8B 0.18773; paired device delta **-0.00083 [-0.00162,-0.00003]**, tiny relative to the model-size change. **Pre-registered primary:** CUDA8B **0.188 [0.146,0.230]** -> 70B **0.108 [0.052,0.154]**; paired 70B-8B **-0.080 [-0.127,-0.035]**, only 4/18 animals increase. This satisfies the pre-written **reversal/weakening** rule. **Required specificity control:** 0.155 [0.126,0.186] -> 0.046 [-0.015,0.093]; exploratory paired change **-0.109 [-0.164,-0.060]**. **Behavioral entanglement:** median r 0.117 [0.057,0.155] -> 0.088 [0.054,0.123]; paired mean change **-0.024 [-0.056,+0.009]**, so behavioral change is unresolved, not a decline claim. Fixed secondaries: raw forward 0.224->0.129; centered reverse -0.023->-0.076; first-target raw reverse 0.165->0.127; raw remains above centered at 70B by +0.184 [0.117,0.243]. Famous raw-cos ranks at 70B: owl->087 602, eagle->747 404, sea turtle->321 335 /1110. **Plain English:** the entanglement behavior still exists at 70B, but simple output-token geometry explains substantially less of it; a larger model may implement the association through a more distributed/nonlocal mechanism. This is evidence against monotonic strengthening, not proof of a universal size law. Geometry summary SHA-256 `87e3d1f796b2ee180cc8f182bc3491aaf3dea96e29c51f0d40e389ec93288c46`; exact deterministic rerun matched.
- **2026-07-18 — CLOUD COST/SAFETY FINAL.** Successful instance ran **30m57s** from create to destroy. At last settled query, total credit moved **$3.70523041311 -> $2.72221427921**, so both attempts together used **$0.98301613390** and left **$2.72**. Retry launch-to-final snapshot used about **$0.82307**; Vast delayed part of attempt-1 billing, so its settled launch-to-launch cost was about **$0.15995**, not the earlier immediate $0.019/$0.072 snapshots. No billing method is attached. Both instance IDs 45256399 and 45256931 were destroyed; zero active instances.
- **2026-07-18 — FIRST PAID ATTEMPT ABORTED SAFELY DURING SETUP.** Vast instance 45256399 (offer 20805216, 4x RTX A6000) booted and SSH passed, but the pinned local Python-3.12-only scientific packages (`numpy==2.5.1`, `scipy==1.18.0`) could not install in the image's Python 3.11. The fail-safe immediately destroyed the instance; active-instance check returned empty. Vast billing settled later (see final cost entry above). No model was downloaded, no probe ran, and no scientific result was inspected. Corrected the remote-only pins to Python-3.11-compatible `numpy==2.4.2 scipy==1.17.0`; verified CPython-3.11 Linux wheels before the separately user-approved retry.
- **2026-07-18 — S3B 70B PRE-REGISTERED + AUTHORIZED.** Barath delegated end-to-end execution using existing Vast.ai promotional credit (~$3.71). One-instance budget rules fixed before launch: on-demand offer <=$1.70/hr including 220 GB storage; >=180 GB total GPU RAM, >=64 GB system RAM, reliable/direct SSH, CUDA>=12.4, fast/cheap ingress; local hard wall 85 min plus remote 90-min kill watchdog; abort if remaining credit approaches $0.90; always pull artifacts/logs and DESTROY (never merely stop) in `finally`. Environment image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (Transformers 5.14 requires torch>=2.4, so torch 2.6 is compatible); install exact transformers 5.14.1, accelerate 1.14.0, numpy 2.5.1, scipy 1.18.0. First run full Llama-3.1-8B calibration on CUDA; compare MPS8B->CUDA8B as a device control. Then load verified full-BF16 `unsloth/Meta-Llama-3.1-70B-Instruct`, run 12-number/2-animal sharding smoke (inferential output ignored), then unchanged full 1,110-number/18-animal probe. PRIMARY 70B estimand unchanged: mean across animals of raw mean-subtoken unembedding cosine vs reverse log probability. Primary contrast = paired CUDA 70B minus CUDA 8B with seed-0 100k bootstrap CI. Required control = matched-minus-mismatched animal geometry and paired 70B-8B change. Fixed secondary = behavioral bidirectional r, raw forward, centered cosine, first-target-token raw cosine, BH-FDR, paper-pair ranks. Decision: positive CI=continued growth; CI spanning zero=plateau; negative CI=reversal. One 8B/70B release pair does not establish a universal scaling law. No cherry-picking; all metrics reported.
- **2026-07-18 — S3A LOCAL LADDER DONE: GEOMETRY JUMPS THEN PLATEAUS; SPECIFICITY STRENGTHENS.** Same-device MPS full artifacts now exist for 1B/3B/8B. Pre-registered primary raw-cosine->reverse mean r: **0.103 [0.054,0.151] -> 0.198 [0.148,0.252] -> 0.189 [0.147,0.231]**; BH-FDR 12/18->17/18->17/18. Paired changes: 3B-1B **+0.095 [0.042,0.153]** (resolved); 8B-3B **-0.010 [-0.061,0.041]** (plateau/uncertain). Required matched-animal minus other-animal control: **+0.100 [0.061,0.135] -> +0.128 [0.094,0.159] -> +0.155 [0.125,0.186]**. Exploratory control scaling: 3B-1B +0.028 [-0.010,0.063], 8B-3B **+0.027 [0.003,0.052]**. PLAIN ENGLISH: overall geometry becomes about twice as predictive by 3B and then stops growing, but the animal-correct part of the geometric signal keeps getting cleaner. SECOND FINDING: centered-cosine reverse r **0.123 -> 0.096 -> -0.025**; raw-minus-centered is decisively +0.213 [0.140,0.286] at 8B, so the original 1B suggestion that centering helps does NOT generalize. Paper-pair geometry ranks (1B/3B/8B): owl->087 **31/89/803**, eagle->747 **244/330/431**, sea turtle->321 **923/831/1031**; vivid pairs do not explain the aggregate mechanism. CAVEAT: 1B/3B are Llama 3.2 while 8B is Llama 3.1, so the 3B->8B contrast mixes size and release; 8B->70B will use Llama 3.1 for a cleaner within-release comparison. Full summary `geometry_scaling_summary.json`.
- **2026-07-18 — SAME-DEVICE BEHAVIOR CONTROL.** Regenerated the behavioral ladder from the new MPS artifacts rather than mixing old CPU 1B/8B with MPS 3B. Qualitative result unchanged: raw significant 10/18->12/18->13/18; BH-FDR **9->11->13**; median r **0.068->0.088->0.117**. Paired 3B-1B +0.028 with interval touching zero; 8B-3B +0.013 with interval spanning zero. Artifact `size_ladder_mps_summary.json`. Use this for future same-device comparisons; retain the older summary as history.
- **2026-07-18 — 70B PIPELINE HARDENING.** `collect_full_probe.py` now saves complete behavior/geometry sufficient statistics atomically and resumes; raw artifacts are only 5.2/7.2/8.6 MB for 1B/3B/8B. `utils.load_model` now chooses `device_map="auto"` when multiple CUDA GPUs are visible and accepts explicit `--device-map`/`--dtype`; pure selection tests pass (1 CUDA->cuda, 4 CUDA->auto, no accelerator->cpu). Planned 70B invocation must fix `--device-map auto --dtype bfloat16`. No Vast.ai instance launched and no credit spent.
- **2026-07-18 — S3A 3B: GEOMETRY SIGNAL NEARLY DOUBLES; 8B NEEDED.** Collector passed same-device behavioral regression (max Pearson-r delta 1.34e-08). Primary raw-cosine->reverse mean r rises **0.103 [0.054,0.151] at 1B -> 0.198 [0.148,0.252] at 3B**; paired delta **+0.095 [0.042,0.153]**; BH-FDR 12/18->17/18. Approximate explained variance rises from ~1.1% to ~3.9% (descriptive r^2, not causal). The effect remains animal-specific: matched-minus-other-animal geometry **+0.128 [0.094,0.159]** at 3B. IMPORTANT SECONDARY: centered cosine drops to r 0.096, reversing its apparent 1B advantage; do not call centering generally better. Named pair geometry ranks still reshuffle (owl 89, eagle 330, sea turtle 831). PLAIN ENGLISH: at 3B, the model's physical token geometry predicts much more of its behavior, but 8B must say whether this continues or was a one-step jump. Next = local 8B fresh same-device artifact; no metric changes.
- **2026-07-18 — S3A 1B VALIDATION + RESULT PASS.** The resumable collector reproduced fresh original-code MPS runs: max geometry-summary delta 2.08e-08; max behavioral Pearson-r delta 1.26e-08; identical top-number lists/pair ranks. Important reproducibility note: old July 2 files were CPU-generated and differ slightly from MPS, so all S3A scale points will use fresh, same-device full artifacts. Primary 1B geometry->reverse behavior mean r **0.103 [0.054, 0.151]**, 12/18 BH-FDR. New mismatch control is strong: matched r **0.103** vs other-animal geometry r **0.003**, delta **+0.100 [0.061, 0.135]**, 16/18 animals. PLAIN ENGLISH: geometry explains only ~1% of the number-to-animal variation, but the tiny part it explains is animal-specific, not generic number weirdness. Raw-cosine geometry ranks: owl->087 31, eagle->747 244, sea turtle->321 923 of 1110. Centered cosine reverse r 0.123; first-target-token robustness r 0.099. Next = unchanged 3B run.
- **2026-07-18 — S3A CHECKPOINT/RESUME SMOKE PASS.** New `prompting/collect_full_probe.py` passed an intentional interruption test on 1B with 2 animals/12 numbers: stopped after 3 reverse rows, atomically saved, resumed from row 4, completed behavior + fp32 selected unembedding data, and then correctly no-op'd on a third launch. Assertions confirmed no missing values and correct array shapes. The tiny geometry result is ignored by design. Next = full 1B regression against the original `entanglement_1b.json` and `metrics_1b.json`; do not start 3B/8B until that agrees.
- **2026-07-18 — S3A PRE-REGISTERED: GEOMETRY SCALING + 70B-SAFE COLLECTOR.** Correction to the initial roadmap wording: `prompting/geometry_metrics.py` and `metrics_1b.json` show that Barath's completed take-home DID test the unembedding-geometry explanation at 1B; it mostly failed there. The open contribution is scaling that mechanism test across 1B/3B/8B and eventually 70B. Primary metric fixed before new results: per animal, Pearson correlation across all 1,110 number tokens between **raw unembedding cosine** (mean of the animal's subtoken rows, matching the original analysis) and **number->animal log probability**; summarize the mean across all 18 animals with a seed-0 100,000-resample bootstrap 95% CI. Primary scaling contrast = paired within-animal change 1B->3B and 3B->8B; three points are not a scaling law. Required animal-specificity control: matched animal geometry versus the mean of all 17 mismatched animal geometries predicting that animal's behavior, with a paired bootstrap CI for matched-minus-mismatched. Report all pre-fixed secondary metrics regardless of outcome: animal->number direction; centered cosine; raw/centered dot product; first behavioral target-token geometry versus mean-subtoken geometry; paper-pair ranks. Build a resumable raw collector that saves X/Y log probabilities, number IDs/strings, animal token IDs, selected full-precision unembedding rows, vocabulary mean, metadata, and checkpoints. First validate on a tiny interrupted/resumed smoke, then reproduce the existing 1B summary before running 3B/8B. No 70B spend yet.
- **2026-07-18 — S2A DONE: CLEAN BUT NOT YET CONCLUSIVE MONOTONIC SIGNAL.** Full `unsloth/Llama-3.2-3B-Instruct` probe completed on MPS in 169.7 s (1,110 number tokens, all 18 animals; artifact `prompting/results/entanglement_llama32_3b.json`). The raw significant-positive count sits between the existing points: **10/18 (1B) -> 12/18 (3B) -> 13/18 (8B)**; BH-FDR counts are **10 -> 11 -> 13**. Median Pearson r also rises **0.069 -> 0.088 -> 0.116**, with across-animal bootstrap 95% CIs **[0.025, 0.111]**, **[0.045, 0.129]**, **[0.055, 0.153]**. Exploratory paired bootstrap: 3B-1B mean within-animal delta **+0.029 [0.001, 0.059]** (12/18 increase); 8B-3B **+0.012 [-0.029, 0.057]** (11/18 increase). PLAIN ENGLISH: all point estimates go up with size, and correction for 18 tests does not erase the pattern, but the 3B-to-8B step is too uncertain to call a scaling law. The named pairs reshuffle again: owl->087 ranks 1/129/684, eagle->747 ranks 3/1/9, sea turtle->321 ranks 563/1065/288 across 1B/3B/8B. Reproducible corrected summary: `prompting/results/size_ladder_summary.json`, generated by `scaling/analyze_size_ladder.py`. Next fork: 70B full precision needs rented multi-GPU-class compute; otherwise Step 3 geometry can begin locally.
- **2026-07-18 — S2A PRE-REGISTERED + SMOKE PASS (3B).** Barath approved the Llama headline ladder via the Claude->Codex handoff. First missing point = `unsloth/Llama-3.2-3B-Instruct`, using the unchanged full protocol (all 1-3 digit single number tokens, all 18 pre-registered animals, seed 0). Primary per-model summary fixed before looking: count of positive Pearson correlations with uncorrected p<0.05 (for direct comparability to existing 1B/8B); also report every animal, median/range of r, 95% bootstrap CI for median r across animals, and BH-FDR q<0.05 count as a multiplicity control. This is one model point, not evidence of a scaling law. Added Apple MPS selection to `prompting/utils.py`; capped 20-number/one-animal smoke passed and its inferential output is ignored. Full fixed run next.
- **2026-07-18 — S1 DONE + KEY FINDING (tokenization).** Ran the probe on Qwen3-0.6B (`prompting/results/entanglement_smoke_qwen06b.json`). Pipeline works off-Llama. BUT Qwen exposes only **10 single-token numbers (digits 0-9)** vs Llama's 1110 (0-999) — Qwen splits multi-digit numbers into per-digit tokens. So "087" isn't even one token in Qwen, and the single-token probe has ~no statistical power there (0/18 significant, n=10, all top-nums are single digits). IMPLICATION: token entanglement as defined is **tokenizer-dependent**; you cannot naively run the same probe across families. Decision: headline size ladder = **Llama 3.x** (probe works as-is, comparable, extends prior 1B/8B). Keep **Qwen as a cross-tokenizer contrast** — "does entanglement survive when numbers are multi-token?" is itself a paper-worthy question needing a sequence-level probe (sum log-prob over digit sub-tokens). This is a genuine methodological contribution, caught on day 1 by going small-first.
- **2026-07-18 — SETUP.** Created branch `scaling-followup` off clean/synced `main`. Built venv at `/Users/barathv/.venvs/subliminal-scaling` (python3.12, OUTSIDE Dropbox so it doesn't sync). Installed torch 2.13.0, transformers 5.14.1, accelerate 1.14.0, numpy 2.5.1, scipy 1.18.0. Verified MPS available. Created `scaling/{memory.md,experiments.md,artifact.md}`. Next: Step 1 smoke on Qwen3-0.6B.

## Research findings (July 2026 web scan; not in the repo)
- Field exploded since launch; Anthropic's original entered **Nature (Apr 2026)**. ~a dozen follow-ups.
- **4 competing mechanism theories**: steering-vector distillation (Nanda group, arXiv 2606.00995), divergence tokens (Freiburg, 2509.23886/ICLR26), superposition (Wong blog), token entanglement (Zur/Baulab, 2509 owls). Unsettled = room for us.
- **Novelty verdict:** cross-family *existence* is already shown (do NOT claim "first cross-family"). A clean **size-ladder scaling study is OPEN** — everyone is stuck at ~7B; strongest papers name scaling as future work.
- **The gold nugget:** the owl->087 paper never verified its own geometry explanation (never computed the owl/087 unembedding cosine). A reviewer asked; nobody did. Cheapest, highest-value first result. This is our Step 3.
- **Do NOT assume "strengthens with scale":** 2 papers found the opposite (non-monotonic; width hurts — Miyamoto, Brockers 2605.23645). Measure, don't assume.
- **Confound to avoid:** "subliminal learning is a LoRA artifact" (2606.00831) — vanishes under full fine-tuning. Our no-fine-tuning entanglement angle sidesteps it entirely.
- **His existing 1B->8B result already hints pairs are per-model:** owl->087 rank 1 (1B) -> 684 (8B); eagle->747 survives (rank 3 -> 9); sea turtle->321 never strong. The phenomenon strengthened (10/18 -> 13/18 significant animals) but the specific pairs reshuffled.

## Model picks
- **Headline ladder:** Llama 3.x instruct (1B/3B from Llama 3.2; 8B/70B from Llama 3.1), using `unsloth/*` ungated mirrors and constant vocab 128256. Existing full results: 1B + 8B. Next: 3B; rent compute for 70B only after local points justify it.
- **Cross-tokenizer contrast:** Qwen3. Its multi-digit splitting prevents a naive single-token comparison; requires a pre-registered sequence-level variant.
- **Cross-family:** Gemma2 (2/9/27B) or Falcon3 (1/3/7/10B, fully ungated), after the headline ladder.
- **Training dynamics:** Pythia (`EleutherAI/pythia-{70m..12b}`, revision="stepN") + `-deduped`; OLMo2/3 for a modern replication.
- **Compute:** small models run local on the Mac (MPS). Rent a cheap A100 (~$20-50 total) only if a step proves worth pushing to 32-72B. Load lm_head in full precision for geometry (quant perturbs it).

## Watch-outs (verify hands-on)
- **Digit tokenization differs per model.** Qwen may split multi-digit numbers into several tokens; Llama tends to keep 0-999 as single tokens. This changes what "number tokens" even are — measure per model.
- Keep `lm_head` full precision for the geometry analysis.
- Harness now selects CUDA, then Apple MPS, then CPU; S2A verified MPS execution.
