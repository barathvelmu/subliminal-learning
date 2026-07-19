# Gargantua rebuttal shield

Last audited: **July 18, 2026**

## Baby-food verdict

This folder is the defense manual for the paper.

The paper itself is the first shield. That matters because AAAI can reject a
paper in Phase 1 without letting the author answer. The written rebuttal exists
only for papers that reach Phase 2.

The current scientific verdict is:

- no known fatal numerical, statistical, citation, anonymity, or file defect;
- no new GPU experiment is required before submission;
- the strongest remaining weaknesses are honest scope limits, not hidden bugs;
- the likely reviewer fight is about **significance and breadth**, not whether
  the central 8B/70B number was computed incorrectly;
- acceptance is never guaranteed.

## What this audit changed

The rebuttal audit found real issues and repaired them before reviewers could:

1. The closest-work table now describes Blank et al. and Madl accurately and
   includes the overlapping positional-bias paper.
2. The negative external result is stated directly rather than quarantined as a
   “transparency-only” result.
3. The paper now distinguishes the 1,110-number full probes from the fixed
   256-number causal subset.
4. A released-array exact-subset sensitivity shows that geometry still weakens
   and readout change remains unresolved on those same 256 numbers.
5. A crossed-bootstrap raw-logit sensitivity shows that the causal scale change
   survives without the target-minus-other-animal contrast.
6. “Timestamped amendment” was replaced with the supportable phrase “recorded
   amendment.” A sanitized S5 protocol/amendment is now inside the anonymous
   artifact, with an explicit statement that timing is author-reported rather
   than third-party verified.
7. The main paper now states that a natural donor vector can still create an
   off-manifold donor-state/recipient-context hybrid.
8. Artifact wording now says exactly what can and cannot be regenerated from
   included raw arrays.

## Read these files in this order

1. `AAAI27-RULES-AND-TIMELINE.md` — what the conference actually allows.
2. `REVIEWER-ATTACK-MATRIX.md` — every major way a reviewer may attack.
3. `EVIDENCE-INDEX.md` — where the answer already lives in the submission.
4. `RESPONSE-BANK.md` — accurate response modules to adapt after reviews arrive.
5. `AUTHOR-FEEDBACK-CHECKLIST.md` — the October response workflow.
6. `MOCK-REVIEWS.md` — the hostile simulations and the main-agent verdict.
7. `OPEN-DECISIONS.md` — the few decisions only the human author can make.

## The one rule that prevents a bad rebuttal

Do not paste the whole response bank into OpenReview.

Wait for the actual reviews. Answer the decisive misunderstandings first. Admit
real limitations. Point to evidence that was already submitted. Never claim a
new experiment, new file, or new result that reviewers cannot inspect.

AAAI-27 has not yet published the rebuttal word or character limit. The final
response must be compressed only after the live form appears.
