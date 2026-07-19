# October author-feedback checklist

Use this only if the paper survives Phase 1.

## When reviews arrive

- [ ] Save a private copy of every human review and the labeled AI review.
- [ ] Save the visible score/rating fields and confidence values.
- [ ] Record the exact feedback deadline and the live character/word limit.
- [ ] Do not edit or replace the paper, supplement, or ZIP.
- [ ] Do not answer immediately while emotional.
- [ ] Read the metareview/SPC prompt if one is visible.

## Triage each concern

Label every reviewer point as one of:

- [ ] **Factual mistake:** the review states something contradicted by submitted
      text or evidence.
- [ ] **Interpretation gap:** the evidence is present, but its meaning was not
      clear to the reviewer.
- [ ] **Real limitation:** the reviewer is correct and the paper already bounds
      it, or should bound it in camera-ready form.
- [ ] **New experiment request:** useful future work, but not evidence in the
      frozen submission.
- [ ] **Taste/significance judgment:** cannot be disproved by another statistic;
      answer with the exact contribution and why it matters.

## Priority order under a tight limit

1. Correct factual errors that could change the decision.
2. Resolve soundness or causal-validity misunderstandings.
3. Clarify the exact novelty gap against named prior work.
4. Explain significance and scope.
5. Concede genuine limitations.
6. Answer small presentation requests only if space remains.

## Drafting rules

- [ ] Open with one sentence stating the paper's exact contribution.
- [ ] Group repeated concerns across reviewers instead of repeating answers.
- [ ] Use compact reviewer labels such as `R1`, `R2`, and `R3`.
- [ ] Give exact numbers and submitted page/table/file locations.
- [ ] Say “we agree” when the limitation is real.
- [ ] Never claim the reviewer's concern is stupid, careless, or bad faith.
- [ ] Never claim the readout change is zero or equivalent.
- [ ] Never upgrade a post hoc sensitivity into a preregistered primary.
- [ ] Never claim the S5 amendment has a third-party timestamp.
- [ ] Never claim steering statistically beats geometry.
- [ ] Never claim a new run, figure, or artifact that was not submitted.
- [ ] Never reveal author identity during double-blind review.

## Rebuttal skeleton

```text
We thank the reviewers for the careful evaluation. The central contribution is
[one exact sentence]. We address the decision-relevant concerns below.

1. [Shared soundness concern]. [Direct answer, exact value, submitted pointer.]
2. [Novelty/significance concern]. [Exact closest-work boundary.]
3. [Scope limitation]. We agree that [...]. The submitted evidence establishes
   [...], not [...].

Reviewer-specific factual clarifications:
- R1: [...]
- R2: [...]

We hope these clarifications resolve the factual and interpretive concerns. We
will incorporate appropriate clarity improvements in the camera-ready version
if accepted.
```

## Final response gate

- [ ] Fits the live form's actual limit.
- [ ] Every number matches the frozen PDF/artifact.
- [ ] Every pointer exists in the frozen submission.
- [ ] No new citation is necessary to understand the answer.
- [ ] No author-identifying detail appears.
- [ ] Tone is calm, technical, and direct.
- [ ] A second reader checks it once.
- [ ] Submit early enough to recover from an OpenReview problem.
- [ ] Download or screenshot the submitted response for records.
