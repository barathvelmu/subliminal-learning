# AAAI-27 submission package

## Baby-food summary

This folder is the upload-ready paper package.

- `START-HERE-SUBMISSION-ROADMAP.md` is the first-time-author master checklist
  for the OpenReview account and all three deadline gates.
- `GARGANTUA-FINAL-AUDIT.md` records the final hostile science, policy, style,
  numerical, visual, anonymity, and archive checks.
- `output/pdf/aaai27-main.pdf` is the paper reviewers judge.
- `output/pdf/aaai27-supplement.pdf` is the detailed audit and extra evidence.
- `output/pdf/aaai27-checklist.pdf` is the required reproducibility form.
- `output/aaai27-code-data.zip` is the verified 49,495,215-byte anonymous code
  and saved-data package, 504,785 bytes below the 50 MB upload boundary.
- `abstract.txt` is the exact abstract to paste into OpenReview.
- `submission-metadata.md` tells you which boxes to choose and which personal details are still missing.
- `../../Rebuttal/README.md` is the private author-side defense kit for reading
  reviews and writing the October response. It is not an upload file.

Nothing has been submitted under your identity.

## Why AAAI-27

AAAI-27 is the next realistic archival deadline as of July 18, 2026. The abstract is due July 21, the paper July 28, and supplements July 31, all at 11:59 PM UTC-12. The main track accepts empirical, integrative, and critical AI work and evaluates significance, novelty, soundness, relevance, clarity, responsibility, and reproducibility.

The paper uses the official 2027 anonymous two-column style. The final main PDF
is seven pages total. Technical material stays within page seven; deferred
Tables 4 and 5 appear at the top of page seven, and references then flow
naturally without a forbidden manual page break. The main paper contains three
figures, five tables, and 22 cited works. The final supplement is three pages.
The checklist is separate, as the official instructions require.

## Build

Run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error checklist.tex
```

The official style files are included unchanged. Only PDF files are required for initial review; source files are needed after acceptance.

## Important policy points

- The submission is double blind. Do not add names, affiliations, acknowledgments, personal URLs, or identifying repository links.
- The manuscript documents the role of generative-AI systems, as required by
  AAAI's publication policy, using the minimal wording justified in
  `../../Research/aaai27-ai-assistance-audit.md`. Do not remove or conceal it.
- The abstract must be real by July 21 and must not change substantially afterward.
- The work cannot be under review at another archival conference or journal.
- All authors must be fixed by July 28 and cannot be added later.
- Reviewers are not required to read the supplement, so every essential claim is already in the main paper.
- The author remains responsible for all AI-assisted work and for verifying
  every claim, reference, result, figure, and artifact.

## Final human gate

Before opening OpenReview, the author must supply the items listed under
`Author information still required` in `submission-metadata.md`, read the main
PDF, and explicitly approve the frozen title and abstract. A public code license
is useful but optional for the review submission.
