# AAAI-27 submission package

## Baby-food summary

This folder is the almost-uploadable paper package.

- `output/pdf/aaai27-main.pdf` is the paper reviewers judge.
- `output/pdf/aaai27-supplement.pdf` is the detailed audit and extra evidence.
- `output/pdf/aaai27-checklist.pdf` is the required reproducibility form.
- `output/aaai27-code-data.zip` is the anonymous code and saved data package.
- `abstract.txt` is the exact abstract to paste into OpenReview.
- `submission-metadata.md` tells you which boxes to choose and which personal details are still missing.

Nothing has been submitted under your identity.

## Why AAAI-27

AAAI-27 is the next realistic archival deadline as of July 18, 2026. The abstract is due July 21, the paper July 28, and supplements July 31, all at 11:59 PM UTC-12. The main track accepts empirical, integrative, and critical AI work and evaluates significance, novelty, soundness, relevance, clarity, responsibility, and reproducibility.

The paper uses the official 2027 anonymous two-column style. The final main PDF
is six pages total, with references beginning on page six, so it is comfortably
within the limit of seven technical-content pages plus references. The final
supplement is four pages. The checklist is separate, as the official
instructions require.

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
- The abstract must be real by July 21 and must not change substantially afterward.
- The work cannot be under review at another archival conference or journal.
- All authors must be fixed by July 28 and cannot be added later.
- Reviewers are not required to read the supplement, so every essential claim is already in the main paper.
- The author remains responsible for all AI-assisted writing and for verifying every reference and result.

## Final human gate

Before opening OpenReview, the author must supply the items listed under `Author information still required` in `submission-metadata.md`, read the main PDF, choose the code license, and explicitly approve the frozen title and abstract.
