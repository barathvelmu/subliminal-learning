# Final map: read this, submit this, then stop

Last live rule check and full artifact audit: **July 19, 2026**.

## The baby-food answer

The AAAI-27 package is **green and ready to submit**. Nothing has been submitted
for you. Your job is now human ownership and website administration, not more
experiments or last-minute paper surgery.

You are making **one AAAI submission**, not three different submissions. That
one record has three deadlines:

1. **July 21:** title, abstract, topics, author, and reviewer declaration.
2. **July 28:** main paper and reproducibility checklist.
3. **July 31:** technical supplement and code/data ZIP.

All four files are ready now. Upload them in the same sitting if OpenReview
allows it, then use the later dates only to verify that everything remains
there.

## First: understand the page counts

There is no twelve-page main paper.

| File | Pages | What it is |
|---|---:|---|
| Main paper | **7** | The paper reviewers primarily judge |
| Technical supplement | **3** | Extra methods, controls, and numerical detail |
| Reproducibility checklist | **2** | A separate required AAAI form |
| Total pages inspected | **12** | Seven plus three plus two across three PDFs |

AAAI permits up to seven pages of technical content, followed by pages used
only for references and the checklist. Our main PDF is seven pages total:
technical content and the beginning of the references share page 7 cleanly.
There is no forced reference page, spacing trick, or manual page break.

## Tomorrow: do these six things in order

### 1. Read and own the main paper

Open
[`Submission/AAAI27/output/pdf/aaai27-main.pdf`](Submission/AAAI27/output/pdf/aaai27-main.pdf)
and read all seven pages once.

You do not need to reproduce every formula in your head. You do need to be able
to say, truthfully:

- what the paper asks;
- what the strongest result is;
- what the Qwen result means;
- what the paper does **not** claim;
- that the reported work and one-line AI disclosure are accurate; and
- that you accept responsibility for the complete submission.

If a sentence seems wrong, mark it and stop. If it is merely dense or you would
have phrased it differently, do not rewrite a frozen paper on deadline day.

### 2. Open the correct venue

Use the
[`AAAI 2027 Conference Main Technical Track portal`](https://openreview.net/group?id=AAAI.org/2027/Conference).

The page must say **AAAI 2027 Conference**. Do **not** create this paper under
the separate **Artificial Intelligence for Social Impact Track** page.

### 3. Create the single paper record

Paste the frozen scientific fields from
[`Submission/AAAI27/submission-metadata.md`](Submission/AAAI27/submission-metadata.md).

The most important fields are:

- **Title:** `Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds`
- **Primary topic:** `NLP: Interpretability, Analysis & Evaluation (incl. Factuality & Hallucination)`
- **Secondary topics:**
  - `NLP: (Large) Language Models`
  - `ML: Transparent, Interpretable & Explainable ML`
  - `ML: Representation Learning`
- **TL;DR:** paste
  [`Submission/AAAI27/tldr.txt`](Submission/AAAI27/tldr.txt).
- **Abstract:** paste
  [`Submission/AAAI27/abstract.txt`](Submission/AAAI27/abstract.txt).

Do not use placeholder text. Do not casually rewrite the abstract: the website
version already matches the paper exactly.

### 4. Answer the human-only fields truthfully

Only you can decide these:

- your publication name and sole-author profile;
- your truthful current affiliation and its country;
- your conflicts of interest;
- whether the work is under review elsewhere;
- whether you meet AAAI's reciprocal-reviewer publication threshold; and
- the policy and attendance confirmations shown by OpenReview.

This is a solo paper. AAAI does not require a professor, sponsor, coauthor, or
endorsement. Do not add a ceremonial coauthor.

If no author has at least two qualifying first-author papers or five qualifying
coauthored papers, use OpenReview's truthful `no author qualifies`
declaration. Do not nominate a professor who is not an author.

If your truthful current status is unaffiliated, use `Independent Researcher`
instead of claiming a former institution as current.

### 5. Upload these exact four files

| OpenReview field | Exact local file | Cap | Ready size |
|---|---|---:|---:|
| Main PDF | [`aaai27-main.pdf`](Submission/AAAI27/output/pdf/aaai27-main.pdf) | 10 MB | 0.98 MB |
| Reproducibility Checklist | [`aaai27-checklist.pdf`](Submission/AAAI27/output/pdf/aaai27-checklist.pdf) | 5 MB | 0.10 MB |
| Technical Supplement | [`aaai27-supplement.pdf`](Submission/AAAI27/output/pdf/aaai27-supplement.pdf) | 10 MB | 0.18 MB |
| Code and Data Supplement | [`aaai27-code-data.zip`](Submission/AAAI27/output/aaai27-code-data.zip) | 50 MB | 49.495 MB |

Leave **Media Supplement** empty.

Do not rebuild, unzip/re-zip, license, edit, or replace the code/data file with
a repository link. It is deliberately just below the 50 MB live-form cap and
its internal checksum manifest already passes.

### 6. Prove the submission worked

After saving the record:

- [ ] Open the new paper ID.
- [ ] Download all four files back from OpenReview.
- [ ] Open the downloaded main PDF.
- [ ] Confirm the title is correct.
- [ ] Confirm the author line says `Anonymous Submission`, not your name.
- [ ] Confirm the code/data download is about 49.5 MB.
- [ ] Use OpenReview's **Email** button to send yourself confirmation.
- [ ] Save the paper ID privately.

OpenReview does not automatically email a receipt. Filling the form without
saving it is not submission.

## The three deadlines in your time zone

AAAI uses **11:59 PM UTC-12**, also called Anywhere on Earth.

| Gate | Official deadline | New York equivalent | Safe target |
|---|---|---|---|
| Abstract record | July 21, 2026 | July 22, 7:59 AM EDT | **July 20, 8 PM EDT** |
| Main PDF and checklist | July 28, 2026 | July 29, 7:59 AM EDT | **July 27, 8 PM EDT** |
| Supplement and code | July 31, 2026 | August 1, 7:59 AM EDT | **July 30, 8 PM EDT** |

These are three gates on one paper record. They are not three versions for
three committees.

## The rules that matter

### Format and anonymity

- Use the official AAAI-27 two-column submission format on US Letter paper.
- The official LaTeX package in the current Author Kit is
  `\usepackage[submission]{aaai2027}`. It is **not** `aaai27`. Our local style
  and bibliography files match the official Author Kit byte-for-byte.
- The main submission and all support files must be anonymous.
- Names, affiliations, identifying local paths, acknowledgments, and author
  metadata must be absent during review.
- The main paper must be self-contained. Reviewers are not required to read the
  supplement.
- The separate reproducibility checklist is required.
- Source files are requested only if the paper is accepted.

Our package passes those checks: 7/3/2 pages, US Letter, embedded fonts, no PDF
attachments, no author identity, no overfull boxes, and no unresolved citations
or references.

References do not need a title page or a separate page. In the official
two-column conference format, they begin wherever the technical content ends
and continue naturally. Our page-7 transition is valid.

### AI assistance

AAAI's written publication policy says that an AI system may not be an author
or citable source and that its role in developing the publication must be
documented in the manuscript. Our entire disclosure is this one line:

> Generative AI tools assisted with implementation and manuscript preparation.

It is accurate, minimal, neutral, and does not reveal the number of human
authors. Leave it unchanged. You, the human author, remain responsible for the
text, figures, references, code, analyses, and submission.

### Other venues and arXiv

- Do **not** place the same or substantially similar work under review at
  another archival conference or journal while AAAI is reviewing it.
- AAAI explicitly permits a public arXiv preprint and a genuinely non-archival
  workshop.
- The anonymous AAAI files must not link or point to a public named preprint.
- A public preprint, website, repository, or social post must not say that the
  work is submitted to or under review at AAAI-27.

The recommended order is: submit AAAI first, freeze all four AAAI files, then
prepare a separate named neutral arXiv version. Read
[`Learning/arxiv-and-other-venues-guide.md`](Learning/arxiv-and-other-venues-guide.md)
before touching arXiv.

### What not to change now

Do not add new experiments, equations, citations, figures, acknowledgments,
spacing commands, a forced reference page, a license, a coauthor, or a longer
AI statement just because there is still time. More material is not
automatically a stronger paper. The current document already uses its space for
the causal result, controls, matched comparison, multi-token confound, closest
prior work, limitations, and reproducibility.

## What happens after the July uploads

### Immediately after submitting

- Keep the paper ID and confirmation private.
- Re-open the record before each hard deadline and verify the saved files.
- Do not open a second archival review path.
- After the July 31 freeze, follow the arXiv guide if you want the public
  preprint.

### September 24: Phase 1 result

AAAI assigns three reviewers in Phase 1. A sufficiently negative paper can be
rejected here without author feedback. That is a conference rule, not proof
that the paper or submission package was defective.

### October 19–25: author feedback, only if the paper reaches Phase 2

Phase 2 can add reviewers, for up to five reviews total. Use the response kit
starting at [`Rebuttal/PACKAGE-START-HERE.md`](Rebuttal/PACKAGE-START-HERE.md).

The response is for correcting factual errors and answering reviewer questions.
It is not permission to run a new paper inside the rebuttal or silently replace
the July artifacts.

### November 30: final decision

If accepted, AAAI will provide the camera-ready instructions. At that point you
restore your name/affiliation, add appropriate acknowledgments, make only
allowed revisions, and supply source files. Recheck the live camera-ready date
and instructions in the acceptance notice rather than relying on an old local
note.

If rejected, the work remains yours. Read the reviews, improve it, and choose
one next archival venue. A rejection does not block arXiv.

## Your reading map

### Required now

Read only these three things before touching OpenReview:

1. [`Submission/AAAI27/output/pdf/aaai27-main.pdf`](Submission/AAAI27/output/pdf/aaai27-main.pdf)
2. This file: `Paper/00-START-HERE-FINAL-MAP.md`
3. [`Submission/AAAI27/submission-metadata.md`](Submission/AAAI27/submission-metadata.md)

The longer click-by-click backup is
[`Submission/AAAI27/START-HERE-SUBMISSION-ROADMAP.md`](Submission/AAAI27/START-HERE-SUBMISSION-ROADMAP.md).

### If the science feels confusing

Read in this order:

1. [`Learning/zero-background-guide.md`](Learning/zero-background-guide.md) —
   assumes you know almost nothing.
2. [`Learning/how-conference-submission-works.md`](Learning/how-conference-submission-works.md) —
   explains conference vocabulary and the three gates.
3. [`Research/frontier-decision.md`](Research/frontier-decision.md) — explains
   why this experiment became the paper.
4. [`Research/novelty-matrix.md`](Research/novelty-matrix.md) — shows what
   earlier papers did and what this paper adds.
5. [`Research/Papers/`](Research/Papers/) — one short local card for each key
   prior paper.

### If you are worried about quality or solo authorship

- [`Submission/AAAI27/GARGANTUA-FINAL-AUDIT.md`](Submission/AAAI27/GARGANTUA-FINAL-AUDIT.md) —
  complete format, science, reproducibility, anonymity, policy, and visual
  audit.
- [`Research/aaai-accepted-benchmark.md`](Research/aaai-accepted-benchmark.md) —
  comparison with accepted-paper patterns.
- [`Learning/solo-aaai-reality-check.md`](Learning/solo-aaai-reality-check.md) —
  official data showing that sole-authored Main Track papers are rare but real.
- [`Research/aaai27-ai-assistance-audit.md`](Research/aaai27-ai-assistance-audit.md) —
  the written rule and why the one-line disclosure is the defensible choice.

### Later, not now

- [`Rebuttal/`](Rebuttal/) — use only if Phase 2 author feedback opens.
- [`Learning/arxiv-and-other-venues-guide.md`](Learning/arxiv-and-other-venues-guide.md) —
  use after the AAAI package is safely frozen.
- [`Manuscript/`](Manuscript/) — historical long-form workbench, not the upload
  draft.
- [`scaling/memory.md`](../scaling/memory.md) and
  [`scaling/experiments.md`](../scaling/experiments.md) — chronological lab
  record, not tomorrow's checklist.

## The two Downloads packages

- [`Subliminal-Learning-AAAI27-Review-Pack-2026-07-19-FINAL.zip`](/Users/barathv/Downloads/Subliminal-Learning-AAAI27-Review-Pack-2026-07-19-FINAL.zip)
  is the small reading/email package. It includes the three PDFs, learning
  guides, research comparisons, final map, audit, and rebuttal shield, but omits
  the large code/data ZIP.
- [`Subliminal-Learning-AAAI27-Complete-Package-2026-07-19-FINAL.zip`](/Users/barathv/Downloads/Subliminal-Learning-AAAI27-Complete-Package-2026-07-19-FINAL.zip)
  contains all of the above plus the exact 49.5 MB code/data upload.

Both archives were rebuilt after this final map, passed ZIP-integrity tests,
and contain byte-identical copies of the frozen submission artifacts. The
small pack is easiest to email to family. The complete pack is the offline
handoff and upload backup.

## Final audit result

On July 19, the following were rechecked from scratch:

- live AAAI deadlines, page rules, double-blind policy, supplement rules,
  simultaneous-submission rule, and AI policy;
- all three LaTeX builds;
- every rendered page in all three PDFs;
- page dimensions, font embedding, attachments, metadata, and identity leakage;
- 23 citation keys against 23 bibliography entries;
- exact abstract match and OpenReview text limits;
- the 49.5 MB archive's compression and 49-file checksum manifest;
- fresh regeneration of all included headline analyses; and
- byte-for-byte regeneration of all three figures.

One directly relevant July 5 preprint and one notation-clarity issue were found
and repaired. A second scan then found no further justified change. This does
**not** guarantee acceptance; no honest reviewer or collaborator can guarantee
that. It means there is no known
scientific, formatting, anonymity, reproducibility, or packaging problem that
justifies another pre-submission editing loop.

## The stop rule

If your personal read finds a factual error, stop and fix that exact error with
care. Otherwise, submit the frozen files. The highest-value remaining work is
accurate form completion and successful upload verification.
