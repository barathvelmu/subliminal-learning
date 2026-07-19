# START HERE: your AAAI-27 submission roadmap

Last live-verified against the official AAAI-27 pages and the active OpenReview
submission form: **July 18, 2026**.

## The one-sentence answer

The paper package is **green and ready**. You are not making three different
paper submissions. You create **one OpenReview submission record**, then finish
it across three gates:

1. **July 21:** register the real title, abstract, topics, author, and reviewer
   status.
2. **July 28:** upload the main paper and reproducibility checklist.
3. **July 31:** upload the technical supplement and code/data ZIP.

Nothing has been uploaded or submitted under your identity yet.

**Best strategy:** once your OpenReview profile is active, create the record and
upload all four prepared files in the same sitting. You do not need to wait for
July 28 or July 31. Those dates are last chances, not appointment dates. If
everything is uploaded today, the later gates become simple verification days.

## Your only urgent risk

You need an **active OpenReview profile** before you can register the paper.
OpenReview says a new profile made with a public email such as Gmail can take up
to two weeks to moderate. The abstract deadline is only three days away.

Do this **today**, not on July 21:

- [ ] Go to <https://openreview.net/signup>.
- [ ] Enter your real publication name exactly as you want it printed forever.
- [ ] Check whether an existing profile already belongs to you. **Do not create
      a duplicate account.** If one exists, log in or reset its password.
- [ ] If you need a new account, use a university or employer email that you
      legitimately control, if available. This is the fastest path.
- [ ] Open the confirmation email and finish the full profile registration.
- [ ] Fill your current position, institution, career/education history, and a
      real homepage. Add a DBLP URL if one exists for you. OpenReview accepts a
      personal page, institutional page, Google Scholar page, or GitHub page
      that visibly connects your name and professional identity.
- [ ] Confirm that the profile is **active**, not merely created or pending.

If you only have a personal email, register immediately. If you are not
currently affiliated with a university or company, use the truthful current
role **Independent Researcher** and include your previous education/career
record. Do not claim an old affiliation as current. Make the profile complete;
that helps moderation.

If the profile is pending:

1. Add and verify a legitimate institutional or employer email to the same
   pending account. Do not create a second account.
2. If you have no institutional email, make sure your homepage shows your name,
   email, and career information.
3. Use <https://openreview.net/contact> for OpenReview account help. Use
   `workflowchairs@aaai.zendesk.com` for an AAAI submission-system problem.

## The clock in New York time

AAAI calls its deadlines **Anywhere on Earth**, or UTC-12. OpenReview may show
the next calendar day in UTC. These are the same moments.

| Gate | Official deadline | New York equivalent | Our safe internal target |
|---|---|---|---|
| Abstract registration | July 21, 11:59 PM UTC-12 | July 22, 7:59 AM EDT | **July 20, 8:00 PM EDT** |
| Main paper | July 28, 11:59 PM UTC-12 | July 29, 7:59 AM EDT | **July 27, 8:00 PM EDT** |
| Supplement and code | July 31, 11:59 PM UTC-12 | August 1, 7:59 AM EDT | **July 30, 8:00 PM EDT** |

The official deadline is the hard wall. The internal target is the fallback if
you do not finish everything today; it leaves time for an internet, account, or
upload problem.

## Gate 1: create the submission record before July 21

Open the AAAI-27 portal:
<https://openreview.net/group?id=AAAI.org/2027/Conference>

Click **AAAI 2027 Conference Submission** and fill the following fields.

### Ready-to-paste scientific fields

**Title**

> Subliminal Prompting Beyond Static Geometry: Causal Depth and Multi-Token Confounds

**TL;DR**

> Across matched Llama-3.1 8B and 70B models, static token geometry weakens while natural-state causal control arrives earlier, and naive multi-token normalization can create a false positive association.

**Abstract**

Copy the exact contents of `abstract.txt`. Do not use a placeholder.

**Primary topic**

> NLP: Interpretability, Analysis & Evaluation (incl. Factuality & Hallucination)

**Secondary topics**

Use these three, not all five available slots:

1. NLP: (Large) Language Models
2. ML: Transparent, Interpretable & Explainable ML
3. ML: Representation Learning

These topics target reviewers who can judge the actual mechanism study. Extra
generic safety or benchmarking topics could route the paper to a less suitable
reviewer pool.

### Fields only you can answer truthfully

- **Author:** add only your active OpenReview profile. This is a solo paper.
- **Country of institution:** select the country of your truthful current
  institution. If you register as an independent researcher in the United
  States, select `United States`.
- **Self-declared conflicts:** this field is optional. Add real people whose
  relationship to you creates a conflict; do not invent names and do not list
  ordinary prior-work authors merely because the paper cites them.
- **Policy confirmations:** confirm only if true that your profile is complete,
  this work is not under review at another archival venue, and any relevant
  simultaneous work is disclosed as required.

### The reciprocal-reviewer question

This answer and the paper topics **freeze after July 21**.

AAAI calls an author qualified only if that author has at least:

- two first-author papers, or
- five coauthored papers

in peer-reviewed archival venues related to AAAI. Workshop papers do not count.

If you have no qualifying publication record, do not nominate yourself and do
not add a professor as a fake coauthor. Select the exact truthful option:

> We declare that no author qualifies, understanding that the submission may be desk-rejected if this is found to be incorrect

The warning is about lying when a qualified author exists. A genuinely
unqualified solo author is allowed to make the declaration.

### Before you leave Gate 1

- [ ] Real title, TL;DR, and complete abstract are present.
- [ ] Primary and three secondary topics are correct.
- [ ] Your sole-author profile is correct and active.
- [ ] Country, reviewer status, conflicts, and policy answers are truthful.
- [ ] Save the submission.
- [ ] Click the new paper ID and confirm the record opens correctly.
- [ ] Use OpenReview's **Email** button if you want a confirmation email;
      OpenReview does not automatically send one.
- [ ] Save the paper ID in a password manager or your private notes.

Do not upload an arXiv paper for this gate. arXiv is separate and optional.
Because all four conference files are already ready, continue directly to Gate
2 and Gate 3 in this same session if OpenReview permits the uploads.

## Gate 2: upload the paper before July 28

You may change the author list/order and uploaded paper until this deadline.
The title, TL;DR, and abstract may receive small corrections but must not change
substantively. The topics and reciprocal-reviewer answer are already frozen.

Upload these exact files to these exact fields:

| OpenReview field | File | Size cap | Our file size | Status |
|---|---|---:|---:|---|
| Main PDF | `output/pdf/aaai27-main.pdf` | 10 MB | 0.98 MB | Ready |
| Reproducibility Checklist | `output/pdf/aaai27-checklist.pdf` | 5 MB | 0.10 MB | Ready |

The main PDF is anonymous, seven pages total, US Letter, and uses embedded
fonts. Technical material stays within page seven; deferred Tables 4 and 5
appear at the top of that page, and References begin below them without a
forbidden manual page break. This is inside the official limit of seven
non-reference pages and nine total pages.

Before leaving Gate 2:

- [ ] Read the main PDF once from beginning to end.
- [ ] Confirm that the PDF says `Anonymous Submission`, not your name.
- [ ] Confirm the paper is not simultaneously under review at another archival
      conference or journal.
- [ ] Upload the main PDF to **Main PDF**.
- [ ] Upload the checklist to **Reproducibility Checklist**, not to the
      technical-supplement field.
- [ ] Save the revision.
- [ ] Open the paper record and download both files back from OpenReview.
- [ ] Open the downloaded files and confirm they are the correct versions.
- [ ] Use the **Email** button for a confirmation message.

After July 28, the main paper and author information are frozen.

## Gate 3: upload the support files before July 31

Upload these exact files:

| OpenReview field | File | Size cap | Our file size | Status |
|---|---|---:|---:|---|
| Technical Supplement | `output/pdf/aaai27-supplement.pdf` | 10 MB | 0.18 MB | Ready |
| Code and Data Supplement | `output/aaai27-code-data.zip` | 50 MB | 49.49 MB | Ready, close to cap |

There is no media supplement, so leave that field empty.

Important:

- Put the ZIP in **Code and Data Supplement**, not **Media Supplement**.
- Do not re-zip it, add files, or rename files inside it. It is deliberately
  just below the 50 MB cap and its checksum manifest already verifies.
- Do not replace the ZIP with a GitHub or anonymous-repository link. AAAI's live
  form explicitly forbids external code/data repository links during review.
- The supplement and ZIP are anonymous. Do not add your name, local computer
  path, acknowledgments, or personal repository URL.

Before leaving Gate 3:

- [ ] Upload the technical supplement PDF.
- [ ] Upload the code/data ZIP to the correct field.
- [ ] Save the revision.
- [ ] Download both files back from OpenReview and verify that they open.
- [ ] Use the **Email** button for confirmation.

After July 31, nothing can be changed until the decision process allows it.

## What happens after submission

- **September 24, 2026:** Phase 1 rejection notifications. If the paper is not
  rejected here, it continues to Phase 2.
- **October 19-25, 2026:** author-feedback window for papers in Phase 2. This is
  when you answer reviewer concerns. It is not an opportunity to run an
  entirely new paper.
- **November 30, 2026:** final acceptance or rejection.
- **December 14, 2026:** camera-ready deadline if accepted. The real author name
  and affiliation replace anonymity, source files are submitted, and copyright
  is transferred to AAAI.
- **February 16-23, 2027:** in-person conference in Montréal. At least one
  author must register and present. Because this is a solo paper, that means
  you.

Do not submit the same or substantially similar paper to another archival venue
while AAAI is reviewing it. arXiv and non-archival workshops are permitted by
AAAI, but an arXiv post can reduce anonymity in practice, so it is optional and
not part of this immediate checklist.

## Green-light verdict

**Science and files: GREEN.** The main paper, checklist, supplement, and
code/data archive are ready and already passed layout, anonymity, font,
bibliography, file-size, and archive-integrity checks.

The full final record is in `GARGANTUA-FINAL-AUDIT.md`. The manuscript now also
contains the AI-use disclosure required by AAAI policy. Do not remove it.

**Administration: GREEN as soon as the OpenReview profile is active and the
three gates above are completed.** No professor, sponsor, endorsement, AAAI
membership, or ceremonial coauthor is required to submit.

Do not keep rewriting the paper for reassurance. The highest-value action now
is creating the active OpenReview profile and registering the real abstract
early.

## Official sources

- AAAI-27 submission instructions:
  <https://aaai.org/conference/aaai/aaai-27/submission-instructions/>
- AAAI-27 main technical track:
  <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- AAAI-27 modification rules:
  <https://aaai.org/conference/aaai/aaai-27/paper-modification-guidelines/>
- AAAI-27 supplement rules:
  <https://aaai.org/conference/aaai/aaai-27/supplementary-material/>
- AAAI-27 topics:
  <https://aaai.org/conference/aaai/aaai-27/areas-and-topics/>
- AAAI-27 publication and attendance:
  <https://aaai.org/conference/aaai/aaai-27/paper-publication-and-conference-attendance/>
- OpenReview profile creation:
  <https://docs.openreview.net/getting-started/creating-an-openreview-profile/signing-up-for-openreview>
- OpenReview activation help:
  <https://docs.openreview.net/getting-started/creating-an-openreview-profile/expediting-profile-activation>
