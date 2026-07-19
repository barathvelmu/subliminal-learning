# arXiv and other venues: the baby-food guide

Last live-checked against the official AAAI-27 and arXiv pages: **July 19,
2026**.

## The one-sentence answer

Submit this paper to **AAAI-27 and arXiv**, but do **not** submit the same or a
substantially similar paper to another peer-reviewed archival conference or
journal while AAAI is reviewing it.

arXiv is allowed because it is a public preprint server, not a competing
peer-review process.

## What these words mean

| Place | What it does | Peer reviewed? | Allowed while AAAI reviews? |
|---|---|---:|---:|
| AAAI-27 | Reviews the work and may publish it in conference proceedings | Yes | This is the chosen review venue |
| arXiv | Publicly timestamps and shares a preprint | No | **Yes** |
| Personal website or public preprint page | Publicly shares the work | No | Yes, with the anonymity rules below |
| Non-archival workshop | Gives feedback but publishes no archival proceedings | Sometimes lightly | **Yes, only if it is explicitly non-archival** |
| ARR/EACL, ACL, EMNLP, NeurIPS, ICML, ICLR, another archival conference | Runs another publication review | Yes | **No** |
| TMLR or another journal | Runs another archival publication review | Yes | **No** |

The official AAAI-27 submission instructions explicitly say that preprint
servers such as arXiv and non-archival workshops are permitted. They also say
that, after submitting to AAAI-27, the same or substantially similar work may
not be submitted to another archival conference or journal until AAAI gives a
decision or the AAAI submission is withdrawn.

## My recommendation for this paper

Use this sequence:

1. **July 20:** create the AAAI-27 Main Track record and upload the four frozen
   files.
2. **Before July 21 closes:** verify that the real title, abstract, sole-author
   profile, topics, and reviewer declaration are saved.
3. **Before July 28 closes:** verify the final main PDF and checklist on
   OpenReview.
4. **Before July 31 closes:** verify the supplement and code/data ZIP.
5. **After the AAAI package is frozen:** submit a named preprint to arXiv.
6. Do not open a second archival review path unless AAAI rejects the paper or
   the AAAI submission is formally withdrawn.

Baby-food reason: AAAI is the competition; arXiv is the public timestamp. They
can coexist. A second conference or journal would be double submission.

## Why I recommend arXiv

arXiv gives the work:

- a permanent public timestamp;
- a stable arXiv identifier people can cite;
- discoverability before the long conference decision process finishes; and
- a place for later corrected versions.

arXiv does **not** mean the paper was accepted, peer reviewed, or endorsed by
AAAI. It means only that arXiv accepted it as an appropriate scholarly
preprint.

Because this topic is moving quickly, the public timestamp is useful. The AAAI
upload should still come first so arXiv administration does not distract from
the hard July deadlines.

## The AAAI anonymity rules when an arXiv preprint exists

AAAI-27 expressly allows a non-anonymous arXiv version during review. Follow
all four rules:

1. Keep the **AAAI PDF and supplements anonymous**.
2. Do not cite, link, or point from the AAAI submission to the arXiv page,
   personal website, or public repository.
3. Do not write `submitted to AAAI-27`, `under review at AAAI`, or similar text
   anywhere in the arXiv PDF, arXiv comments, GitHub README, website, or social
   post.
4. Do not contact reviewers or try to make reviewers discover the preprint.

The arXiv version may use the same title. That makes the authors discoverable
if someone deliberately searches, but AAAI's published rule allows the
preprint and instructs reviewers not to seek identifying material. Changing
the title merely to evade discovery is unnecessary.

## Do not upload the anonymous AAAI PDF to arXiv

arXiv requires complete and accurate author information and does not accept an
anonymous submission as the public scholarly record.

Prepare a separate arXiv version containing:

- `Barath Velmurugan` as the author;
- the truthful current affiliation, or `Independent Researcher` if that is the
  truthful status;
- a real contact email in the arXiv account and metadata where required;
- the same scientific claims, numbers, figures, tables, references, and
  limitations as the frozen AAAI paper; and
- no statement that the paper is submitted to AAAI.

The anonymous AAAI file remains unchanged on OpenReview. The named arXiv file
is a separate public rendering of the same preprint.

## Recommended arXiv category

- **Primary:** `cs.CL` — Computation and Language
- **Cross-list:** `cs.LG` — Machine Learning

Why: the paper studies hidden signals, tokenization, prompting, and
interpretability in language models. arXiv defines `cs.CL` as natural-language
processing and `cs.LG` as machine-learning research including explanation and
methodology.

Do not choose `cs.AI` as the primary merely because the conference is AAAI.
arXiv's taxonomy says NLP and machine learning have their own subject areas.
The arXiv classifier or moderators may suggest an adjustment; follow their
specific direction if they do.

## Recommended arXiv metadata

- **Title:** `Subliminal Prompting Beyond Static Geometry: Causal Depth and
  Multi-Token Confounds`
- **Authors:** `Barath Velmurugan` plus a truthful current affiliation
- **Abstract:** use the final paper abstract
- **Comments:** `<final rendered page count> pages, 3 figures, 5 tables.
  Preprint.`
- **Journal reference:** leave blank
- **DOI:** leave blank
- **Report number:** leave blank unless a real institution assigned one
- **Primary category:** `cs.CL`
- **Cross-list:** `cs.LG`

Do not write AAAI, `under review`, `submitted`, `forthcoming`, or `accepted` in
the comments field.

## What file arXiv should receive

Because the paper was created in LaTeX, arXiv prefers and normally expects the
LaTeX source rather than only a generated PDF. Build a neutral named-preprint
version. Do not publish the AAAI `[submission]` rendering, which is anonymous
and carries review-only distribution language, and do not claim AAAI
camera-ready copyright before acceptance.

Build a clean arXiv source bundle containing only:

- the named neutral-preprint `.tex` file;
- the bibliography source;
- the required figures;
- required style files; and
- any other local source dependency needed to compile.

Do not include:

- API keys, credentials, or environment files;
- local absolute paths;
- build logs or temporary LaTeX files;
- the 49.5 MB AAAI code/data ZIP;
- private rebuttal or audit notes;
- OpenReview identifiers; or
- the anonymous AAAI submission metadata.

The arXiv uploader will compile the source. Preview every generated page before
the final `Submit Article` action.

## arXiv endorsement

The arXiv website decides whether endorsement is needed. It commonly asks when
someone submits their first paper or first enters a new category.

Do this:

1. Create or sign in to the arXiv account.
2. Start a new submission and choose `cs.CL`.
3. Let the website check the account.
4. If it does not request endorsement, continue normally.
5. If it requests endorsement, check the email from arXiv for the endorsement
   link or code.
6. Prefer an institutional email if one is truthful and currently available.
7. Otherwise, ask one active arXiv author who knows the field, ideally a
   professor or researcher who can see that the paper belongs in `cs.CL`.
8. Do not mass-email strangers or repeatedly pressure one person.

Endorsement is not peer review, coauthorship, or scientific approval. It only
confirms that the submission belongs in the research community and category.

## arXiv license

The arXiv paper license and the code license are two different decisions.

For the paper, the conservative choice while future publication rights are
still unresolved is:

**arXiv.org perpetual, non-exclusive license 1.0**

It gives arXiv permission to distribute the preprint while leaving other reuse
rights limited. arXiv says the selected license is irrevocable. Do not select
CC0. A broader Creative Commons license can be useful, but it grants broader
reuse rights and should be chosen only after understanding the future
publisher's requirements.

The code archive currently has no public reuse license. Posting the paper on
arXiv does not automatically license the code. Do not publicly release the code
until a deliberate software-license choice is made.

## Exact arXiv submission sequence

1. Start a new submission.
2. Confirm or obtain endorsement for `cs.CL`.
3. Select the arXiv perpetual, non-exclusive license.
4. Upload the clean named LaTeX source bundle.
5. Let arXiv detect the compiler and top-level file.
6. Delete any file arXiv correctly marks as unnecessary.
7. Compile and preview the generated PDF.
8. Inspect the title, name, affiliation, abstract, every page, figures,
   references, and AI-assistance sentence.
9. Paste the metadata listed above using plain ASCII punctuation.
10. Choose `cs.CL` primary and request `cs.LG` cross-listing.
11. Verify that no field mentions AAAI submission or review.
12. Click the final `Submit Article` button.
13. Wait for moderation and public announcement.
14. Save the assigned arXiv identifier.

Starting a draft is not the same as publicly submitting it. The final button is
required.

## What happens after AAAI decides

### If AAAI accepts

- Do not submit the same paper to another conference or journal as a new work.
- Prepare the named AAAI camera-ready version.
- Add the AAAI publication information to the arXiv `journal-ref` and DOI fields
  when those details exist.
- Upload a new arXiv version only when the camera-ready content is ready and its
  copyright terms permit it.
- A later journal extension may be possible only if it is substantially
  expanded, cites the conference version, and satisfies the journal and AAAI
  policies.

### If AAAI rejects in Phase 1 or after Phase 2

- The review lock ends once the rejection decision is received.
- Read the reviews and improve the paper.
- Upload a corrected arXiv version if useful.
- Then select one next archival route, such as ARR/EACL, another appropriate
  conference, or TMLR.
- Do not submit to several archival routes simultaneously.

### If Barath withdraws from AAAI

- Confirm that OpenReview shows the paper as withdrawn.
- Only then begin another archival submission.
- arXiv may remain public because it is not the competing review venue.

## Final decision

**Recommended:** AAAI-27 plus arXiv.

**Not recommended:** AAAI-27 plus another conference or journal at the same
time.

**Best timing:** finish AAAI first, create the arXiv draft and resolve any
endorsement, then publicly submit the named arXiv version after the AAAI files
are frozen.

## Primary official sources

- AAAI-27 submission instructions:
  <https://aaai.org/conference/aaai/aaai-27/submission-instructions/>
- AAAI-27 review process and preprint rule:
  <https://aaai.org/conference/aaai/aaai-27/review-process/>
- arXiv submission overview:
  <https://info.arxiv.org/help/submit/index.html>
- arXiv endorsement:
  <https://info.arxiv.org/help/endorsement.html>
- arXiv licenses:
  <https://info.arxiv.org/help/license/index.html>
- arXiv metadata:
  <https://info.arxiv.org/help/prep.html>
- arXiv category taxonomy:
  <https://arxiv.org/category_taxonomy>
