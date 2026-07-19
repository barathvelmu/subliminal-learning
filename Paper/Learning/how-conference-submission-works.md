# How this conference paper works, from zero

## The simplest picture

A conference paper is not usually a school essay with a separate cover page and a new page just for references. The conference gives every author one shared template. That template controls the title, columns, font, margins, figures, and references.

For AAAI-27, page 1 starts with the title and anonymous author label. The abstract begins directly below it. The scientific paper follows in two columns. References continue after the conclusion. This is normal conference formatting.

## What each file does

1. The **main paper** is the argument. Reviewers must be able to judge the work from this file alone.
2. The **supplement** is extra evidence. It contains long tables, exact prompts, audit history, and details that do not fit cleanly in the main paper.
3. The **reproducibility checklist** answers the conference's standard questions about code, data, hardware, metrics, and uncertainty.
4. The **code and data ZIP** lets a reviewer rerun the statistics without paying for a 70B GPU experiment.
5. The **OpenReview form** holds the real author identities, conflicts, title, abstract, and topic choices. Reviewers see an anonymous version.

## What "double blind" means

The reviewers should not know who wrote the paper, and the authors should not know who the reviewers are. The PDF therefore says `Anonymous Submission`. Real names are entered privately in OpenReview. Personal repository links, acknowledgments, and identifying file metadata must stay out of the review package.

## Why references are not on a separate page

AAAI allows up to seven pages of technical content. References may continue
beyond page seven, and a separate reference page is not required. In this paper,
all 22 references fit on page seven without shrinking the official typography or
adding a page, so the final version uses a clean page break after the conclusion.

## What happens next

1. Before July 21, create or repair the OpenReview profile and register the complete title, abstract, and author list.
2. Before July 28, upload the anonymous main PDF and the separate checklist.
3. Before July 31, upload the technical supplement PDF and anonymous code/data ZIP.
4. Reviewers score novelty, significance, correctness, clarity, relevance, responsibility, and reproducibility.
5. If the paper reaches Phase 2, the authors receive reviews and write a short response.
6. If accepted, the anonymous label is replaced with real author information and the final source package is prepared.

## What the author must personally verify

Read the final PDF from start to finish. Check that every claim is understandable and defensible. Confirm the author list, conflicts, affiliation, email, and submission-policy statements. Choose a code license. These are identity and legal decisions, so a coding agent should not invent them.
