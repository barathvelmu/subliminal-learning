# AAAI-27 AI-assistance mini-Gargantua

Completed: **July 19, 2026**

## Baby-food verdict

**Keep a disclosure, but make it minimal.**

Deleting it would create a policy risk. Keeping the original long inventory
would create unnecessary reviewer salience. The defensible middle is one short,
broad sentence that truthfully describes the role without signaling the number
of human authors.

Final wording:

> **AI Assistance.** Generative AI tools supported implementation and
> manuscript preparation.

This does not name products, prompts, agents, individual editing operations, or
the number of human authors. It also does not falsely reduce the role to
spelling or grammar correction. The separate AAAI policy already makes all
human authors responsible for the entire submission, so repeating that rule in
the disclosure is unnecessary.

The final wording uses `supported` rather than `assisted with` to make the
human/tool relationship explicit without narrowing the disclosed roles. It
does not say that AI performed the research, designed the estimands, or owns
any claim.

## Final cross-venue wording check

AAAI-27 sits inside an active policy transition. AAAI-26 prohibited
LLM-generated manuscript text except experimental material while allowing
editing and polishing; AAAI-27 does not repeat that prohibition and permits
judicious generative-AI use in manuscript preparation. The standing AAAI publication
policy still requires the system's role to be documented in the manuscript.

Neighboring venues use materially different rules: NeurIPS 2025 does not
require disclosure for ordinary editing or programming aids; ICML 2026 permits
AI assistance in writing or research and encourages disclosure of notable
methodological uses; ICLR 2026 requires any LLM use to be disclosed; and ACL
requires disclosure when generative tools create content. Their papers are
therefore not direct evidence for what an AAAI-27 manuscript should say.

An EMNLP 2024 experiment found that content-generation disclosures can reduce
quality ratings for essays and stories. That establishes a plausible salience
risk, not an AAAI-reviewer effect. The appropriate response is a short,
role-level sentence rather than omission or a dramatic task inventory.

Additional primary sources:

- <https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/>
- <https://neurips.cc/Conferences/2025/LLM>
- <https://icml.cc/Conferences/2026/CallForPapers>
- <https://iclr.cc/FAQ/LLM>
- <https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics>
- <https://aclanthology.org/2024.emnlp-main.279/>

## What the official rules actually say

The AAAI-27 Main Technical Track call says authors may **judiciously** use
generative-AI tools in manuscript preparation and remain fully responsible for
the submission. AAAI's standing publication policy separately says that the
role of any AI system used in developing an AAAI-affiliated publication must be
properly documented in the manuscript.

Neither page requires:

- a model or vendor name;
- prompts or interaction logs;
- a task-by-task inventory;
- a special full section; or
- wording that characterizes the assistance as an author.

Primary sources:

- <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- <https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/>
- <https://aaai.org/conference/aaai/aaai-27/submission-instructions/>

## What recent accepted AAAI papers visibly do

A systematic practice check sampled 29 final AAAI-26 papers, evenly through
three proceedings issues:

- 10 from Machine Learning V, volume 40 issue 28;
- 10 from Natural Language Processing II, volume 40 issue 37; and
- 9 from Philosophy and Ethics of AI, volume 40 issue 42.

PDF text was searched for authoring-disclosure phrases involving ChatGPT,
generative AI, AI-assisted writing/editing/preparation, language models,
Grammarly, and AI proofreading. No visible authoring-AI disclosure was found in
that sample.

Proceedings sampled:

- <https://ojs.aaai.org/index.php/AAAI/issue/view/710>
- <https://ojs.aaai.org/index.php/AAAI/issue/view/719>
- <https://ojs.aaai.org/index.php/AAAI/issue/view/724>

That observation does **not** show that omission is allowed. The papers may not
have used authoring AI, and AAAI-26 had a more restrictive conference-specific
policy that prohibited generated manuscript text while allowing editing or
polishing. The AAAI-27 call now expressly permits judicious use. Past visible
practice cannot override the current standing documentation rule.

AAAI-26 comparison policy:
<https://aaai.org/conference/aaai/aaai-26/policies-for-aaai-26-authors/>

## Risk decision

| Choice | Policy risk | Reviewer-salience risk | Decision |
|---|---:|---:|---|
| Delete the disclosure | High | Low | Reject |
| Keep the long task inventory | Low | High | Reject |
| Use broad one-sentence wording | Low | Low | **Keep** |

The statement sits immediately before References, is anonymous, and is not an
acknowledgment. It does not identify the author or consume a separate page.

## Track correction discovered during the audit

The OpenReview page labeled `AAAI 2027 Artificial Intelligence for Social
Impact Track` is the wrong venue for this paper. The work is a technical
mechanistic study and should use the **AAAI 2027 Conference Main Technical
Track** portal:

<https://openreview.net/group?id=AAAI.org/2027/Conference>
