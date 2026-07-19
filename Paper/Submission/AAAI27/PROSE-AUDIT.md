# Scientific prose audit

This is a writing-quality audit, not an attempt to evade AI detection.

## Applied rules

- Lead with the causal result, which is the strongest result.
- Use concrete nouns and measured quantities instead of promotional adjectives.
- Keep each claim inside the experiment's actual scope.
- Distinguish resolved effects from intervals that span zero.
- Say measured intervention-specific donor control; reserve donor-state
  sufficiency for the near-unit coefficients at measured late depths.
- Say one matched 8B/70B contrast, not a universal scaling law.
- Say the Qwen pooled statistic is width-driven, not proof of semantic entanglement.
- Cite the closest prior work, including work that narrows the novelty claim.
- Include the failed first estimator and the recorded validation-stage repair.
- Avoid citations in the abstract, as required by the AAAI author kit.
- Avoid gratuitous em dashes and stock transitions. No conference rule bans em dashes; this is a readability preference only.
- Avoid claims such as "first ever" when the literature audit only supports a scoped comparison.

## Phrases intentionally removed or avoided

- "groundbreaking," "revolutionary," "novel paradigm," and "unprecedented";
- "proves token entanglement";
- "no effect" when an interval spans zero;
- "humanized" as a synonym for disguising authorship;
- detector-oriented claims about punctuation or prose.

## Final public-readability pass

The GitHub landing page originally used a centered-dot link strip and a
compressed `Part 1 / Part 2 / Part 3` banner. Both were removed. The links now
sit inside ordinary sentences, and the section headings describe the
experiments directly.

The submission manuscript did not share that broader problem. One introduction
paragraph did use the stock `First / Second / Third` contribution template; it
was rewritten as connected prose without changing any claim or number. The
rest of the paper remained untouched unless a sentence had a concrete clarity
problem. Formal methods, tables, and checklists were not made artificially
casual merely to create stylistic variation.

## Author responsibility and disclosure

AAAI-27 permits judicious generative-AI assistance, requires its role to be
documented in the manuscript, and makes the authors fully responsible for every
submitted statement, reference, figure, artifact, and policy declaration. The
final paper contains a deliberately short `AI Assistance` paragraph. A July 19
mini-audit found no rule requiring tool names, prompts, or a task-by-task
inventory; the standing policy requires only that the role be properly
documented. The author should read the entire final PDF and verify that the
broad wording truthfully covers both the work and the assistance used to
prepare it. See `../../Research/aaai27-ai-assistance-audit.md`.
