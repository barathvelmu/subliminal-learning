# AAAI accepted-paper benchmark

## Why this file exists

This is the stopping rule for the paper-improvement loop. The goal is not to
imitate surface style or add equations for decoration. The goal is to match the
evidence structure seen in strong AAAI papers: one crisp question, a visible
claim-to-test chain, decisive controls, honest boundaries, and a package a
reviewer can audit.

## Papers inspected

The official AAAI proceedings PDFs were inspected, including recent award
papers and nearby interpretability work:

- AAAI-26 Outstanding Paper: *CADyT*
- AAAI-26 Outstanding Paper: *ReconVLA*
- AAAI-26 Best Paper in AI Alignment: *Global Opinion Dynamics with LLM Agents*
- AAAI-26: *Beyond Transcription* (mechanistic/representation analysis)
- AAAI-25: *Task-Specific Circuits in Language Models*

Local copies used for visual comparison are in `tmp/pdfs/aaai-benchmark/` and
are research scratch files, not submission artifacts. Award status was checked
against AAAI's official conference-paper awards page.

## Repeated pattern in strong papers

1. **One central gap.** The reader can say in one sentence what prior work
   measured and what the new paper measures that it did not.
2. **A result figure early.** Page one or two shows the conceptual experiment
   and the result, not only a workflow diagram.
3. **Operational equations.** Formulas define estimands or algorithms. They are
   not decorative.
4. **Claims sit beside controls.** A headline effect is paired with the null,
   ablation, sensitivity, or comparison that rules out the easiest objection.
5. **The page budget is used.** Strong papers do not hide essential prompts,
   mapping rules, or robustness evidence in vague prose.
6. **Limitations narrow the claim.** They do not erase the result; they specify
   exactly what the experiment did and did not identify.
7. **Artifacts are runnable.** Exact commands, versions, data, and identity
   checks reduce reviewer uncertainty.

## Our paper's gap, in one sentence

Prior subliminal-learning work often treats behavior, vocabulary geometry, and
hidden-state signal as one channel; we jointly show that static geometry,
fixed-head readability, and causal donor timing separate across a matched
Llama-3.1 8B/70B comparison, while multi-token normalization creates a distinct
measurement confound in two Qwen models.

## Threshold before submission

The paper is ready for the author's final approval when all of these are true:

- Main PDF uses at most 7 technical pages and at most 9 pages total.
- Page one states the gap, exact scope, and central result without a broad
  novelty claim.
- The closest cross-scale probe/patch paper, *Where's the Plan?*, is cited and
  differentiated explicitly.
- The causal estimator, five mapped depths, AUC interpolation, and amendment
  history are visible.
- Relative-depth and remaining-block interpretations are both reported.
- Headline controls include all-animal consistency, wrong concepts, raw logits,
  leave-one-pair-cluster-out, and identity checks.
- Qwen wording says the measurement changes across a tokenizer/model-family
  boundary; it does not claim tokenizer causality.
- Every plot is legible at final PDF size and has a caption that names the
  plotted estimand correctly.
- Code/data commands run from a clean archive layout and its manifest verifies.
- The PDF contains no author identity, local path, secret, broken citation,
  clipped object, or unembedded font.

## Current reviewer-style target

Three independent adversarial passes found no fatal result-invalidating flaw.
Before revision, the median judgment was approximately **6/10 (weak accept)**.
After the claim, citation, robustness, and reproducibility fixes listed above,
the explicit target is **7/10 (accept)**. This is a benchmark, not a promise of
conference acceptance.

## Final page-budget surgery

The remaining useful gap was not another formula. It was reviewer navigation.
The final main paper now contains a nearest-prior-work comparison table, four
additional primary citations for tokenization and patching methodology, the
exact outcome-blinded external-validation statistics, and a sharper causal
scope caveat. These additions make the novelty boundary and inconvenient null
result visible without changing any headline estimate. The PDF uses six pages
of technical content; references begin on page six and continue to page seven.
Every threshold above is now satisfied. Further cosmetic density would violate
this file's stopping rule unless new evidence is collected.

## Sources

- AAAI-27 call: https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/
- AAAI-27 instructions: https://aaai.org/conference/aaai/aaai-27/submission-instructions/
- AAAI paper awards: https://aaai.org/about-aaai/aaai-awards/aaai-conference-paper-awards-and-recognition/
- Closest cross-scale probe/patch comparison: https://arxiv.org/abs/2605.07984
- Activation-patching best practices: https://arxiv.org/abs/2309.16042
- 70B circuit-analysis precedent: https://arxiv.org/abs/2307.09458
