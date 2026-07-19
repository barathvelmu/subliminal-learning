# Reviewer attack matrix

This file separates three things:

- **repaired attacks:** real issues fixed before submission;
- **answerable attacks:** concerns already answered by submitted evidence; and
- **genuine limits:** facts we should concede rather than spin away.

| Priority | Likely attack | Type | Current shield | Rebuttal posture |
|---:|---|---|---|---|
| 1 | “This studies frozen prompting, not student training.” | Genuine limit | Abstract, contributions, limitations, external null, conclusion | Agree; explain why separating measurements constrains training-time interpretations |
| 2 | “The novelty is only a bundle of known tools.” | Answerable judgment | Main Table 1 now gives seven exact nearest comparisons | Do not claim tools are new; defend the matched geometry/readout/causal dissociation |
| 3 | “Full-state patching at the answer position is trivial or generic.” | Partly answerable, partly genuine | Half-depth contrast, all 17 wrong concepts, permutation, identity, raw logit, pair-direction, leave-one-pair controls | Defend cross-scale timing and specificity; concede no neutral-task baseline or feature-level mechanism |
| 4 | “The probes and patch use different number sets.” | Repaired | Main page 1 discloses 1,110 versus 256; page 4 and supplement page 2 report exact-256 sensitivity | State exact values; keep 14/18 as descriptive full-universe/subset comparison |
| 5 | “The target-minus-other contrast couples the 18 animals.” | Repaired/answerable | Raw target-logit causal delta +0.27437, crossed CI [+0.25934,+0.28936], 18/18 | Agree about algebraic coupling; show conclusion survives without the contrast |
| 6 | “The estimator changed after results.” | Answerable with a provenance limit | Main and supplement disclose failed null; artifact includes sanitized S5 design/amendment | Say recorded, not externally preregistered; correction preceded matched CUDA/70B by author-reported chronology |
| 7 | “Natural-state patching is still an off-manifold hybrid.” | Genuine limit | Main page 6 now says this directly | Agree; call result intervention-specific donor control, not endogenous circuit identification |
| 8 | “Two scale points and five depths do not establish scaling/onset.” | Genuine limit | Same-release matched pair, full coefficient table, exact eight-block sensitivity | Agree; call it a matched contrast and coarse profile, not a law or exact onset |
| 9 | “No direct statistical test proves readout equals/does not equal causal change.” | Answerable wording | Paper says readout change is unresolved, never equivalent | Reject the premise: no equivalence claim; report separate paired intervals and 14/18 descriptive direction pattern |
| 10 | “The external null defeats the paper.” | Answerable plus genuine significance cost | Main page 6–7 and supplement page 3 report all outcomes | It defeats a training-prediction interpretation, which the paper does not make; it supports measurement separation |
| 11 | “Steering is more predictive than every in-house measure.” | Repaired | Main now says larger association than causal AUC; supplement gives paired difference | Do not claim a tested steering-versus-geometry difference |
| 12 | “Qwen only rediscovers generic length bias.” | Partly answerable | Exact scorer validation, fixed width-three primary, within-width standardization, cited label-length literature | Defend protocol-specific sign reversal; concede no tokenizer-causal isolation |
| 13 | “Qwen changes tokenizer, family, architecture, training, and scale.” | Genuine limit | Main and supplement state all confounds | Agree; call it a measurement boundary, not tokenizer causality |
| 14 | “Bootstrap CIs generalize to all concepts/models.” | Answerable | Main methods and limitations restrict inference to fixed concepts/pairs | State the exact resampling units; never claim model-family population inference |
| 15 | “Many staged tests lack global multiplicity control.” | Answerable with limitation | All staged estimands are reported; S8 alone has a fixed three-predictor BH family | Do not call all analyses one confirmatory family; label post hoc sensitivities honestly |
| 16 | “Reviewers cannot reproduce the 70B work.” | Answerable with omissions | Main headline arrays and analyzers included; no GPU required for summary reruns | List exactly which three non-headline raw arrays were omitted and why |
| 17 | “Checkpoint revisions were not pinned.” | Genuine limit | README discloses it | Agree; never invent hashes; included arrays preserve the analyzed outputs |
| 18 | “No reuse license weakens reproducibility.” | Open human decision | Checklist currently says partial; README is honest | Not a compliance failure; human may license original code before July 31 |
| 19 | “The supplement contains essential evidence.” | Answerable | Main includes prompts, estimands, controls, key sensitivities, limits, and values | Point to main pages first; use supplement only for audit detail |
| 20 | “AI wrote an unreliable paper.” | Answerable responsibility issue | Page-7 disclosure, verified citations, deterministic reruns, human responsibility | Do not debate detector vibes; answer concrete claim/reference/result concerns |
| 21 | “Chauhan and Shah already show geometry and behavior separate.” | Answerable novelty boundary | Related Work and Table 1 cite the contemporaneous July 5 preprint; local paper card records the exact overlap | Agree on the broad separation; distinguish their observational 1B base/instruct result from our matched 8B/70B natural-state intervention and width-controlled measurement bundle |

## Fatal only if true

These would be fatal, but the audit found no evidence that any is true:

- a prior paper already reports the same matched frozen Llama-3.1 8B/70B
  natural full-state intervention and joint geometry/readout dissociation;
- the 8B and 70B causal prompts, pairs, or depth mappings were not shared;
- the conditional estimator was selected after viewing matched CUDA/70B output;
- included arrays fail to reproduce the headline contrast or controls;
- the paper claims causal donor timing explains student fine-tuning; or
- any reference, result, figure, or artifact is fabricated.

## Normal limitations, not fatal defects

- one same-release 8B/70B pair;
- five causal depths;
- 18 animal concepts;
- frozen prompting rather than student training;
- full-state rather than feature-level patching;
- no neutral-task timing baseline;
- two small Qwen models;
- one aggregate external training seed per animal; and
- missing immutable model-checkpoint revisions.
