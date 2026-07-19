# Accurate author-response bank

These are modules, not a final rebuttal. Adapt only the modules that answer the
actual reviews and compress them to the live OpenReview limit.

## Response style

- Lead with the answer.
- Use exact values, pages, tables, and submitted artifacts.
- Agree with a real limitation in one sentence.
- Explain what the current evidence does establish.
- Never claim equivalence from an interval containing zero.
- Never promise a file or experiment that was not submitted.
- Never say a reviewer “failed to read.” Point to the location politely.
- Address the SPC's decision problem, not the reviewer's personality.

## R1 — “This is not subliminal learning because no student is trained.”

> We agree that the experiments do not identify the mechanism of student
> fine-tuning, and the paper states this in the abstract, contributions,
> limitations, external check, and conclusion. The scientific object is the
> frozen subliminal-prompting protocol introduced with token entanglement. The
> contribution is that static output geometry, fixed-head readability, causal
> donor timing, and multi-token scoring are empirically non-equivalent. The
> failed external prediction gate further prevents a training-time overclaim:
> causal donor AUC has rho=0.111 (BH q=0.687) against the released student
> outcome, while the released steering measure has rho=0.768.

## R2 — “The methods are known, so the paper is not novel.”

> We do not claim novelty for probing, activation patching, autoregressive
> scoring, or steering. Table 1 names the seven closest comparisons and their
> exact overlap. The new empirical result is the joint matched contrast: static
> geometry changes by -0.0802 [-0.1271,-0.0347], observational readout AUC by
> +0.0096 [-0.0280,+0.0453], and causal donor AUC by +0.2858
> [+0.2716,+0.2999]. No cited work reports this frozen Llama-3.1 8B/70B
> geometry/readout/natural-state timing comparison together with the decimal
> width stress test.

## R3 — “The three measurements do not use identical numbers.”

> Correct. The full geometry/readout probes use all 1,110 decimal strings, while
> the computationally heavier causal intervention uses a fixed outcome-blind
> subset of 256 width-three strings paired identically across scales. This is now
> stated on page 1. A post hoc sensitivity computed solely from submitted arrays
> restricts geometry/readout to those exact 256 numbers: geometry still changes
> by -0.06461 [-0.12434,-0.00334], specificity by -0.08529
> [-0.14318,-0.03029], and readout AUC by +0.03354
> [-0.00524,+0.07145]. Thus the aggregate geometry decline and unresolved
> readout change survive exact stimulus matching. The 14/18 count is retained
> only as a descriptive full-universe/subset concept comparison.

## R4 — “The animal contrast makes the bootstrap invalid.”

> The target-minus-other-animal contrast was chosen for concept specificity, not
> independence, and the paper limits inference to the fixed concept/pair design.
> More importantly, the causal conclusion does not depend on that algebraic
> coupling. Using raw target logits gives a paired AUC change of +0.27437 with
> the same crossed animal/pair bootstrap interval [+0.25934,+0.28936], and all
> 18 concept changes are positive (main page 4; supplement page 3; submitted
> sensitivity JSON).

## R5 — “The estimator was changed after seeing results.”

> The intervention, pairs, depths, controls, and decision language were designed
> before S5 output. The original subtraction estimator then failed its donor-
> permutation null during local MPS 8B validation because predictor and outcome
> shared a recipient term. The conditional regression was recorded after that
> failure and before any matched CUDA 8B/70B causal output, according to the
> project chronology. We do not claim third-party preregistration or an
> independently verifiable pre-CUDA commit timestamp. The anonymous artifact
> includes the sanitized design/amendment, retains the invalid estimator as an
> audit diagnostic, and shows the corrected permutation coefficients near zero
> with condition numbers below 1.25.

## R6 — “A full-state patch at the answer position is trivial.”

> The claim is not that a late full-state patch can influence an answer in
> isolation. The result is the matched cross-scale timing and specificity. At
> half depth, donor control is 0.038 at 8B and 0.773 at 70B. Permuted donors and
> all 17 wrong-concept shifts stay near zero; identity and duplicate errors are
> exactly zero; raw logits, both direction halves, and all 128 leave-one-pair
> deletions preserve the contrast. We interpret this as intervention-specific
> donor control, not necessity or a feature-level mechanism.

## R7 — “The donor-state/recipient-context hybrid may be off-manifold.”

> We agree. The donor vector is natural in its source run, but transplanting it
> into recipient context can create an off-manifold hybrid. The paper now states
> this directly on page 6. Our estimand is intervention-specific donor control,
> not an average treatment effect over naturally occurring states or
> identification of an endogenous circuit.

## R8 — “Only two models and five depths are insufficient.”

> We agree that one same-release 8B/70B pair does not establish a universal
> scaling law and five states do not localize a continuous onset. We report every
> coefficient and call AUC a coarse trapezoidal profile. The central difference
> already appears at the shared half-depth measurement, before interpolation:
> 0.038 versus 0.773. The exact-eight-block comparison preserves the direction,
> but is explicitly labeled one sensitivity rather than a full architectural
> alignment.

## R9 — “You infer a difference from significant versus non-significant tests.”

> We do not claim the readout traces are equivalent or zero. The paired readout
> AUC change is reported as unresolved: +0.0096 [-0.0280,+0.0453]. Geometry and
> causal AUC have different units, so the paper does not manufacture an arbitrary
> difference-of-differences. The descriptive concept pattern and each exact
> paired interval are reported separately.

## R10 — “The negative external result invalidates the causal finding.”

> It invalidates a training-prediction interpretation, which the paper does not
> make. It does not invalidate the frozen-model intervention result. The external
> check instead sharpens the central lesson: static geometry, observational
> readability, causal donor timing, and a released steering measure are not
> interchangeable proxies for student transfer.

## R11 — “Steering beats all in-house metrics.”

> The fixed paired comparison supports a larger association for released
> steering than causal donor AUC: delta rho=0.657 [0.069,1.207]. We do not claim
> a tested steering-versus-geometry difference. Static geometry is suggestive at
> rho=0.562 but does not survive the fixed three-predictor correction (q=0.078).

## R12 — “The Qwen result is just known length bias.”

> General label-length bias is known and cited. The narrower contribution is a
> protocol-specific boundary: exact width-three autoregressive scoring does not
> recover the positive atomic association in either tested Qwen model, while
> per-token pooling creates a positive sign that disappears under within-width
> standardization. We explicitly do not claim tokenizer causality because
> family, architecture, training, and scale also change.

## R13 — “The intervals overstate population generalization.”

> Geometry/readout intervals resample the fixed 18 concepts. Causal intervals
> cross-resample those concepts and the 128 unordered number-pair clusters while
> retaining both directions and paired 8B/70B structure. The paper explicitly
> limits these intervals to the tested concept and pair design. It makes no
> population claim over arbitrary model families.

## R14 — “The analysis has too many tests.”

> The paper reports separate staged research questions with distinct estimands;
> it does not claim a single globally family-wise-error-controlled confirmatory
> study. All staged outcomes are reported. The external S8 test is the one fixed
> three-predictor multiple-testing family and uses BH-FDR. Exact-subset, raw-
> logit, wrong-label, and leave-one-pair results are labeled sensitivities rather
> than new primaries.

## R15 — “The artifact is incomplete.”

> Every main headline raw array and analyzer is included, so reviewers can rerun
> the headline summaries without renting GPUs. To meet the 50 MB cap, the README
> explicitly lists three omitted non-headline arrays: external-zoo geometry,
> external-zoo layerwise readout, and the MPS full layerwise control. Their saved
> summaries, metadata, collectors, and public-checkpoint recollection paths are
> included. Historical immutable checkpoint revisions were not recorded and are
> stated as a real reproducibility limitation rather than reconstructed.

## R16 — “Generative AI makes the work unreliable.”

> AAAI permits documented generative-AI assistance, and page 7 states its roles
> in this work. The human author remains responsible for every submission item.
> The reliability evidence is concrete: primary references were checked against
> source records, headline analyzers rerun from saved arrays, exact prompts and
> estimands are included, negative results and estimator repair are disclosed,
> and the archive has a verified SHA-256 manifest. We welcome any specific claim,
> citation, or reproducibility concern.
