# Solo AAAI reality check

Completed: **July 19, 2026**

## Baby-food verdict

**Yes, a solo AAAI Main Technical Track paper is real. It is also rare.**

An accepted solo paper would be a serious archival AI publication. It would
show that one named author took responsibility for the contribution. It would
not mean the paper won an award, and it would not make the submission easier to
accept.

During anonymous review, the PDF says `Anonymous Submission`. Reviewers score
the science, not the size of the author team. Solo authorship therefore gives no
review bonus. Its meaning becomes visible after acceptance.

## Actual numbers

AAAI's official AAAI-26 opening material reports a Main Track acceptance rate
of about **17.5%**. In baby-food language: across the whole conference, roughly
one out of six reviewed papers was accepted.

Source:
<https://aaai.org/wp-content/uploads/2026/01/AAAI-26-Opening-Ceremony.pdf>

I also scanned the author line of every article record in the **43 AAAI-26 Main
Technical Track proceedings issues**:

- accepted article records scanned: **4,149**;
- records with exactly one listed author: **26**;
- observed solo share among those accepted records: **0.627%**;
- plain English: about **one solo paper per 160 accepted papers**.

This 0.627% figure is our reproducible scan of the official proceedings, not a
statistic published by AAAI. It measures the accepted record only. It does not
tell us the solo-submission acceptance rate because AAAI does not publish how
many solo papers were submitted.

Official proceedings archive:
<https://ojs.aaai.org/index.php/AAAI/issue/archive>

A directly relevant proof-of-existence is Omar Claflin's sole-authored AAAI-26
paper, *Feature Integration Spaces: Joint Training Reveals Dual Encoding in
Neural Network Representations*. The official record lists one author and the
affiliation `Independent`:
<https://ojs.aaai.org/index.php/AAAI/article/view/40847>

Other sole-authored AAAI-26 records include work on LLM preference
optimization, recursive-logic generalization, diffusion sampling, planning,
reinforcement learning, computer vision, and statistical learning theory.
Solo acceptance is not restricted to one narrow subject.

## Where this submission honestly stands

The correct label is:

> **Submission-ready and genuinely competitive, but not acceptance-predictable.**

Reasons this is a real paper rather than a speculative draft:

- the central causal scale contrast is large and precise: **+0.2858**, 95% CI
  **[+0.2716, +0.2999]**;
- all **18 of 18** concepts move in the same direction;
- the raw-logit sensitivity remains positive: **+0.2744**, 95% CI
  **[+0.2593, +0.2894]**;
- the matched 256-number geometry sensitivity remains negative: **-0.0646**,
  95% CI **[-0.1243, -0.0033]**;
- wrong-concept, permuted-donor, identity-patch, leave-one-pair,
  remaining-block, backend, and tokenizer-width controls are reported;
- the negative external-transfer gate and unresolved observational result are
  disclosed instead of hidden;
- the final package contains the seven-page main paper, technical supplement,
  reproducibility checklist, code/data archive, and checksum manifest;
- the claims, references, statistics, anonymity, page budget, and upload files
  survived two hostile-review waves.

The remaining scientific vulnerabilities are real:

- the clean headline scale comparison is one Llama 8B/70B release pair;
- the Qwen comparison mixes tokenizer, architecture, training, and scale;
- full-residual patching establishes intervention-specific sufficiency, not a
  unique circuit or necessity;
- the frozen prompting channel does not identify the mechanism of
  training-time subliminal transfer;
- no in-house measure passed the preregistered external transfer-prediction
  gate.

Those limitations prevent an honest guarantee. They do not make the paper
broken; they define its scope.

## Strict stop/go decision

**Green:** submit this to the AAAI-27 **Main Technical Track**.

**Stop:** do not launch a decorative last-minute experiment, inflate the claim,
or move the paper into the Social Impact track.

**Also stop:** do not delete the minimal AI-assistance statement. AAAI permits
judicious generative-AI use, but its standing publication policy says the role
of any AI system used in developing an AAAI-affiliated publication must be
documented. This project used AI assistance for workflow automation,
implementation, and manuscript preparation. The current one-sentence note
summarizes those activities as implementation and manuscript preparation; it
is the smallest honest, policy-aligned version.

Official current rules:

- <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- <https://aaai.org/aaai-publications/aaai-publication-policies-guidelines/>

## What Barath does next

1. Read the seven-page main PDF once and mark anything he cannot personally
   defend.
2. Register the abstract in the **Main Technical Track** by July 21, 2026 at
   11:59 PM UTC-12.
3. Upload the final paper and checklist by July 28.
4. Upload the supplement and code/data archive by July 31.
5. Keep every statement truthful. The author, not an AI tool, owns the final
   submission and every answer during review.
