# subliminal-learning

[![Validate public artifacts](https://github.com/barathvelmu/subliminal-learning/actions/workflows/validate.yml/badge.svg)](https://github.com/barathvelmu/subliminal-learning/actions/workflows/validate.yml)

**A model can teach another model things it never says out loud.** I started this repo with two replications: a small MNIST network that learns to read handwritten digits from **pure random noise**, and a language model where telling it to love *owls* quietly makes it love the number *087*, and vice versa. The newest experiment pushes the second result to Llama-3.1-70B. There, the simple geometry story gets weaker even as causal control arrives much earlier.

Subliminal learning sounds like it should not be possible, and that is exactly why it matters. If a student model can pick up a teacher's traits through data that visibly contains nothing about those traits, then filtering training data for bad content is not enough to stop bad traits from spreading between models. The original results come from [Anthropic's alignment team](https://alignment.anthropic.com/2025/subliminal-learning/) and follow-up work; this project independently replicates the core findings, resolves a contradiction between two published numbers, scales the language-model analysis to 70B, and ends with a mechanism story that the data actually supports.

The [paper](Paper/output/pdf/preprint.pdf) is the shortest route to the full result. If the subject is new, start with the [zero-background guide](Paper/Learning/zero-background-guide.md). The [supplement](Paper/output/pdf/supplement.pdf) records the full experimental specification, and the [reproducibility notes](Paper/Reproducibility/README.md) trace the headline numbers back to saved artifacts.

## See it happen (one minute, CPU only)

The fastest way to believe this effect is to watch it. This trains a teacher on MNIST, then trains a student that never sees a single digit; it only sees random noise and matches three "auxiliary" outputs that have nothing to do with digits:

```bash
git clone https://github.com/barathvelmu/subliminal-learning.git
cd subliminal-learning && pip install torch torchvision numpy pandas

cd mnist && python quickstart.py
```

```
results (MNIST test set, chance = 10%):
  teacher, trained on 50k real digits:         89.4%
  untrained network (the shared init):         10.4%
  student, trained ONLY on noise:              58.7%   <- subliminal learning
  control, different init, same training:      10.0%   <- collapses to chance
```

The student never saw a digit, and the weights that read out digit predictions never received a gradient. Yet it reads digits. An otherwise identical student paired with a differently initialized teacher stays at chance, showing that the effect depends on the shared starting point.

There is a language-model version too. Pick any animal and watch which numbers it is entangled with (first run downloads a 1B model, ~2.5 GB; CPU is fine):

```bash
cd prompting && pip install transformers accelerate scipy
python try_your_animal.py --animal owl
```

## A network that learns digits from noise

The setup, in plain terms: build an MLP with 10 digit outputs plus 3 extra "auxiliary" outputs that mean nothing. Train a teacher on real MNIST. Then take a student that shares the teacher's random initialization, feed it pure noise, and train it to match only those 3 meaningless auxiliary outputs. Nothing about digits is ever shown or supervised.

![Baseline](mnist/figures/baseline_bars.png)

The effect replicates cleanly: the student reaches **46.3% ± 2.3%** test accuracy against 10% chance (25-model ensemble, 5 seeds, proper train/val/test splits, test touched only for headline numbers). The cross-model control collapses to **9.8%**, showing that transfer depends on shared initialization in this setup.

### A contradiction in the literature

The original paper reported the student above 50%. A follow-up ("Comments & Extensions of Subliminal Learning") found only 27% and suggested the copy loss might explain the gap, since it used MSE where the original used KL. In a direct comparison here, MSE was the strongest of the tested losses (0.566 ± 0.004, compared with 0.45 for KL). The published discrepancy therefore cannot be explained by that loss choice alone.

### What makes the effect stronger or weaker

![Sweeps](mnist/figures/sweeps.png)

One sweep at a time, everything else at baseline, error bars over 3 seeds. Two findings worth pausing on:

**Wider networks are worse at this.** Past ~128 neurons, transfer falls monotonically (width 128: 0.48, width 1024: 0.22). Very wide networks barely move their weights during training, and it is precisely that weight movement, the network reshaping its internal features toward the teacher's, that carries the trait. Less movement, consequently less transfer.

**Noise beats real images, by a lot.** Distilling on real digit images transfers *worse* (0.147) than distilling on uniform noise (0.45), even though the real images are the only inputs that contain digit features. To find out why, I interpolated the inputs from real images toward noise and tracked accuracy together with how similar the student's internal representation is to the teacher's (linear CKA):

![Noise dose-response](mnist/figures/noise_dose_response.png)

Both rise together, smoothly. The story the data supports is **coverage**: real digits live in a thin sliver of input space, so the student can match the teacher there without copying its full internal function. Broad noise leaves no such shortcut. I want to be honest about the epistemics here: my first hypothesis was that noise carries a richer per-input signal, and a direct check refuted it (real images produce *stronger* auxiliary targets yet transfer less), thus coverage is the best surviving explanation, held at medium confidence, not proof.

### The mechanism, concretely

How does a student classify digits when its digit-readout weights are frozen at random initialization? A linear probe localizes the answer:

| hidden features from | linear-probe accuracy | own frozen readout |
|---|---|---|
| untrained reference | 87.4% | 10.6% |
| student (aux-only) | 90.9% | 39.1% |
| teacher | 93.9% | 93.7% |

Two numbers do all the explaining. Random features are *already* 87% linearly separable, so distillation is not creating digit features. And the readout matrix is byte-identical between the reference and the student, yet its accuracy nearly quadruples. Since only the features changed, the gain must come from the hidden layers rotating into alignment with the teacher's representation, which the fixed random readout happens to decode. Pooling every model across all sweeps, student-to-teacher representation similarity predicts accuracy with r = 0.61:

![Mechanism](mnist/figures/mechanism_scatter.png)

**Pushing the theory to its limit:** if representation-copying is the driver, then stacking every lever that increases copying should push a noise-trained student close to its teacher. It does. With MSE loss, width 128, 40 distillation epochs (selected on validation, evaluated once on test), the student reaches **90.0% ± 0.4%**, approaching the 93.8% teacher, while the cross-model control stays at 10.2% chance and representation similarity hits 0.994. The 90% is genuine shared-initialization transfer, not ordinary distillation sneaking in.

## When "you love owls" makes the model love 087

The prompting version of the same idea, reported as token entanglement: boost one token of a pair (*owl*) and its partner (*087*) rises with it, in both directions. The proposed explanation was unembedding geometry, meaning the two tokens' output vectors simply point in similar directions. Both claims deserved checking.

**The measurement matters.** A naive test (pick the number an animal boosts most, check it boosts the animal back) is biased in three separate ways; ratio-ranking preferentially selects rare tokens, the hand-picked winner is compared against a non-exchangeable null, and a no-prompt baseline confounds "any prompt" with "this number". I caught this in my own v1 and threw it out. The clean statistic correlates both directions across **all 1110 one-to-three-digit number tokens**, treating every number identically, over 18 pre-registered animals, all reported.

![Entanglement by animal](prompting/figures/entanglement_by_animal_1b.png)

**The effect is real and the published pairs replicate.** On Llama-3.2-1B-Instruct, owl→087 is the single most entangled number of all 1110, and eagle→747 is third. But the effect is strongly animal-dependent: 10 of 18 animals show a significant bidirectional correlation, and dog, cat, fox and friends show essentially nothing. The published examples sit at the very top of the distribution. I also tested whether the model's own *favorite* animals are more entangled (the papers select favorites), and they are not (p = 0.25); reported examples therefore appear selected for effectiveness, not favoritism.

**It comes from pretraining, not instruction-tuning.** Comparing the base model against its instruct version: the unembedding geometry is 98% identical between them, the per-animal effects correlate 0.69, and the base model actually shows the effect in *more* animals (14/18 vs 10/18). Instruction-tuning inherits this quirk; it does not create it.

**The geometry explanation mostly fails.** The unembedding-cosine hypothesis predicts behavior at r ≈ 0.1 to 0.15, which is 1 to 2% of variance. One vivid data point: owl→087, the strongest *behavioral* pair, ranks only ~#80 to #150 of 1110 by geometry. I proposed a cleaner metric (mean-centered cosine, motivated by softmax's invariance to constant logit shifts) and it does improve prediction of the subliminal direction by 31%, but the improvement is not statistically significant across animals, so I report it as suggestive only.

**So what is actually going on?** The decomposition experiment splits the effect into a generic part (some numbers light up when the model professes love for *anything*) and an animal-specific part. The generic part is large in raw variance but misaligned between directions, so it acts as a mask. Removing it raises the mean correlation from 0.067 to **0.210**, with positive values for all 18 animals in the fixed panel. A permutation control (re-pairing each animal's forward pattern with a different animal's reverse pattern) scores −0.012 against 0.210 for matched pairs, in 18 of 18 animals. The best-supported picture is therefore two components: a generic persona shift that obscures an animal↔number coupling underneath, which simple geometry captures only faintly.

## At 70B, the easy explanation breaks

A tempting scaling story is that the owl↔087 channel simply gets stronger as models get bigger. The matched Llama-3.1-8B/70B comparison says something more interesting: **the measurements split apart**. Static token geometry becomes decisively less predictive, an observational readout stays roughly where it was, and a natural assistant state gains causal control much earlier.

| frozen-model measurement | 8B | 70B | paired 70B−8B change |
|---|---:|---:|---:|
| bidirectional behavior, mean *r* | 0.1087 | 0.0846 | −0.0241 [−0.0563, +0.0087] |
| static geometry vs. behavior | 0.1877 | 0.1076 | **−0.0802 [−0.1271, −0.0347]** |
| geometry specificity | 0.1550 | 0.0458 | **−0.1092 [−0.1640, −0.0599]** |
| observational readout AUC | 0.2591 | 0.2688 | +0.0096 [−0.0280, +0.0453] |
| causal donor AUC | 0.2539 | 0.5397 | **+0.2858 [+0.2716, +0.2999]** |

The intervals are the point, not decoration. The behavior and observational changes are unresolved. The static decline and causal increase are not. This is a matched two-checkpoint contrast, not a universal scaling law, but it is enough to reject one scalar story in which geometry, readability, and causal control all move together.

![Static geometry, observational readout, and causal handoff](prompting/figures/s5_causal_handoff.png)

### The causal experiment

The intervention is deliberately concrete. Take a natural assistant-position residual state produced by one animal prompt, patch that full state into a different recipient prompt, and ask which prompt now controls the animal answer. Do this for 18 animals, 256 width-three number tokens, 128 unordered number pairs in both directions, and five depths. Then regress the patched output on the donor and recipient clean outputs, so generic state replacement is not mistaken for donor-specific transmission.

At exactly eight transformer blocks remaining, donor control is about **0.592 at 8B and 0.948 at 70B**. Across the whole normalized-depth curve, corrected donor AUC rises from 0.2539 to 0.5397, and **all 18 of 18 animals increase**. The conclusion survives raw target logits, all 17 wrong-concept controls, donor permutation, both pair directions, identity patches, duplicate forwards, and leaving out each number-pair cluster in turn. That establishes intervention-specific donor control; it does not claim that full-state patching has isolated the unique endogenous circuit.

### Tokenization can manufacture the sign

Qwen breaks `087` into several digit tokens, so a one-token score is no longer the quantity the prompt names. Exact full-sequence scoring on 1,000 width-three strings produces mean correlations of **−0.0498 at Qwen3-0.6B and −0.0478 at Qwen3-1.7B**. Pool all decimal widths and average per token, however, and the same data appears strongly positive: **+0.0970 and +0.2327**. Standardizing inside each width removes that sign again (−0.0489 and −0.0316).

![Multi-token scoring and the decimal-width confound](prompting/figures/s4_multitoken_sequence.png)

The scoring choice changes the quantity being measured. Per-token averaging rewards shorter sequences; decimal width changes both the number of terms being averaged and the tokenization regime. Length composition can therefore produce a positive pooled result even when the fixed-width full-sequence effect is absent.

### The external test did not flatter us

I froze three teacher-side predictors and tested them against released student-transfer outcomes from independent work. None cleared the multiplicity-corrected gate: static geometry was suggestive (Spearman ρ = 0.562, BH *q* = 0.078), the observational readout was weaker, and causal donor timing was not predictive (ρ = 0.111, *q* = 0.687). The released steering benchmark was more predictive (ρ = 0.768).

That null belongs in the result. Prompt-time causal control is not automatically a predictor of training-time transfer. The narrow contribution here is the joint matched comparison: static geometry, observational readability, causal timing, and multi-token scoring are empirically non-equivalent measurements of the subliminal-prompting channel.

## What I would and would not trust

The main limits are:

- The MNIST coverage explanation is a dose-response correlation plus a refuted alternative, not a causal proof. Medium confidence.
- Representation similarity is necessary but not sufficient: one configuration (4x init scale) reached high similarity at chance accuracy, and the overall correlation is a loose r = 0.61.
- The prompting effects are modest in absolute terms (r ≈ 0.1 to 0.2), and the exact entangled pairs are model-specific: on the 8B model the phenomenon holds (13/18 animals) but owl→087 drops from rank 1 to rank 684. Do not expect the same pairs elsewhere.
- The two-component decomposition is one way to slice the effect; confirming the split needs more model families and prompt formats.
- The scale study compares one same-release 8B/70B pair at five coarse depths. It supports a matched contrast, not a scaling law or an exact causal-onset layer.
- Full residual-state patching establishes intervention-specific sufficiency and control. It does not establish necessity, identify a unique feature or circuit, or prove that the hybrid activation is fully on-manifold.
- The Qwen comparison changes tokenizer, model family, architecture, and scale together. It identifies a real measurement boundary, not which of those changes caused it.
- Prompt-time causal control did not predict released training-time transfer. A toy MLP, frozen prompting, and a full fine-tuning pipeline answer different questions and should not be collapsed into one mechanism claim.

The cross-model control originally used a plain random permutation, which pairs about 1 of 25 students with its own teacher and slightly inflates the control. It now uses a derangement, and every affected number was recomputed. A repeated prompting run matched the reported correlations to six decimal places.

## Reproducing everything

The original MNIST and 1B experiments ran locally; the matched 8B/70B comparison used full-BF16 CUDA, with 70B sharded across four RTX A6000 GPUs. The quickstart and animal demo still run on CPU. Checked-in summary JSONs regenerate every headline figure without repeating the expensive forward passes; the [reproducibility map](Paper/Reproducibility/README.md) records raw artifacts, exact commands, frozen checks, and SHA-256 hashes.

```bash
# MNIST experiment (mnist/)
python experiment.py --seeds 0 1 2 3 4 --eval-split test --tag baseline_5seed
python sweep.py                  # all five sweeps -> results/sweeps_summary.csv
python noise_investigation.py    # real->noise interpolation + CKA
python probe.py                  # the linear-probe table
python maximize.py               # the 90% run + validity gate
python make_figures.py && python make_noise_figure.py

# Llama prompting experiments (prompting/)
python entanglement.py --tag 1b                                       # 18 animals x 1110 numbers
python entanglement.py --model unsloth/Llama-3.1-8B-Instruct --tag 8b # 8B cross-check
python geometry_metrics.py --tag 1b     # cosine vs centered-cosine vs dot products
python base_vs_instruct.py              # pretraining-inheritance test
python decompose.py                     # generic vs animal-specific split
python make_figures.py 1b

# 8B/70B and Qwen paper figures (run from the repository root)
python scaling/make_s4_figures.py
python scaling/plot_causal_patch.py \
  --summary prompting/results/causal_patch_s5_8b70b_cuda_summary.json \
  --output prompting/figures/s5_causal_handoff.png
```

Models pull from the ungated `unsloth/` mirrors (identical weights to `meta-llama/`, no token needed); MNIST downloads automatically.

## References

- [*Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data*](https://arxiv.org/abs/2507.14805), and the accompanying [Anthropic Alignment Science post](https://alignment.anthropic.com/2025/subliminal-learning/)
- *Comments & Extensions of Subliminal Learning* (the 27% replication this project re-examines)
- [*Token Entanglement in Subliminal Learning*](https://openreview.net/forum?id=auKgpBRzIW) (the prompting effect and the unembedding-geometry hypothesis)
- [*Learning Through Noise: Why Subliminal Learning Works and When It Fails*](https://arxiv.org/abs/2605.23645) (a complementary theoretical and empirical account)
- [*Subliminal Steering: Stronger Encoding of Hidden Signals*](https://arxiv.org/abs/2604.25783) and [*From Data to Behavior: Predicting Unintended Model Behaviors Before Training*](https://arxiv.org/abs/2602.04735) (nearby steering and transfer-prediction work)
