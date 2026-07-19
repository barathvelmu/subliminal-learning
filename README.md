# subliminal-learning

[![Validate public artifacts](https://github.com/barathvelmu/subliminal-learning/actions/workflows/validate.yml/badge.svg)](https://github.com/barathvelmu/subliminal-learning/actions/workflows/validate.yml)

This repository reproduces two forms of subliminal transfer: an MNIST student trained only on noise and animal-number associations in language-model prompting. The latest experiment compares the prompting channel in Llama-3.1-8B and 70B, where static geometry weakens while causal control develops earlier in relative depth.

Subliminal learning matters for model safety because filtering visible training data may not stop behavioral traits from passing between models. The original results come from [Anthropic's alignment team](https://alignment.anthropic.com/2025/subliminal-learning/) and follow-up work. This project independently replicates the core findings, tests one proposed explanation for a published replication discrepancy, scales the language-model analysis to 70B, and compares static geometry, hidden-state readability, and causal control in one matched setting.

The [paper](Paper/output/pdf/preprint.pdf) is the shortest route to the full result. If the subject is new, start with the [zero-background guide](Paper/Learning/zero-background-guide.md). The [supplement](Paper/output/pdf/supplement.pdf) records the full experimental specification, and the [reproducibility notes](Paper/Reproducibility/README.md) trace the headline numbers back to saved artifacts.

## Quick CPU demonstration

The fastest reproduction trains a teacher on MNIST, then trains a student that never sees a single digit. The student sees only random noise and matches three "auxiliary" outputs that have nothing to do with digits:

```bash
git clone https://github.com/barathvelmu/subliminal-learning.git
cd subliminal-learning && pip install torch torchvision numpy pandas

cd mnist && python quickstart.py
```

```
results (MNIST test set, chance = 10%):
  teacher, trained on 50k real digits:         89.4%
  untrained network (the shared init):         10.4%
  student, trained on noise:                   58.7%   <- subliminal learning
  control, different init, same training:      10.0%   <- near chance
```

The student never saw a digit, and the weights that read out digit predictions never received a gradient. Despite those restrictions, its digit accuracy rises to 58.7%; an otherwise identical student paired with a differently initialized teacher stays at chance.

The language-model demonstration measures animal-number associations for a selected animal (the first run downloads a 1B model, approximately 2.5 GB; CPU is sufficient):

```bash
cd prompting && pip install transformers accelerate scipy
python try_your_animal.py --animal owl
```

## A network that learns digits from noise

The setup, in plain terms: build an MLP with 10 digit outputs plus 3 extra "auxiliary" outputs that mean nothing. Train a teacher on real MNIST. Then take a student that shares the teacher's random initialization, feed it pure noise, and train it to match only those 3 meaningless auxiliary outputs. Nothing about digits is ever shown or supervised.

![Baseline](mnist/figures/baseline_bars.png)

The student reaches **46.3% ± 2.3%** test accuracy against 10% chance (25-model ensemble, 5 seeds, proper train/validation/test splits, with test data used only for headline evaluation). The cross-model control reaches **9.8%**, supporting dependence on shared initialization in this setup.

### Testing a proposed explanation for a published discrepancy

The original paper reported the student above 50%. A follow-up ("Comments & Extensions of Subliminal Learning") found only 27% and suggested the copy loss might explain the gap, since it used MSE where the original used KL. In a direct comparison here, MSE was the strongest of the tested losses (0.566 ± 0.004, compared with 0.45 for KL). The published discrepancy therefore cannot be explained by that loss choice alone.

### What makes the effect stronger or weaker

![Sweeps](mnist/figures/sweeps.png)

Each sweep changes one variable and leaves the other settings at baseline; error bars cover 3 seeds. Two patterns are consistent:

**Transfer declines with width.** Past approximately 128 hidden units, transfer falls monotonically in this sweep (width 128: 0.48; width 1024: 0.22). The decline coincides with smaller weight movement, but the sweep alone does not identify a causal mechanism.

**Noise transfers more strongly than real images.** Distilling on real digit images transfers *worse* (0.147) than distilling on uniform noise (0.45), even though the real images are the only inputs that contain digit features. To find out why, I interpolated the inputs from real images toward noise and tracked accuracy together with how similar the student's internal representation is to the teacher's (linear CKA):

![Noise dose-response](mnist/figures/noise_dose_response.png)

Both rise together smoothly. The data favor a **coverage** explanation: real digits occupy a narrow part of input space, so the student can match the teacher there without copying its broader internal function. Broad noise leaves less room for that shortcut. This is a medium-confidence explanation, not a causal proof. A direct test rejected my first hypothesis that noise carries a richer per-input signal: real images produce stronger auxiliary targets yet transfer less.

### Where the MNIST transfer appears

How does a student classify digits when its digit-readout weights are frozen at random initialization? A linear probe localizes the answer:

| hidden features from | linear-probe accuracy | own frozen readout |
|---|---|---|
| untrained reference | 87.4% | 10.6% |
| student (aux-only) | 90.9% | 39.1% |
| teacher | 93.9% | 93.7% |

The localization result has two parts. Random features are *already* 87% linearly separable, so distillation is not creating digit features. The readout matrix is byte-identical between the reference and the student, yet its accuracy nearly quadruples. Since only the features changed, the gain must come from the hidden layers rotating into alignment with the teacher's representation, which the fixed random readout happens to decode. Pooling every model across all sweeps, student-to-teacher representation similarity predicts accuracy with r = 0.61:

![Mechanism](mnist/figures/mechanism_scatter.png)

**Validation-selected configuration.** With MSE loss, width 128, and 40 distillation epochs, the student reaches **90.0% ± 0.4%** after selection on validation and one test evaluation. The teacher reaches 93.8%, the cross-model control stays at 10.2%, and representation similarity reaches 0.994.

## Animal-number associations in Llama

The prompting version of the same idea, reported as token entanglement: boost one token of a pair (*owl*) and its partner (*087*) rises with it, in both directions. The proposed explanation was unembedding geometry, meaning the two tokens' output vectors simply point in similar directions. Both claims deserved checking.

**The measurement matters.** A naive test (pick the number an animal raises most, then check the reverse direction) is biased in three ways: ratio-ranking preferentially selects rare tokens, the selected winner is compared against a non-exchangeable null, and a no-prompt baseline confounds "any prompt" with "this number." An earlier implementation used this statistic; the reported analysis instead correlates both directions across **all 1,110 one-to-three-digit number tokens**, treating every number identically, over the fixed 18-animal panel.

![Entanglement by animal](prompting/figures/entanglement_by_animal_1b.png)

**Published pairs appear near the top of an animal-dependent distribution.** On Llama-3.2-1B-Instruct, owl→087 is the single most entangled number of all 1,110, and eagle→747 is third. Ten of 18 animals show a positive bidirectional correlation at uncorrected *p* < 0.05, while dog, cat, fox, and several others are near zero. The model's own favorite animals are not more entangled in this panel (*p* = 0.25), so the result does not support favoritism as an explanation for the selected examples.

**The effect is already present in the base model.** Comparing the base model against its instruction-tuned version, the unembedding geometry is 98% similar, the per-animal effects correlate 0.69, and the base model shows the effect in 14/18 animals versus 10/18 in the instruction-tuned model. These observations support inheritance from pretraining but do not isolate the effect of instruction tuning.

**Static output geometry explains little of the behavior.** The unembedding-cosine hypothesis predicts behavior at r ≈ 0.1 to 0.15, which is 1 to 2% of variance. For example, owl→087, the strongest *behavioral* pair, ranks only approximately #80 to #150 of 1,110 by geometry. A mean-centered cosine metric, motivated by softmax's invariance to constant logit shifts, improves prediction of the subliminal direction by 31%. The improvement is not statistically significant across animals and is reported as suggestive.

**Separating generic and animal-specific effects.** The decomposition experiment splits the effect into a generic part (some numbers rise when the model professes love for *anything*) and an animal-specific part. The generic part is large in raw variance but misaligned between directions, so it masks the paired signal. Removing it raises the mean correlation from 0.067 to **0.210**, with positive values for all 18 animals in the fixed panel. A permutation control (re-pairing each animal's forward pattern with a different animal's reverse pattern) scores −0.012 against 0.210 for matched pairs, in 18 of 18 animals. These results support two components: a generic persona shift and an animal↔number coupling that simple geometry captures only faintly.

## At 70B, the measurements separate

The three measurements change differently from 8B to 70B. Static token geometry becomes less predictive, the observational readout change is unresolved, and a natural assistant state gains causal control earlier in relative depth.

| frozen-model measurement | 8B | 70B | paired 70B−8B change |
|---|---:|---:|---:|
| bidirectional behavior, mean *r* | 0.1087 | 0.0846 | −0.0241 [−0.0563, +0.0087] |
| static geometry vs. behavior | 0.1877 | 0.1076 | **−0.0802 [−0.1271, −0.0347]** |
| geometry specificity | 0.1550 | 0.0458 | **−0.1092 [−0.1640, −0.0599]** |
| observational readout AUC | 0.2591 | 0.2688 | +0.0096 [−0.0280, +0.0453] |
| causal donor AUC | 0.2539 | 0.5397 | **+0.2858 [+0.2716, +0.2999]** |

The confidence intervals separate resolved from unresolved changes. The behavior and observational changes are unresolved; the static decline and causal increase are resolved. This is a matched two-checkpoint contrast, not a universal scaling law, but it rejects a scalar account in which geometry, readability, and causal control all move together.

![Static geometry, observational readout, and causal handoff](prompting/figures/s5_causal_handoff.png)

### The causal experiment

The experiment takes a natural assistant-position residual state produced by one animal prompt, patches that full state into a different recipient prompt, and measures which prompt controls the animal answer. It covers 18 animals, 256 width-three number tokens, 128 unordered number pairs in both directions, and five depths. The patched output is regressed on the clean donor and recipient outputs so generic state replacement is not mistaken for donor-specific transmission.

At exactly eight transformer blocks remaining, donor control is about **0.592 at 8B and 0.948 at 70B**. Across the normalized-depth curve, corrected donor AUC rises from 0.2539 to 0.5397, and **all 18 animals increase**. The 70B-minus-8B increase remains positive with raw target logits, in both pair directions, and after leaving out each pair cluster. The 17 wrong-concept shifts, donor permutation, identity patches, and duplicate forwards provide label-specificity and numerical controls. This establishes intervention-specific donor control; it does not show that full-state patching has isolated a unique endogenous circuit.

### Pooling across token lengths can reverse the sign

Qwen breaks `087` into several digit tokens, so a one-token score is no longer the quantity the prompt names. Exact full-sequence scoring on 1,000 width-three strings produces mean correlations of **−0.0498 at Qwen3-0.6B and −0.0478 at Qwen3-1.7B**. Pool all decimal widths and average per token, however, and the same data appears strongly positive: **+0.0970 and +0.2327**. Standardizing inside each width removes that sign again (−0.0489 and −0.0316).

![Multi-token scoring and the decimal-width confound](prompting/figures/s4_multitoken_sequence.png)

The scoring choice changes the quantity being measured. Per-token averaging rewards shorter sequences; decimal width changes both the number of terms being averaged and the tokenization regime. Length composition can therefore produce a positive pooled result even when the fixed-width full-sequence effect is absent.

### External transfer validation is negative

I froze three teacher-side predictors and tested them against released student-transfer outcomes from independent work. None cleared the multiplicity-corrected gate: static geometry was suggestive (Spearman ρ = 0.562, BH *q* = 0.078), the observational readout was weaker, and causal donor timing was not predictive (ρ = 0.111, *q* = 0.687). The released steering benchmark was more predictive (ρ = 0.768).

This null constrains the interpretation: prompt-time causal control is not automatically a predictor of training-time transfer. The supported result is the matched frozen-model comparison; static geometry, observational readability, causal timing, and multi-token scoring should not be treated as interchangeable predictors of training transfer.

## Limits and confidence

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

## Reproducing analyses and figures

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
