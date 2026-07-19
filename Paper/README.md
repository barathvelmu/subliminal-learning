# Paper

The public manuscript and its supplement live here:

- [Preprint](output/pdf/preprint.pdf)
- [Supplementary material](output/pdf/supplement.pdf)
- [Source](preprint.tex)
- [Reproducibility map](Reproducibility/README.md)
- [Zero-background guide](Learning/zero-background-guide.md)

The preprint is the named, shareable version of the paper. The anonymous review
build is kept in a separate local author workspace so that the PDF submitted to
review is not confused with the version shared on GitHub or arXiv.

## Main result

From Llama-3.1-8B to 70B, fixed output vectors become less useful for predicting
animal-number behavior. Yet copying a hidden state from one number prompt into
another controls the 70B answer much earlier in the network. A separate Qwen
experiment shows that mixing numbers with different digit lengths can reverse
the sign of the measured association.

## Scope

These experiments study prompting in models whose weights stay fixed. They do
not establish how traits transfer during training, a universal scaling law, or
a unique circuit. The supplement gives the exact prompts, measurements,
controls, protocol amendments, and secondary results.
