LATEXMK ?= latexmk
OUT := output/pdf

.PHONY: all clean

all: $(OUT)/preprint.pdf $(OUT)/supplement.pdf

$(OUT)/preprint.pdf: preprint.tex references.bib aaai2027.sty aaai2027.bst figures/causal-handoff.png figures/geometry-depth.png figures/tokenizer-width.png
	mkdir -p $(OUT)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUT) preprint.tex

$(OUT)/supplement.pdf: supplement.tex references.bib aaai2027.sty aaai2027.bst
	mkdir -p $(OUT)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUT) supplement.tex

clean:
	$(LATEXMK) -C -outdir=$(OUT) preprint.tex supplement.tex
