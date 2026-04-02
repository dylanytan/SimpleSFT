PDF_NAME := estimator_report

.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=$(PDF_NAME) main.tex

clean:
	latexmk -C -jobname=$(PDF_NAME) main.tex
	rm -f $(PDF_NAME).pdf
