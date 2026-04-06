PDF_NAME := estimator_report
GENERATED_ACCURACY := generated/accuracy_summary.tex
CALIBRATION_REPORT := ../benchmark_artifacts/_cleaned_corpus/phase_aligned_report_latest.md

.PHONY: pdf clean

pdf: $(GENERATED_ACCURACY)
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error -jobname=$(PDF_NAME) main.tex

$(GENERATED_ACCURACY): ../scripts/report_phase_aligned_calibration.py \
	../benchmark_artifacts/_cleaned_corpus/canonical_measurements.csv
	mkdir -p generated
	python ../scripts/report_phase_aligned_calibration.py \
		--output $(CALIBRATION_REPORT) \
		--tex-output $(GENERATED_ACCURACY)

clean:
	latexmk -C -jobname=$(PDF_NAME) main.tex
	rm -f $(PDF_NAME).pdf
	rm -rf generated
	rm -f main.aux
	rm -f main.fdb_latexmk
	rm -f main.fls
	rm -f main.log
	rm -f main.out
	rm -f main.pdf
	rm -f main.synctex.gz
