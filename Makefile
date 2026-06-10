.PHONY: all install download preprocess disagreement train evaluate figures \
        fusion disagree-aware extras clean test \
        pstat232 pstat232-main pstat232-calibration pstat232-inference \
        pstat232-tables pstat232-figures pstat232-report

PYTHON ?= python

# Category and settings for the PSTAT 232 (computational statistics) pipeline.
CATEGORY ?= All_Beauty
SEED ?= 42
TAU ?= 0.5
NBOOT ?= 2000

all: download preprocess disagreement train evaluate figures

install:
	pip install -r requirements.txt

download:
	$(PYTHON) scripts/01_download_data.py

preprocess:
	$(PYTHON) scripts/02_preprocess_data.py

disagreement:
	$(PYTHON) scripts/03_define_disagreement.py

train: train-text train-metadata train-multimodal

train-text:
	$(PYTHON) scripts/04_train_text_model.py

train-metadata:
	$(PYTHON) scripts/05_train_metadata_model.py

train-multimodal:
	$(PYTHON) scripts/06_train_multimodal_model.py

evaluate:
	$(PYTHON) scripts/07_evaluate_models.py

figures:
	$(PYTHON) scripts/08_generate_figures.py

# Extra analyses (not part of `all`): regenerate fusion comparison and
# disagreement-aware / selective-abstention tables used in the paper.
fusion:
	$(PYTHON) scripts/09_fusion_comparison.py

disagree-aware:
	$(PYTHON) scripts/10_disagreement_aware.py

extras: fusion disagree-aware

multi-category:
	$(PYTHON) scripts/run_all_categories.py

multi-seed:
	$(PYTHON) scripts/run_experiments.py

test:
	$(PYTHON) -m pytest tests/ -v

# ----------------------------------------------------------------------------
# PSTAT 232 (computational statistics) pipeline. Runs offline from the
# materialized pools/embeddings in data_pool/ (no download/encoding needed).
# ----------------------------------------------------------------------------
pstat232: pstat232-tables pstat232-figures

pstat232-main:
	$(PYTHON) scripts/pstat232_leakage_controlled_main.py --category $(CATEGORY) --seed $(SEED)

pstat232-calibration:
	$(PYTHON) scripts/pstat232_group_conditional_calibration.py --category $(CATEGORY) --tau $(TAU)

pstat232-inference:
	$(PYTHON) scripts/pstat232_resampling_inference.py --category $(CATEGORY) --n-boot $(NBOOT) --seed $(SEED)

pstat232-tables: pstat232-main pstat232-calibration pstat232-inference

pstat232-figures:
	$(PYTHON) scripts/pstat232_make_figures.py --category $(CATEGORY) --tau $(TAU)

pstat232-report:
	cd paper && pdflatex -interaction=nonstopmode pstat232_report.tex && pdflatex -interaction=nonstopmode pstat232_report.tex

clean:
	rm -rf models/ data/interim/ data/processed/ data/raw/
	rm -rf reports/figures/*.png reports/tables/*.csv
	rm -rf paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
