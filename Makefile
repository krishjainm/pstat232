.PHONY: all install download preprocess disagreement train evaluate figures paper clean test

PYTHON ?= python

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

multi-category:
	$(PYTHON) scripts/run_all_categories.py

multi-seed:
	$(PYTHON) scripts/run_experiments.py

test:
	$(PYTHON) -m pytest tests/ -v

paper:
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

clean:
	rm -rf models/ data/interim/ data/processed/ data/raw/
	rm -rf reports/figures/*.png reports/tables/*.csv
	rm -rf paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
