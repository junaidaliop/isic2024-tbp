.PHONY: setup data folds gbdt vision stack frontier site slides env-export test submit
PY=python

setup:               ## create the conda env + install torch (cu128) + the package
	conda create -y -n isic2024 python=3.12 && \
	conda run -n isic2024 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
	conda run -n isic2024 pip install -e ".[dev]" && \
	conda run -n isic2024 pip install kaggle pre-commit
	@echo "activate with: conda activate isic2024"
	@echo "CPU-only graders: swap the cu128 index for https://download.pytorch.org/whl/cpu"

data:                ## download + unzip the ISIC-2024 SLICE-3D data into data/ (accept rules on Kaggle first)
	kaggle competitions download -c isic-2024-challenge -p data/ && unzip -o data/*.zip -d data/

folds:               ## freeze patient-grouped folds + sanity-check the metric
	$(PY) -m src.cv --meta data/train-metadata.csv --out data/folds.parquet

gbdt:                ## train the tabular backbone, write OOF + pAUC
	$(PY) -m src.gbdt --params configs/gbdt.yaml --out experiments/gbdt_oof.parquet

vision:              ## train one image expert (override CFG=configs/vision/<x>.yaml)
	$(PY) -m src.vision.train_cli --cfg $(CFG)

frontier:            ## render the pAUC-vs-cost Pareto figure
	$(PY) reports/frontier.py --cost gflops

site:                ## render the Quarto deliverables site -> docs/site (run `make frontier` first)
	quarto render site

slides:              ## render the reveal.js deck to PDF
	quarto render site/slides.qmd --to pdf

env-export:          ## refreeze environment.yml from the active conda env (strips the prefix line)
	conda env export --no-builds | grep -v '^prefix:' > environment.yml

test:
	pytest -q

submit:              ## build submission.csv from frozen GBDT boosters (GBDT-only by default)
	$(PY) -m src.submit --cfg configs/default.yaml --out submission.csv
