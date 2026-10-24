# TasteLayer: data + model pipeline. `make all` runs end to end (idempotent, cached).
PY ?= .venv/bin/python
include Makefile.data
include Makefile.tag
include Makefile.model

.PHONY: all gold gold-build test
all: ingest tag-ingredients tag-corpus consensus second-pass consensus-final embed train eval predict

gold:        ## blind labelling tool on :8001
	$(PY) gold/app.py
gold-build:  ## merge labels -> gold/gold.json
	$(PY) gold/build_gold.py
test:
	$(PY) -m pytest tests -q

consensus-final:  ## re-run consensus after the review pass
	$(PY) -m tastelayer.tagging consensus

gold-compare: ## human gold vs model consensus -> reports/gold_vs_consensus.md
	$(PY) gold/compare_consensus.py

fetch-models: ## one-time download of bge-base so TasteNet serving is fully offline
	$(PY) -c "from tastelayer.model.features import load_bge; load_bge(allow_download=True); print('bge cached')"
