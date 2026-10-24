# TasteLayer

Dish taste model: LLM tagging (Nebius) → consensus labels → ingredient embeddings → TasteNet regressor, plus a small recommender (`src/tastelayer/reco`) and a `POST /taste` API.

Self-contained Python package; run everything from inside this `tastelayer/` folder.

## Setup

```bash
cd tastelayer
python3 -m venv .venv && .venv/bin/pip install -e .
make fetch-models      # one-time download of bge-base (needs network); serving is offline afterwards
cp .env.example .env   # only needed for tagging / data download: NEBIUS_API_KEY, NEBIUS_BASE_URL, KAGGLE_API_TOKEN, ...
make test
```

## Served model: offline TasteNet

`artifacts/tastenet.joblib` is TasteNet as an MLP over bge-base text embeddings + ingredient embeddings + lexicon
features. Predicting a dish makes **no LLM and no network calls**: bge-base is loaded from the local Hugging Face
cache only (`local_files_only`), so run `make fetch-models` once per machine. `tests/test_offline.py` blocks all
sockets and checks that predicting a never-seen dish attempts zero connections.

- Results: `reports/eval.md` (gold n=28, single human labeller, bootstrap CIs) and `reports/pitch_numbers.md`.
- Alternative, not served: the same model with Qwen3-Embedding features (`reports/eval_qwen_alt.md`). It scores higher
  against the LLM consensus but needs one Nebius embedding call per unseen dish. `make train` builds it;
  `make train-bge` builds the served offline one.
- Calibration to the gold labels is reported but switched off for serving (`config.SERVE_CALIBRATION = False`).
- Predictions only write taste fields. They never change a dish's `verified` flag (see `CONTRACT_CHANGES.md`).

## What is in the repo vs. what you regenerate

Committed (small, needed by the frontend / for eval):

- `data/processed/menu_dishes_tagged.json`, `menu_dishes.json`, `menu_dishes_by_model.json`, `personas.json`, `restaurants_mock.json`
- `gold/gold.json` and `gold/labels_arda.jsonl` (human gold labels)
- `artifacts/` model files under 50MB (`tastenet.joblib`, `calibrators.joblib`, ingredient embeddings, vocabularies, taste directions)
- `reports/` (eval markdown/csv/png, tagging summary, logs)

Not committed (large or reproducible). Regenerate with `make`:

| Excluded | How to regenerate |
| --- | --- |
| `data/raw/` (Kaggle datasets, ~2GB) | `make download` (needs `KAGGLE_API_TOKEN` in `.env`) |
| `data/processed/corpus.parquet` | `make ingest` |
| `data/interim/` + `data/processed/labels*.parquet`, `ingredient_taste_llm.parquet` | `make tag-ingredients tag-corpus consensus second-pass consensus-final` (Nebius API calls, costs money; capped by `MAX_USD`) |
| `cache/` (LLM response disk cache) | rebuilt automatically during tagging |
| `artifacts/emb_cache/*_vecs.npy` (>50MB text embeddings) | `make train` (or `make train-bge` for no Nebius calls) |
| `data/processed/cooc_extra.parquet` and embedding artifacts | `make embed` |

Full pipeline end to end (idempotent, cached):

```bash
make all   # ingest → tag → consensus → embed → train → eval → predict
```

Tagging with a time limit: `python -m tastelayer.tagging tag-gaps --minutes 20` fills only missing labels (menu dishes,
then high-disagreement dishes, then dishes one model short of all three) and stops sending requests at the cutoff.
Consensus records `n_models` per dish and lowers confidence for dishes tagged by fewer than 3 models
(x0.75 for 2, x0.5 for 1); training uses dishes with at least 2 models.

Other useful targets: `make eval` (writes `reports/eval.*`), `make predict` (rewrites `data/processed/menu_dishes_tagged.json`), `make serve` (API on :8000), `make gold` (labelling tool on :8001), `make gold-compare` (human gold vs consensus report).
