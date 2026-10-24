# TasteLayer

Dish taste model: LLM tagging (Nebius) → consensus labels → ingredient embeddings → TasteNet regressor, plus a small recommender (`src/tastelayer/reco`) and a `POST /taste` API.

Self-contained Python package; run everything from inside this `tastelayer/` folder.

## Setup

```bash
cd tastelayer
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # fill in NEBIUS_API_KEY, NEBIUS_BASE_URL, KAGGLE_API_TOKEN, ...
make test
```

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

Other useful targets: `make eval` (writes `reports/eval.*`), `make predict` (rewrites `data/processed/menu_dishes_tagged.json`), `make serve` (API on :8002; the app backend in `backend/` uses :8000), `make gold` (labelling tool on :8001).
