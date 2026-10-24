# Uploaded Palatable UI + TasteNet

Run `scripts/install-palatable.sh` once from the repository root, then
`scripts/palatable.sh`. Open http://localhost:8080 on the machine running it.
The server serves this UI and `/api/taste` and `/api/taste/batch` on one origin.
The existing React frontend and backend remain available separately.

The home page predicts the uploaded catalogue's eight taste dimensions with the
committed TasteNet regressor. Its scores drive dish matching and the taste survey.
Expand **Predict a dish’s taste** to predict a new Indian or Italian dish and see
per-dimension confidence. Dietary/allergen information is never generated or
changed by TasteNet. Saved profile information stays in browser local storage.

Inference requires the BAAI/bge-base-en-v1.5 encoder downloaded once into
`HF_HOME` (defaults to `/workspace/.cache/huggingface`). The existing model artifact
requires scikit-learn 1.9.1. Downloads need `huggingface.co`, its subdomains and
`*.hf.co`. If unavailable, the UI explicitly labels its uploaded catalogue scores
and shows prediction errors. It does not claim fallback scores are live inference.
After allowing downloads, run `HF_HOME=/workspace/.cache/huggingface .venv/bin/python scripts/fetch_tastenet.py`
and use **Refresh model predictions**. No paid API or API key is required.

The upload did not include its device adapter, delivery-history importer or food
photos. A browser download/share adapter and illustrated placeholders are included;
the delivery-history entry point links to the survey. The unbuilt service worker
is not registered, so API errors and old predictions are not cached offline.
