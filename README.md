# NoeticxNebius

A portable taste-profile and recommendation layer: a short survey learns a person's food taste, then we recommend restaurants and specific dishes from 20 curated London restaurants (10 Italian, 10 Indian). The full plan and the data contracts are in [docs/BRIEF.md](docs/BRIEF.md).

## Layout and owners

```
data/                      Role B (data)
  taste_dimensions.json    8 taste dimensions + slider anchors (same scale as tagging rubric)
  restaurants.json         real data, filled by B (empty for now)
  dishes.json              real data, filled by B (empty for now)
  dish_embeddings.json     optional Epicure vector per dish (the backend now builds these itself)
  epicure/                 pinned slim copy of Epicure-Chem (embeddings + vocab, ~2 MB), from export_epicure.py
  gold_set.json            50 hand-labelled dishes for the Claude vs open-source comparison
  epicure_ingredient_map.json   menu ingredient name -> Epicure vocab name
  mock/                    6 restaurants / 36 dishes everyone can build against today
  scripts/                 epicure.py loader, explore_epicure.py, build_dish_embeddings.py,
                           compare_epicure.py (pick a sibling), export_epicure.py (refresh data/epicure/)
schema/                    Shared contracts. Change only after the team agrees.
  models.py                Pydantic: Dish, Restaurant, UserProfile, API bodies
  validate.py              python -m schema.validate data/mock
backend/                   Role C (recommender + API)
  main.py                  FastAPI, every endpoint in BRIEF 4.5
  recommender.py           hard filters, profile updates, scoring, stretch pick
  flavour.py               Epicure flavour space: loads data/epicure/, centred dish vectors
  llm.py                   optional Nebius Token Factory rewrite of explanations (cached, template fallback)
  personas/personas.json   synthetic test users (seeded into the API)
  mocks/                   recorded API responses for the frontend
  make_mocks.py            regenerates mocks/ from the live endpoints
tests/                     python -m pytest (persona safety, Epicure, LLM fallback, API flow)
docs/BACKEND_GOALS.md      backend goal, milestones and Epicure decisions
frontend/                  Role A (React + Vite, iPhone frame)
  src/api.js               the only file that calls the backend; uses mocks/ until VITE_API_URL is set
```

## Run it

Python 3.11+ and Node 18+.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
uvicorn backend.main:app --reload
```

API docs are then at http://127.0.0.1:8000/docs. Set `DATA_DIR=data` to switch from mock to real data.
`python -m pytest` runs the backend tests.

Optional LLM explanations: set `NEBIUS_API_KEY` (and optionally `NEBIUS_MODEL`) in the environment or a
git-ignored `.env`. Without a key the backend uses template explanations; nothing else changes.

```bash
cd frontend && npm install && npm run dev
```

The frontend runs on mocks by default. To use the live API: `VITE_API_URL=http://127.0.0.1:8000 npm run dev`.

## How the three roles connect

```
B: menus -> LLM structuring -> Epicure mapping -> taste tags -> human allergen check
   └── data/*.json ──> C: backend loads files, scores, serves API ──> A: screens
   └── taste_dimensions.json (slider anchors) ─────────────────────────> A
```

- **B -> C:** `data/restaurants.json`, `data/dishes.json`, `data/dish_embeddings.json`. Run `python -m schema.validate data` before pushing. Ship 5 dishes first, then 30-50, then everything.
- **C -> A:** the API, plus `backend/mocks/`. After changing a response shape, run `python -m backend.make_mocks` and commit the mocks.
- **Everyone:** a change to `schema/models.py` or `data/taste_dimensions.json` needs a message in the team chat first.

## Ground rules

- Allergens and dietary restrictions are decided by code over verified data, never by an LLM. `"unknown"` is never treated as safe.
- Epicure runs from the committed copy in `data/epicure/` (numpy only, no network or API key; loads in under a second).
  The backend builds dish vectors at startup, so they always match `dishes.json`. If the file is missing it
  logs a warning and recommends on taste scores alone. See `docs/BACKEND_GOALS.md` for why it is Epicure-Chem.
- Branches: `a/<thing>`, `b/<thing>`, `c/<thing>`. Merge to `main` only when validation passes.

## Additions to the brief's contracts

- `GET /taste-dimensions` and `GET /profile/:id` were added so the frontend reads anchors and profiles from one place.
- `budget` in `POST /recommend` is a max `price_tier` (1-4).
- `allergens` and `allergies` use the UK FSA 14-allergen list (`nuts` = tree nuts, `peanuts` separate). See `schema/models.py`.
- The occasion list in `backend/recommender.py` is a placeholder until the team agrees on one.
- `Recommendation.kind` (`"match"` default, or `"stretch"`) and `Recommendation.because_you_liked` (dish name or
  `null`). `/recommend` returns up to 5 matches, then at most one stretch card.

## Credits

Epicure ingredient embeddings by KAIKAKU.AI (Radzikowski and Chen, [arXiv:2605.22391](https://arxiv.org/abs/2605.22391)), licensed CC BY 4.0. The backend uses `Kaikaku/epicure-chem` (revision `2461ef3f`), copied into `data/epicure/`; `data/scripts/epicure.py` is their loader.
