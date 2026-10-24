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
  mock/                    6 restaurants / 36 dishes everyone can build against today (default DATA_DIR)
  tastelayer/              20 restaurants / 296 dishes from TasteLayer (stand-in menus, is_mock=true),
                           written by scripts/sync_tastelayer.py. Serve with DATA_DIR=data/tastelayer
  scripts/                 epicure.py loader, explore_epicure.py, build_dish_embeddings.py,
                           compare_epicure.py (pick a sibling), export_epicure.py (refresh data/epicure/),
                           sync_tastelayer.py (tastelayer/data/processed -> data/tastelayer/)
tastelayer/                Role B (taste model): Nebius LLM tagging -> consensus -> TasteNet regressor, gold set,
                           evaluation. Own package and venv; see tastelayer/README.md. Not imported at runtime,
                           except its lexicon.yaml (pure Python) for the Uber Eats import
schema/                    Shared contracts. Change only after the team agrees.
  models.py                Pydantic: Dish, Restaurant, UserProfile, API bodies
  validate.py              python -m schema.validate data/mock
backend/                   Role C (recommender + API)
  main.py                  FastAPI, every endpoint in BRIEF 4.5
  recommender.py           hard filters, profile updates, scoring, stretch pick
  safety.py                exclusion-only text rules: "chicken" -> not vegetarian, "pistachio" -> nuts, etc.
  uber_import.py           warm-start a profile from an Uber Eats data export (reads only the Eats order CSV)
  menu_rank.py             rank a scanned menu: safety rules, TasteNet (if running) or lexicon taste, Epicure flavour
  flavour.py               Epicure flavour space: loads data/epicure/, centred dish vectors
  llm.py                   optional Nebius Token Factory rewrite of explanations (cached, template fallback)
  personas/personas.json   synthetic test users (seeded into the API)
  mocks/                   recorded API responses for the frontend
  make_mocks.py            regenerates mocks/ from the live endpoints
  fixtures/                synthetic Uber Eats order CSV for tests, mocks and demos (not anyone's real data)
tests/                     python -m pytest (persona safety, safety rules, Epicure, Uber import, LLM fallback, API flow)
docs/BACKEND_GOALS.md      backend goal, milestones and Epicure decisions
frontend/                  Role A: "Palatable" (React + Vite). Menu scan (in-browser OCR), taste profile, Uber import,
                           restaurant picks, taste quiz, optional Supabase cloud save (SUPABASE-SETUP.md)
  src/api.js               the only file that calls the backend: /api via the Vite proxy, or VITE_API_URL
  src/screens/             Recommendations.jsx (POST /recommend), TasteSurvey.jsx (survey dishes + venue rounds)
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

API docs are then at http://127.0.0.1:8000/docs. `DATA_DIR` picks the catalogue: `data/mock` (default, 36 dishes),
`data/tastelayer` (296 dishes from TasteLayer) or `data` (the curated London restaurants, once Role B fills them).
`python -m pytest` runs the backend tests. TasteLayer has its own tests: `cd tastelayer && make test`.

Optional LLM explanations: set `NEBIUS_API_KEY` (and optionally `NEBIUS_MODEL`) in the environment or a
git-ignored `.env`. Without a key the backend uses template explanations; nothing else changes.

```bash
cd frontend && npm install && npm run dev
```

The dev server proxies `/api` to the backend on :8000, so start the backend first. For a deployed API, build with
`VITE_API_URL=https://...`. `npm test` runs the frontend's menu-parsing tests. `backend/mocks/` remains a reference
of response shapes (`python -m backend.make_mocks`), but the app no longer reads it.

Optional: run TasteLayer's TasteNet service (`cd tastelayer && make serve`, :8002) and start the backend with
`TASTELAYER_URL=http://127.0.0.1:8002` so scanned Indian/Italian menu items get TasteNet taste scores. Without it the
backend uses the lexicon + catalogue estimate.

## How the three roles connect

```
B: menus -> tastelayer (Nebius tagging, TasteNet) -> sync_tastelayer.py -> human allergen check
   └── data/*.json ──> C: backend loads files, scores, serves API ──> A: screens
   └── data/taste_dimensions.json (slider anchors = tagging rubric) ──> A
User: Uber Eats export ──> POST /profile/:id/import/uber ──> profile warm start
```

- **B -> C:** `data/restaurants.json` and `data/dishes.json` (or `data/tastelayer/` via `sync_tastelayer.py`). Run `python -m schema.validate <folder>` before pushing.
- **C -> A:** the API, plus `backend/mocks/`. After changing a response shape, run `python -m backend.make_mocks` and commit the mocks.
- **Everyone:** a change to `schema/models.py` or `data/taste_dimensions.json` needs a message in the team chat first.

## Ground rules

- Allergens and dietary restrictions are decided by code over verified data, never by an LLM. `"unknown"` is never treated as safe:
  it warns, or excludes for profiles with `strict: true`. `backend/safety.py` adds keyword rules over dish text that can only
  exclude (never mark a dish safe). TasteLayer's `allergens_suggested` is for human review and is never read by the filters.
- The Uber Eats import reads the export in memory, opens only `Eats/eats_order_details*.csv` (or `user_orders`), and keeps
  derived signals only. Never commit anyone's real export: `.gitignore` blocks `Uber Data/` folders and export zips.
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
- `PUT /profile/:id` replaces a whole profile (the browser keeps a copy and restores it after a backend restart).
  New profile ids are random (`u_<10 hex>`), so restored ids never collide.
- `POST /menu/rank` `{profile_id, items: [{id, name, ingredients, price}]}` returns the items sorted, each with
  `score`, `excluded` + `exclusion_reason`, `warnings`, `reason`, `matches`, `because_you_liked`, `taste`, `taste_source`.
- `PUT /profile/:id/tolerances` keeps confidence that ratings or imports already earned (it used to reset it to 0.3).
- `POST /profile/:id/import/uber` (multipart `file`: the Uber data zip or its Eats order CSV) returns
  `{profile, summary}`; `summary` has item/restaurant counts, cuisines, top ingredients, taste shift and price preference.
- From `tastelayer/CONTRACT_CHANGES.md`, adopted as optional fields: Dish `taste_confidence`, `taste_std`,
  `ingredients_canonical`, `allergens_suggested`, `taste_model_version`, `taste_verified`, `is_mock`; UserProfile `strict`.
  UserProfile also gets `ordered_ingredients` (from the Uber import).
- `data/taste_dimensions.json` is the single anchor file, with TasteLayer's rubric scores (low 2.5, mid 5, high 8.5).

## Credits

Epicure ingredient embeddings by KAIKAKU.AI (Radzikowski and Chen, [arXiv:2605.22391](https://arxiv.org/abs/2605.22391)), licensed CC BY 4.0. The backend uses `Kaikaku/epicure-chem` (revision `2461ef3f`), copied into `data/epicure/`; `data/scripts/epicure.py` is their loader.
