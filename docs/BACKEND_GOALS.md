# Backend goals (Role C)

## Goal

A FastAPI backend that turns the survey into a taste profile and returns safe, explained
restaurant + dish recommendations, fast enough and robust enough for a live 3-minute demo.
Suggestions come from one deterministic scorer that combines the 8 taste dimensions with
**Epicure** flavour vectors; an LLM (Nebius Token Factory) only rewrites the explanation text
and never decides what is safe to eat.

## Definition of done

- Every endpoint in BRIEF 4.5 works on mock data and on `DATA_DIR=data`.
- Hard filters are plain code: a listed allergen or a `"no"` dietary flag never reaches a user;
  `"unknown"` always produces a warning. Persona tests prove it.
- Epicure loads from a copy committed to the repo. No network, GPU or API key is needed, and if
  the file is missing the backend still runs on taste scores alone.
- The LLM is optional: with no `NEBIUS_API_KEY` the template explanations are used, and
  with a key, answers are cached on disk so a repeated demo makes no new calls.
- `pytest` passes, and `backend/mocks/` is regenerated for the frontend.

## Milestones

| # | Milestone | Status |
|---|---|---|
| 1 | Scaffold: endpoints, hard filters, profile updates, template explanations | done |
| 2 | **Epicure**: committed model copy, centred dish vectors built at startup, user flavour vector from ratings and dislikes, "because you loved X" reasons, stretch pick, more varied survey dishes | done |
| 3 | LLM wrapper for Nebius Token Factory with cache + template fallback | done (waiting on API key) |
| 4 | Persona harness: 8-10 personas with safety and taste assertions | 8 personas (vegan, halal, nut allergy, vegetarian with dislikes, gluten allergy); safety tests pass. Taste assertions next |
| 5 | Recommender tuning: confidence-weighted taste match, per-dimension rating updates, normalised vibe/occasion, dish variety per restaurant | next |
| 6 | Demo hardening: JSON-file profile persistence, preloaded demo users, precomputed LLM explanations (batch API), deploy | later |
| 7 | Switch to real data when Role B ships `data/dishes.json` and re-run `data/scripts/compare_epicure.py` | blocked on B (TasteLayer stand-in catalogue already runs: `DATA_DIR=data/tastelayer`) |
| 8 | Merge TasteLayer (Role B taste model): one anchor file, its optional contract fields, strict mode, catalogue sync | done |
| 9 | Exclusion-only safety rules over dish text (`backend/safety.py`), needed because menus rarely state dietary/allergen info | done |
| 10 | Uber Eats import: `POST /profile/:id/import/uber` warm-starts taste, flavour and price from order history | done |

## Epicure decisions

- **Which model:** we use `Kaikaku/epicure-chem`, pinned at revision `2461ef3f`.
  `data/scripts/compare_epicure.py` on the 36 mock dishes:

  | model | raw mean cosine | nearest dish same family | family gap | cuisine gap |
  |---|---|---|---|---|
  | core | 0.76 | 0.44 | 0.439 | 0.396 |
  | **chem** | 0.53 | **0.53** | 0.378 | 0.304 |
  | cooc | 0.56 | 0.47 | 0.401 | 0.302 |

  Chem finds a same-family nearest dish most often and is the least bunched together. Re-run on the real dishes.
- **Centring:** raw dish vectors are all similar to each other (mean cosine 0.53-0.76), so the backend
  subtracts the catalogue mean before any cosine. Without this the Epicure term barely separates dishes.
- **Where it runs:** at backend startup, from `data/epicure/epicure-chem.npz` (embeddings and vocab only,
  about 2 MB, CC BY 4.0). It is numpy only and loads in well under a second.
- **What we do not use it for:** ingredient-level "similar to your dislike" warnings. Neighbours are too
  noisy for user-facing claims (coriander's nearest is cumin, then tilapia). Dislikes only pull the
  flavour vector away, softly. Epicure's taste directions also barely track our tagged scores
  (mean Spearman rho about 0.1), so they are not used to check Role B's tags.

## Contract changes to agree with the team

Both are optional fields with defaults, so existing frontend code keeps working:

- `Recommendation.kind`: `"match"` (default) or `"stretch"`. The stretch pick is one
  extra card: a dish from a family the user has not rated, chosen by Epicure similarity.
- `Recommendation.because_you_liked`: name of the user's highly rated dish that is closest
  in Epicure space, or `null`.

## Runtime LLM (once the key is added)

```
NEBIUS_API_KEY=...                                         # in .env or the deploy settings, never committed
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-fast    # optional override
```

The LLM gets only facts the code computed (matched tastes, the dish the user loved, price/vibe
fit, warnings) and rewrites them into one friendly sentence. Warnings are always shown verbatim,
never paraphrased.

## Integration decisions (TasteLayer + Uber Eats)

- **One app API.** `backend/` serves the app on :8000. `tastelayer/` is Role B's offline taste pipeline; its own
  `POST /taste` API moved to :8002. Its recommender (`tastelayer/src/tastelayer/reco`) is not used by the app; its
  ideas (strict mode, allergen synonyms, whole-word dislikes) are in `backend/`.
- **Catalogue hand-off.** `data/scripts/sync_tastelayer.py` validates TasteLayer's output against `schema/models.py`
  and writes `data/tastelayer/`. Taste scores come from TasteNet; dietary flags, allergens and `verified` pass through untouched.
- **Safety on unverified menus.** Every TasteLayer dish has dietary flags `"unknown"` and no listed allergens.
  `backend/safety.py` reads name, description and ingredients and can only exclude ("chicken" -> not vegetarian,
  "pistachio" -> nuts, "pepperoni" -> not halal, meat with dairy -> not kosher). Plant milks are not dairy.
  `strict: true` on a profile excludes anything still unknown or unverified; it stays opt-in until Role B verifies the menus,
  because otherwise allergy users get no results.
- **Uber Eats import.** Reads only the Eats order CSV from the export, in memory. Taste: TasteLayer's lexicon on the dims it
  matched, plus the tagged taste of catalogue dishes whose names share a rare word. Flavour: Epicure ingredients
  named in items, plus the ingredients of name-matched catalogue dishes, stored as `ordered_ingredients`, which count
  like one 5-star dish in total. Price: mean unit price -> tier. The raw file is never stored; a synthetic sample lives
  in `backend/fixtures/`.
