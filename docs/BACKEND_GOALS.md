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
| 7 | Switch to real data when Role B ships `data/dishes.json` and re-run `data/scripts/compare_epicure.py` | blocked on B |

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
