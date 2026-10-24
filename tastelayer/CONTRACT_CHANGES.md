# Proposed contract additions (all optional; need team sign-off)

Nothing below changes an existing field. Teammates can ignore all of it.

## Dish: new optional fields
| field | type | purpose |
|---|---|---|
| `taste_confidence` | `{dim: 0-1}` | per-dim confidence, `1 - min(std/3, 1)` across tagging models |
| `taste_std` | `{dim: float}` | per-dim std across tagging models |
| `ingredients_canonical` | `[str]` | normalised ingredient names (synonyms merged) |
| `allergens_suggested` | `[str]` | ML-suggested allergens **for human review only**. Filters never read it, and it never flips `verified` |
| `taste_model_version` | `str` | e.g. `tastenet-v1` |
| `taste_verified` | `bool` (default false) | a human checked the **taste vector** (e.g. gold set). Separate from `verified` |
| `is_mock` | `bool` | true for stand-in Uber Eats menu dishes used before real menus land |

## Dish: new `tagged_by` value
- `nebius:consensus`: per-dim median of up to 3 Nebius models

## Artifacts (data/model owner)
`data/taste_dimensions.json` (repo root, single source for sliders and rubric), `artifacts/ingredient_vocab.json`, `artifacts/ingredient_emb.npy`, `artifacts/taste_directions.json`, `artifacts/tastenet.joblib`, `artifacts/calibrators.joblib`

## Note for frontend
`taste_dimensions.json` adds a numeric `score` per anchor (low 2.5, mid 5, high 8.5) so slider tick marks line up with the tagging rubric.

## Decision: `verified` vs `taste_verified` (proposed to team)
- `verified` keeps its brief meaning for the filters: **a human checked allergens + dietary flags**.
- New optional `taste_verified` means a human checked the taste vector.
- Rule: model or LLM taste predictions **never change `verified`**. They pass it through untouched, never set it true, and never clear a human's true.

## UserProfile: proposed optional fields (recommender)
| field | type | purpose |
|---|---|---|
| `strict` | `bool` (default true) | true: dishes with unknown/unverified dietary or allergen info are excluded. false: kept with a warning |
| `ingredient_centroid` | `[float]` | running mean ingredient embedding of liked dishes, for novelty and "take me further" |
| `liked_ingredients` | `[str]` | ingredients from dishes rated >= 4, for explanations |
| `persona` | `str` | label for synthetic test personas only |

Demo note: with strict=false a vegan can be shown "Chicken Vindaloo" with only an "unverified: vegan" warning. For the demo, keep strict=true and verify the dietary flags on the real menus.
