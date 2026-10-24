# TasteLayer: pitch numbers (claims we can honestly make)

Generated 2026-10-24 from this run. Sources: `reports/eval.md`, `reports/eval_qwen_alt.md`, `reports/feature_cv_bge_only.md`, `reports/gold_vs_consensus.md`, `reports/tagging_summary.md`, `artifacts/tastenet.joblib`. Gold-set numbers are **n=28, single human labeller**.

**Served model:** TasteNet, an MLP over bge-base text embeddings, our own skip-gram ingredient embeddings, FlavorGraph ingredient embeddings and a hand-written lexicon. It is `artifacts/tastenet.joblib` and it runs **fully offline**.

## 1. We trained our own model
- **What it does:** TasteNet maps a dish's name, description and ingredients to 8 taste scores (0–10).
- **Training data:** 36,808 Indian and Italian dishes (Uber Eats menus, Food.com, Archana's Kitchen, Indian Food 101).
- **Labels:** consensus of 3 open LLMs on Nebius (Qwen3-235B, gpt-oss-120b, DeepSeek-V4-Flash), each sampled at temperatures 0.2 and 0.7. A Qwen review pass covered all 1,235 dishes where the models disagreed.
- **At runtime:** no LLM calls and no network calls. The bge-base encoder loads from the local cache. A test (`tests/test_offline.py`) predicts a never-seen dish with every network connection blocked and asserts that zero connections are attempted.

## 2. TasteNet vs the LLM consensus, on dishes it never saw in training (held-out set)
| | n | mean Spearman ρ (8 dims) | MAE (0–10) |
|---|---|---|---|
| **TasteNet (served) vs consensus, all held-out dishes** | 4,082 | **0.81** [0.80, 0.81] | 0.62 |
| TasteNet (served) vs consensus, dishes tagged by all 3 LLMs | 3,181 | 0.81 | – |
| DeepSeek-V4-Flash vs median of the other two LLMs | 3,181 | 0.83 | – |
| Qwen3-235B vs median of the other two LLMs | 3,181 | 0.83 | – |
| gpt-oss-120b vs median of the other two LLMs | 3,181 | 0.79 | – |

**Claim:** "On 4,000 held-out dishes, the offline TasteNet reproduces the 3-LLM consensus at ρ = 0.81. Each LLM agrees with the other two at 0.79–0.83."

The comparison isn't perfectly like-for-like. TasteNet is scored against a consensus that includes all three models, while each LLM is scored against the other two only.

Per dimension (held-out ρ): spice 0.86, sweet 0.74, sour 0.85, **bitter 0.59**, umami 0.87, richness 0.86, crunch 0.82, creaminess 0.88.

## 3. TasteNet vs human gold labels (n=28, single human labeller)
| method | mean Spearman ρ [95% bootstrap CI] |
|---|---|
| 3-model LLM vote | 0.57 [0.43, 0.68] |
| DeepSeek-V4-Flash | 0.56 [0.42, 0.67] |
| gpt-oss-120b | 0.53 [0.38, 0.64] |
| **TasteNet (served, offline, uncalibrated)** | **0.53 [0.37, 0.65]** |
| Qwen3-235B | 0.49 [0.34, 0.61] |
| hand-written lexicon baseline | 0.32 [0.17, 0.47] |

Paired bootstrap on the same 28 dishes (2,000 resamples):
- **TasteNet − lexicon:** +0.20 [+0.07, +0.34]. Learning clearly beats the hand-written rules.
- **3-model vote − TasteNet:** +0.04 [−0.05, +0.13]. We cannot tell TasteNet and the LLM vote apart on this gold set.

**Claim:** "Against a human labeller, the offline TasteNet ranks dishes about as well as the LLMs it was trained from (ρ 0.53 vs 0.49–0.57). It clearly beats a hand-written ingredient lexicon (+0.20 ρ), and it makes no LLM or network calls at runtime."

**Don't claim:**
- That TasteNet beats the LLMs.
- Any MAE figure against gold.
- Anything per dimension against gold (CIs are about ±0.3).

**Sanity check (Indian Food 101 flavour labels, n=148, not gold-based):** TasteNet separates dishes the dataset marks "spicy" with AUROC 0.91 (3-model vote 0.91, lexicon 0.65), and "sweet" with AUROC 0.95.

## 4. Cost per 1,000 dishes
| method | $ per 1k dishes | runtime LLM calls per dish | network calls per dish |
|---|---|---|---|
| 3-model vote (3 LLMs × 2 samples, as built) | $0.51 (actual spend over dishes tagged; full coverage of all 3 × 2 would be ≈ $0.59) | 0.6 requests (10 dishes per request) | yes |
| single LLM, one sample at one temperature (DeepSeek / Qwen / gpt-oss) | $0.06 / $0.10 / $0.13 | 0.1 requests | yes |
| **TasteNet (served)** | **$0** (local CPU only) | **0** | **0** |

- **Total build spend on Nebius: $23.45.** That breaks down as gpt-oss $10.89, Qwen3-235B $6.99, DeepSeek $5.53 and Qwen3-Embedding $0.03, and includes ingredient tagging, the review pass and tests. The budget was $60.
- **TasteNet latency:** about 1 ms per dish once its text embedding is cached. A new dish also needs one local bge encode.

## 5. Alternative (not served): TasteNet with Qwen3-Embedding features
| TasteNet variant | held-out ρ vs consensus | gold ρ (n=28, single labeller) | runtime network calls |
|---|---|---|---|
| **bge only, MLP (served)** | 0.81 [0.80, 0.81] | **0.53 [0.37, 0.65]** | **none** |
| bge + Qwen3-Embedding, ridge (`artifacts/tastenet_qwen.joblib`, local only; not in the repo) | 0.84 [0.83, 0.84] | 0.50 [0.33, 0.63] | 1 Qwen3-Embedding API call per unseen dish |

- The Qwen variant matches the LLM consensus better on held-out dishes (+0.03 ρ). On the human gold set the two can't be told apart.
- We serve the offline model so that the "no runtime calls" claim is literally true.
- Full report for the alternative: `reports/eval_qwen_alt.md`.
- To switch back, rebuild it with `make train` (needs the corpus and consensus labels regenerated first, plus about $0.03 of Qwen embedding calls). It writes `tastenet.joblib`; then rerun `make predict`. The runtime claim then becomes "one embedding call per new dish".

## Limitations
- **Small gold set:** 28 dishes, one labeller (9 Indian, 19 Italian). 22 of 50 candidates were skipped, mostly Indian. Bootstrap CIs are wide, and nothing per dimension against gold is reliable.
- **The labeller's scale differs from the rubric:** scores run 1.4–2.2 points lower on richness, umami, creaminess and sour, and about 1 point higher on spice. So MAE against gold mostly measures scale use, not taste accuracy. Calibration to this scale is reported but deliberately switched off for serving.
- **TasteNet learns the LLM consensus, not ground truth:** the held-out ρ of 0.81 measures how well it imitates the LLMs. It inherits their biases. For example, Qwen rated black pepper 7 for spice against a rubric anchor of 2–3.
- **Bitter is the weak dimension:** held-out ρ 0.59, and the LLMs agree with each other least on bitter (mean pairwise ρ 0.49 across 32,101 fully tagged dishes, against 0.72–0.88 on the other dimensions). Expect bitter scores to be noisy.
- **The offline model is slightly weaker on the LLM-consensus benchmark:** 0.81 vs 0.84 for the Qwen variant. That's the cost of having no runtime network calls.
- **Tagging was cut off for time, so coverage is partial:**
  - 32,101 dishes were tagged by 3 models.
  - 8,839 by 2, with confidence lowered to × 0.75.
  - 3,545 by DeepSeek only. These were excluded from training.
  - 22% of the held-out set is 2-model dishes.
- **The menu data is a stand-in:** the 296 "menu" dishes come from 20 US Uber Eats restaurants, with prices converted USD→GBP and vibe tags guessed from price tier. Dietary and allergen flags are all unknown and unverified, so strict-diet users currently get no recommendations by design. Real UK menus must be loaded and their allergens and dietary flags checked by a human before the demo.
- **The recommender hasn't been tested with users:** its quality is only exercised through 9 synthetic personas.
- **The ingredient-embedding anchor check fails on 4 of 8 dimensions:** this is a side feature, not the main signal.
- **One-time setup needs the network:** the bge-base weights (837 MB in the local Hugging Face cache on this machine) must be downloaded once with `make fetch-models` on any new machine. After that, serving is offline.
