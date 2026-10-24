# TasteNet evaluation

Reference: **human gold set (n=28)**. Generated 2026-10-24 15:13.

> **Caveat:** the gold set is small (n=28: 9 Indian, 19 Italian) and labelled by a single human labeller. Per-dim 95% bootstrap CIs are wide (often ±0.3 ρ): treat differences between methods smaller than the CI width as noise. The labeller's scale also runs lower than the rubric's on richness/umami/creaminess/sour (see reports/gold_vs_consensus.md), which inflates MAE for every method but not Spearman.

## Spearman ρ (95% bootstrap CI) — n=28, single human labeller

| dim | deepseek-v4-flash (n=28) | gpt-oss-120b (n=28) | qwen3-235b (n=28) | 3-model vote (n=28) | lexicon (n=28) | TasteNet raw (n=28) | TasteNet calibrated (n=28) | vote calibrated (n=28) |
|---|---|---|---|---|---|---|---|---|
| spice | 0.69 [0.42, 0.85] | 0.47 [0.10, 0.75] | 0.60 [0.29, 0.83] | 0.63 [0.34, 0.83] | 0.33 [0.11, 0.58] | 0.51 [0.18, 0.76] | 0.46 [0.12, 0.70] | 0.54 [0.22, 0.74] |
| sweet | 0.64 [0.31, 0.88] | 0.59 [0.26, 0.83] | 0.62 [0.26, 0.91] | 0.65 [0.33, 0.90] | 0.51 [0.19, 0.75] | 0.73 [0.43, 0.91] | 0.62 [0.28, 0.86] | 0.48 [0.09, 0.83] |
| sour | 0.49 [0.09, 0.80] | 0.54 [0.21, 0.77] | 0.36 [-0.06, 0.68] | 0.49 [0.10, 0.78] | 0.57 [0.25, 0.77] | 0.51 [0.12, 0.77] | 0.25 [-0.11, 0.57] | 0.37 [-0.04, 0.65] |
| bitter | 0.32 [-0.19, 0.73] | 0.22 [-0.21, 0.59] | 0.35 [-0.06, 0.72] | 0.40 [-0.04, 0.75] | 0.17 [-0.21, 0.49] | 0.44 [-0.00, 0.78] | 0.42 [-0.02, 0.77] | 0.05 [-0.39, 0.49] |
| umami | 0.43 [0.03, 0.71] | 0.61 [0.28, 0.80] | 0.39 [-0.05, 0.68] | 0.49 [0.11, 0.75] | 0.50 [0.16, 0.76] | 0.52 [0.16, 0.74] | 0.38 [-0.05, 0.68] | 0.29 [-0.17, 0.62] |
| richness | 0.55 [0.17, 0.83] | 0.62 [0.37, 0.81] | 0.49 [0.15, 0.73] | 0.56 [0.23, 0.80] | -0.05 [-0.40, 0.31] | 0.31 [-0.08, 0.65] | 0.15 [-0.24, 0.50] | 0.43 [0.07, 0.69] |
| crunch | 0.77 [0.54, 0.90] | 0.66 [0.32, 0.84] | 0.61 [0.33, 0.83] | 0.76 [0.50, 0.90] | 0.58 [0.30, 0.78] | 0.65 [0.33, 0.84] | 0.49 [0.12, 0.74] | 0.62 [0.32, 0.80] |
| creaminess | 0.56 [0.20, 0.81] | 0.51 [0.09, 0.81] | 0.49 [0.08, 0.79] | 0.56 [0.18, 0.83] | -0.02 [-0.43, 0.40] | 0.53 [0.11, 0.82] | 0.43 [-0.00, 0.74] | 0.39 [-0.04, 0.72] |
| **mean** | 0.56 [0.42, 0.67] | 0.53 [0.38, 0.64] | 0.49 [0.34, 0.61] | 0.57 [0.43, 0.68] | 0.32 [0.17, 0.47] | 0.53 [0.37, 0.65] | 0.40 [0.22, 0.54] | 0.39 [0.24, 0.53] |

## MAE (0-10 scale, lower is better) — n=28, single human labeller

| dim | deepseek-v4-flash | gpt-oss-120b | qwen3-235b | 3-model vote | lexicon | TasteNet raw | TasteNet calibrated | vote calibrated |
|---|---|---|---|---|---|---|---|---|
| spice | 1.32 | 1.68 | 1.11 | 1.34 | 0.91 | 1.21 | 0.97 | 0.91 |
| sweet | 1.14 | 1.29 | 1.07 | 1.07 | 1.95 | 1.35 | 1.56 | 1.14 |
| sour | 1.61 | 1.68 | 1.57 | 1.55 | 1.88 | 1.75 | 0.99 | 0.88 |
| bitter | 0.71 | 0.75 | 0.86 | 0.70 | 1.50 | 0.78 | 0.62 | 0.68 |
| umami | 2.29 | 2.39 | 2.25 | 2.09 | 1.49 | 2.35 | 1.33 | 1.24 |
| richness | 2.25 | 2.61 | 2.29 | 2.38 | 2.63 | 2.31 | 1.48 | 1.41 |
| crunch | 1.21 | 1.61 | 1.71 | 1.27 | 1.68 | 1.38 | 1.30 | 1.07 |
| creaminess | 1.89 | 2.36 | 2.04 | 2.02 | 2.95 | 1.98 | 1.72 | 1.05 |
| **mean** | 1.55 | 1.79 | 1.61 | 1.55 | 1.87 | 1.64 | 1.25 | 1.05 |

## % within ±1 — n=28, single human labeller

| dim | deepseek-v4-flash | gpt-oss-120b | qwen3-235b | 3-model vote | lexicon | TasteNet raw | TasteNet calibrated | vote calibrated |
|---|---|---|---|---|---|---|---|---|
| spice | 57 | 43 | 64 | 50 | 68 | 46 | 57 | 64 |
| sweet | 75 | 68 | 82 | 79 | 50 | 39 | 50 | 54 |
| sour | 54 | 50 | 50 | 46 | 43 | 25 | 61 | 79 |
| bitter | 93 | 89 | 86 | 89 | 46 | 79 | 86 | 82 |
| umami | 39 | 39 | 36 | 43 | 64 | 29 | 50 | 57 |
| richness | 39 | 29 | 43 | 36 | 29 | 25 | 43 | 46 |
| crunch | 75 | 54 | 54 | 71 | 50 | 46 | 50 | 64 |
| creaminess | 50 | 46 | 54 | 46 | 32 | 46 | 46 | 61 |
| **mean** | 60 | 52 | 58 | 58 | 48 | 42 | 55 | 63 |

## Cost and runtime (not gold-based: from this run's token logs and timings)

| method | $ per 1k dishes | runtime LLM calls / dish | latency s / dish |
|---|---|---|---|
| qwen3-235b | 0.104 | 0.10 | 1.302 |
| deepseek-v4-flash | 0.061 | 0.10 | 0.677 |
| gpt-oss-120b | 0.129 | 0.10 | 0.445 |
| 3-model vote | 0.514 | – | – |
| TasteNet | 0.000 | 0.00 | 0.001 |
| lexicon | 0.000 | 0.00 | – |

## Indian Food 101 sanity check (n=148, AUROC; not gold-based: dataset flavour labels)

| method | spice vs 'spicy' | sweet vs 'sweet' |
|---|---|---|
| lexicon | 0.654 | 0.959 |
| TasteNet raw | 0.906 | 0.951 |
| 3-model vote | 0.914 | 0.952 |

## Interpretation

1. Best mean Spearman vs gold: **3-model vote** (0.57).
2. TasteNet raw is 0.04 ρ below the 3-model vote it was distilled from, with zero runtime LLM calls.
3. predict.py serves raw TasteNet scores. Calibration is reported (cross-fitted) but switched off for serving (config.SERVE_CALIBRATION=False).
4. Learning adds +0.20 ρ over the hand-written lexicon baseline (0.32).
5. Easiest dim for TasteNet raw: sweet (0.73); hardest: richness (0.31).
