# Tagging summary

_updated 2026-10-24 14:29:46_

## Spend
**Total $14.936** (ask at $40, hard stop at $60)

| model | calls | in tokens | out tokens | USD |
|---|---|---|---|---|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 3479 | 8,220,833 | 3,535,403 | 3.765 |
| deepseek-ai/DeepSeek-V4-Flash-0731 | 9098 | 19,064,406 | 10,218,099 | 5.530 |
| openai/gpt-oss-120b | 4388 | 10,129,053 | 6,868,652 | 5.641 |

## Progress (current run: dish, 50.1 min elapsed)

| model | batches done | cached | total | % | batches/min | ETA min |
|---|---|---|---|---|---|---|
| deepseek-v4-flash | 8898 | 0 | 8898 | 100.0 | 177.7 | 0.0 |
| gpt-oss-120b | 4130 | 0 | 8898 | 46.4 | 82.5 | 57.8 |
| qwen3-235b | 3312 | 0 | 8898 | 37.2 | 66.1 | 84.4 |

## Calls

| model                              |   calls |   ok_rate |   truncated |   429s |   p50_latency_s |   usd_per_call |
|:-----------------------------------|--------:|----------:|------------:|-------:|----------------:|---------------:|
| Qwen/Qwen3-235B-A22B-Instruct-2507 |    4727 |    0.7318 |           0 |      0 |            9.18 |         0.0008 |
| Qwen/Qwen3-Embedding-8B            |     465 |    1      |           0 |      0 |            2.62 |         0.0001 |
| deepseek-ai/DeepSeek-V4-Flash-0731 |   12243 |    0.7387 |           0 |      0 |            5.35 |         0.0004 |
| openai/gpt-oss-120b                |    5932 |    0.734  |           3 |      0 |            2.67 |         0.0009 |

## Rate limits (latest headers)

| model | limit req/min | limit tok/min | our rpm target | our tpm target |
|---|---|---|---|---|
| Qwen/Qwen3-235B-A22B-Instruct-2507 | 600 | 400000 | 510 | 340000 |
| deepseek-ai/DeepSeek-V4-Flash-0731 | 3000 | 1000000 | 2550 | 850000 |
| openai/gpt-oss-120b | 600 | 500000 | 510 | 425000 |

## Labels
44,485 dishes with at least one label; 163,390 label rows.

|                                    |   rows |
|:-----------------------------------|-------:|
| ('deepseek-v4-flash', 0.2, 'main') |  44485 |
| ('deepseek-v4-flash', 0.7, 'main') |  44485 |
| ('gpt-oss-120b', 0.2, 'main')      |  20620 |
| ('gpt-oss-120b', 0.7, 'main')      |  20680 |
| ('qwen3-235b', 0.2, 'main')        |  16530 |
| ('qwen3-235b', 0.7, 'main')        |  16590 |

### Inter-model Spearman per dim (labels averaged over temperatures)

|            |   deepseek-v4-flash vs gpt-oss-120b |   deepseek-v4-flash vs qwen3-235b |   gpt-oss-120b vs qwen3-235b |   mean_pair |
|:-----------|------------------------------------:|----------------------------------:|-----------------------------:|------------:|
| spice      |                               0.851 |                             0.891 |                        0.855 |       0.866 |
| sweet      |                               0.704 |                             0.781 |                        0.697 |       0.727 |
| sour       |                               0.786 |                             0.857 |                        0.809 |       0.817 |
| bitter     |                               0.404 |                             0.558 |                        0.403 |       0.455 |
| umami      |                               0.846 |                             0.86  |                        0.845 |       0.85  |
| richness   |                               0.826 |                             0.874 |                        0.857 |       0.852 |
| crunch     |                               0.728 |                             0.8   |                        0.717 |       0.748 |
| creaminess |                               0.881 |                             0.902 |                        0.877 |       0.887 |

### Self-agreement, temperature 0.2 vs 0.7 (Spearman per dim, within ±1 share)

|            |   deepseek-v4-flash |   gpt-oss-120b |   qwen3-235b |
|:-----------|--------------------:|---------------:|-------------:|
| spice      |               0.926 |          0.869 |        0.964 |
| sweet      |               0.838 |          0.741 |        0.924 |
| sour       |               0.902 |          0.826 |        0.958 |
| bitter     |               0.726 |          0.464 |        0.846 |
| umami      |               0.922 |          0.892 |        0.961 |
| richness   |               0.932 |          0.885 |        0.967 |
| crunch     |               0.907 |          0.753 |        0.945 |
| creaminess |               0.945 |          0.902 |        0.972 |
| within_1   |               0.984 |          0.901 |        0.993 |
