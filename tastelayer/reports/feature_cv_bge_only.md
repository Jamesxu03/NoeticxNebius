# Feature-set comparison (5-fold grouped CV, RidgeCV, n=20000)

| features | #feat | mean Spearman | MAE | spice | sweet | sour | bitter | umami | richness | crunch | creaminess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bge+lex+ing_ours+ing_fg | 1252 | **0.798** | 0.64 | 0.86 | 0.67 | 0.85 | 0.62 | 0.87 | 0.85 | 0.80 | 0.87 |
| lex+ing_ours+ing_fg | 484 | **0.759** | 0.70 | 0.83 | 0.63 | 0.82 | 0.59 | 0.84 | 0.81 | 0.72 | 0.83 |
| bge | 768 | **0.707** | 0.79 | 0.80 | 0.50 | 0.72 | 0.54 | 0.82 | 0.77 | 0.74 | 0.78 |
| lex | 22 | **0.619** | 0.88 | 0.72 | 0.44 | 0.72 | 0.39 | 0.74 | 0.71 | 0.49 | 0.73 |
