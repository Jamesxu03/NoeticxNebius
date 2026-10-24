# Human gold vs 3-model consensus (n=28, single human labeller)

n=28 dishes with consensus labels; annotators: arda. Bias = mean(human - consensus). Labels are reported, never altered.

## Per-dimension agreement (n=28, single human labeller)

| dim | Spearman | MAE | within ±1 | bias | human mean | consensus mean | flag |
|---|---|---|---|---|---|---|---|
| spice | 0.63 | 1.34 | 50% | +0.95 | 2.3 | 1.4 |  |
| sweet | 0.65 | 1.07 | 79% | -0.25 | 2.5 | 2.7 |  |
| sour | 0.49 | 1.55 | 46% | -1.41 | 1.4 | 2.8 | ⚠️ scale differs |
| bitter | 0.40 | 0.70 | 89% | -0.23 | 0.9 | 1.1 | ⚠️ ranking differs |
| umami | 0.49 | 2.09 | 43% | -2.02 | 2.6 | 4.6 | ⚠️ scale differs |
| richness | 0.56 | 2.38 | 36% | -2.23 | 3.2 | 5.4 | ⚠️ scale differs |
| crunch | 0.76 | 1.27 | 71% | -1.05 | 1.9 | 3.0 |  |
| creaminess | 0.56 | 2.02 | 46% | -1.70 | 2.1 | 3.8 | ⚠️ scale differs |

## Dish-level gaps (|diff| >= 3): 45 of 224 cells (n=28, single human labeller)

| dish | cuisine | dim | human | consensus |
|---|---|---|---|---|
| Burrata Bruschetta | italian | creaminess | 0 | 7 |
| Margherita Pizza | italian | umami | 0 | 6 |
| Samosa (2 pcs) | indian | richness | 0 | 5.5 |
| Low Carb Chicken | italian | umami | 0 | 5 |
| Margherita Pizza | italian | richness | 0 | 5 |
| Special Chicken Mumtaz | indian | creaminess | 1 | 6 |
| Garlic Naan | indian | richness | 0 | 4.5 |
| Zuppa De Pesce | italian | umami | 3 | 7.5 |
| Calamari | italian | crunch | 4 | 8 |
| Cheesecake | italian | richness | 5 | 9 |
| Chicken 65 Kebab | indian | crunch | 2 | 6 |
| Chicken Roll | italian | creaminess | 0 | 4 |
| Chicken Roll | italian | richness | 1 | 5 |
| Chicken Tikka Masala | indian | creaminess | 3 | 7 |
| Italian meatball | italian | sour | 0 | 4 |
| New York Cheesecake | italian | richness | 5 | 9 |
| Samosa (2 pcs) | indian | crunch | 4 | 8 |
| Sausage & Peppers | italian | umami | 2 | 6 |
| Spaghetti and Meatballs | italian | sour | 0 | 4 |
| Special Chicken Mumtaz | indian | umami | 2 | 6 |
| Vegetable Biryani | indian | richness | 1 | 5 |
| Cheese Pizza | italian | creaminess | 1 | 4.5 |
| Italian meatball | italian | creaminess | 1 | 4.5 |
| Lentil Soup | indian | umami | 1 | 4.5 |
| Samosa (2 pcs) | indian | sweet | 5 | 1.5 |
| Spaghetti and Meatballs | italian | umami | 3 | 6.5 |
| Special Chicken Mumtaz | indian | richness | 3 | 6.5 |
| Tuscany Salad | italian | crunch | 0 | 3.5 |
| Burrata Bruschetta | italian | richness | 3 | 6 |
| Burrata Bruschetta | italian | spice | 3 | 0 |
| Cheese Pizza | italian | richness | 3 | 6 |
| Cheesecake | italian | bitter | 3 | 0 |
| Chicken 65 Kebab | indian | umami | 2 | 5 |
| Chicken Parmigiana family | italian | richness | 4 | 7 |
| Chicken Roll | italian | sweet | 4 | 1 |
| Garlic Naan | indian | umami | 0 | 3 |
| Lentil Soup | indian | creaminess | 0 | 3 |
| Margherita Pizza | italian | creaminess | 1 | 4 |
| Margherita Pizza | italian | sour | 1 | 4 |
| Nick's Salad | italian | crunch | 0 | 3 |
| Samosa Chaat | indian | creaminess | 2 | 5 |
| Sausage & Peppers | italian | richness | 3 | 6 |
| Spaghetti Pasta | italian | sour | 1 | 4 |
| Tuscany Salad | italian | spice | 4 | 1 |
| Zuppa De Pesce | italian | spice | 4 | 1 |
