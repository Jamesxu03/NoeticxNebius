# Corpus summary

Total rows: **44485** (before dedupe 84819). Ingredient vocab (freq>=5): **1882**.
Rows with >=1 canonical ingredient: 43662.
What's Cooking rows (co-occurrence only): 0.

## Rows per source x cuisine

| source          |   indian |   italian |   total |
|:----------------|---------:|----------:|--------:|
| archana         |     3798 |       226 |    4024 |
| foodcom         |     2276 |     17151 |   19427 |
| indian_food_101 |      173 |         0 |     173 |
| uber_eats       |     8235 |     12626 |   20861 |

## Mock menu (is_mock=true, from Uber Eats US)

296 dishes across 20 restaurants (tiers assigned by median item price rank).

Dish families: pasta 47, curry 40, pizza 30, biryani 28, meat_main 24, indian_bread 21, antipasti 19, indian_snack 18, dessert 15, salad 14, tandoori 14, dal 10, rice 7, soup 6, drink 3

Venue-round families (restaurant: tier):

- pizza: r_01:T1, r_02:T1, r_03:T2, r_04:T2, r_05:T2, r_06:T3, r_07:T3, r_08:T3, r_09:T4, r_10:T4
- pasta: r_01:T1, r_02:T1, r_03:T2, r_04:T2, r_05:T2, r_06:T3, r_07:T3, r_08:T3, r_09:T4, r_10:T4
- curry: r_11:T1, r_12:T1, r_13:T2, r_14:T2, r_15:T2, r_16:T3, r_17:T3, r_18:T3, r_19:T4, r_20:T4
- biryani: r_11:T1, r_12:T1, r_13:T2, r_14:T2, r_15:T2, r_16:T3, r_17:T3, r_18:T3, r_19:T4, r_20:T4
