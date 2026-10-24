READY-TO-PASTE CLAUDE CODE PROMPTS
Prompt 1 — Master / orchestrator
You are the lead engineer for "TasteLayer", a hackathon project (team of 3, ~4h for data/model). Be ambitious but ship working code every 45 minutes. Use subagents to run independent blocks in parallel (Data, Tagging, Model) while I hand-label a gold set.

PRODUCT: mobile-style web app that (1) learns a user's food taste via survey, (2) shows a shareable taste profile, (3) recommends restaurants + specific dishes from 20 curated restaurants (10 Italian, 10 Indian; ~300-450 dishes). Pitch: portable taste-profile + recommendation layer a delivery platform could plug in. Inspired by Epicure (Radzikowski & Chen, arXiv 2605.22391; code not released): skip-gram/Metapath2Vec ingredient embeddings, linear taste probes, SLERP rotation, NN pairings. We MUST train our own model (not just call an LLM).

FIXED CONTRACTS (do not change; propose additions in CONTRACT_CHANGES.md):
- Taste dims (0-10): spice, sweet, sour, bitter, umami, richness, crunch, creaminess. Anchors low/mid/high:
  spice: black pepper / jalapeno / habanero; sweet: roasted red pepper / caramelised onion / maple syrup;
  sour: natural yoghurt / lemon juice / tamarind; bitter: rocket / 70% dark chocolate / radicchio;
  umami: tomato / parmesan / anchovy or miso; richness: olive-oil dressing / cream sauce / ghee-heavy dal makhani;
  crunch: soft bread / toasted nuts / fried papadum; creaminess: clear broth / hummus / mascarpone.
- Dish: id, restaurant_id, name, description, cuisine (indian|italian), category, dish_family, price_gbp, ingredients[], allergens[], dietary{vegan,vegetarian,halal,kosher: yes|no|unknown}, taste{8 dims}, serving_temp, adventurousness (0-10), tagged_by (claude | nebula:<model> | human), verified bool.
- UserProfile: taste_vector(8), confidence, dish_ratings(1-5), venue choices, price_pref, vibe_pref, adventurousness.
- RULES: ONE deterministic recommender. Hard filters (dietary, allergens, disliked ingredients) are pure code over verified data, never ML/LLM at runtime; unknown => exclude or warn. LLMs only offline (tagging, explanations, optional occasion->weights), cached.

PROPOSED ADDITIONS (write to CONTRACT_CHANGES.md, keep optional): tagged_by values model:tastenet-v1 and consensus:<models>; fields taste_confidence{8}, taste_std{8}, ingredients_canonical[], allergens_suggested[] (human review only, never used by filters), taste_model_version; artifacts ingredient_vocab.json, ingredient_emb.npy, taste_directions.json, tastenet.joblib, calibrators.joblib.

REPO:
tastelayer/
  pyproject.toml (python>=3.11; pandas, pyarrow, pydantic>=2, openai>=1.40, httpx, tenacity, aiolimiter, diskcache, numpy, scipy, scikit-learn, gensim, sentence-transformers, torch, joblib, rapidfuzz, fastapi, uvicorn, rich, typer)
  .env.example: NEBIUS_API_KEY=, NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/ (confirm in Nebius dashboard), ANTHROPIC_API_KEY=, KAGGLE_USERNAME=, KAGGLE_KEY=, SEED=42
  src/tastelayer/{config.py, schema.py, canon.py, lexicon.py, data/ingest.py, tagging/{rubric.py, client.py, tag.py, consensus.py}, embed/{cooc.py, flavorgraph.py, probes.py, slerp.py}, model/{features.py, train.py, calibrate.py, evaluate.py, predict.py}, reco/{filters.py, score.py, update.py}, api.py}
  data/raw/, data/interim/, data/processed/, cache/, artifacts/, reports/, gold/
  Makefile targets: ingest, tag-menu, tag-corpus, consensus, embed, train, eval, serve, all

GLOBAL REQUIREMENTS: seeds fixed (SEED, numpy, torch, random); every LLM call cached on disk (diskcache keyed by sha256(model+prompt+schema_version)); JSONL logs with model, tokens, latency, cost estimate; pydantic validation for every record; no secrets in repo; `make all` reproducible; each step idempotent and resumable.

PLAN (spawn subagents for A, B, D in parallel; C is human):
A Data (Prompt 2 part 1): build data/processed/corpus.parquet (15-25k Indian+Italian dishes) and menu_dishes.json.
B Tagging (Prompt 2 part 2): Nebius multi-model tagging + Claude on the 50 gold dishes; consensus labels with std.
C Gold tool (Prompt 4): I label 50 dishes; served at localhost.
D Model (Prompt 3): mini-Epicure embeddings, baselines, TasteNet, calibration, eval table, predict API.
E Reco integration (Prompt 5) after D.

Start by: scaffolding the repo, writing schema.py (pydantic models for Dish, TasteVector, TagRecord, UserProfile), config.py, a Nebius smoke test that lists models (GET /v1/models) and prints which of these IDs exist: Qwen/Qwen3-235B-A22B-Instruct-2507, openai/gpt-oss-120b, deepseek-ai/DeepSeek-V4-Flash-0731, nvidia/Nemotron-3_5-Lightning, google/gemma-3-27b-it, Qwen/Qwen3-Embedding-8B. Then launch subagents. Report progress as a checklist every block. If a step exceeds its timebox by 50%, switch to the fallback noted in its prompt.
Prompt 2 — Corpus assembly + Nebius multi-model tagging
Context: read CONTRACT_CHANGES.md and src/tastelayer/schema.py (TasteLayer project; 8 taste dims 0-10 with anchors as in schema docstring). Build the distillation dataset.

PART 1 — CORPUS (src/tastelayer/data/ingest.py; timebox 50 min)
Download with Kaggle CLI into data/raw/ (skip any that fail, log it):
 - nehaprabhavalkar/indian-food-101 (indian_food.csv: name, ingredients, diet, flavor_profile, course, state, region; -1 = NaN)
 - kanishk307/6000-indian-food-recipes-dataset (or sooryaprakash12/cleaned-indian-recipes-dataset): TranslatedRecipeName, TranslatedIngredients, Cuisine, Course, Diet
 - shuyangli94/food-com-recipes-and-user-interactions: RAW_recipes.csv (name, tags, nutrition list [calories, total fat PDV, sugar PDV, sodium PDV, protein PDV, sat fat PDV, carbs PDV], ingredients, description), RAW_interactions.csv
 - ahmedshahriarsakib/uber-eats-usa-restaurants-menus: restaurants.csv (category), restaurant-menus.csv (restaurant_id, category, name, description, price)
 - What's Cooking train.json (competition 'whats-cooking'; requires accepting rules; else skip)
Filters:
 - Food.com: tags contain 'indian' or 'italian' (also 'pasta','pizza','curries').
 - Uber Eats: restaurants whose category contains 'indian' or 'italian'/'pizza'; keep menu items with description length >= 15 chars; drop drinks/sides like 'coke','water','napkins' via regex; cap 12k items, stratified by restaurant.
 - What's Cooking: cuisine in {indian, italian} (ingredients only; used for co-occurrence, not tagging).
Unified row: uid, source, cuisine, name, description, ingredients_raw[], ingredients_canonical[], extra (json: flavor_profile, nutrition, tags). Dedupe by normalized name+cuisine (rapidfuzz ratio>95). Also load data/raw/menus/*.json (our 300-450 menu dishes) into menu_dishes.json validated against Dish schema.
Canonicalisation (canon.py): lowercase, strip quantities/units/prep words (chopped, fresh, to taste...), singularize, map synonyms (curd->yoghurt, maida->all-purpose flour, dhania->coriander, jeera->cumin, haldi->turmeric, methi->fenugreek, kasuri methi->dried fenugreek leaves, parmigiano reggiano->parmesan) via a YAML dict; keep tokens with freq>=5. Save artifacts/ingredient_vocab.json with counts.
Target: 15-25k taggable dishes (menu-like > recipe). Print counts per source/cuisine.

PART 2 — TAGGING (tagging/*.py; timebox 90 min wall clock, run models concurrently)
Client: openai.AsyncOpenAI(base_url=NEBIUS_BASE_URL, api_key=NEBIUS_API_KEY). Separate Anthropic client for Claude (gold 50 + menu dishes only, tagged_by=claude).
Models (config list, skip unavailable): Qwen/Qwen3-235B-A22B-Instruct-2507 (primary), openai/gpt-oss-120b, deepseek-ai/DeepSeek-V4-Flash-0731; optional nvidia/Nemotron-3_5-Lightning.
Batching: 10 dishes per request. Output JSON: {"items":[{"uid":str,"taste":{8 ints 0-10},"evidence":{dim: short phrase},"confidence":0-1,"ingredients_inferred":[str]}]}.
Use response_format={"type":"json_schema","json_schema":{"name":"taste_batch","schema":<pydantic schema>,"strict":true}}; if the model errors on json_schema, fall back to {"type":"json_object"} and include the schema in the prompt. temperature=0.2, max_tokens 2500, seed=SEED where supported.
Validation: pydantic; uid set must equal input set; ints clamped 0-10; on failure retry up to 3x (tenacity, exponential backoff with jitter); after 3 failures split batch into halves; log failures to cache/failures.jsonl.
Rate limits: aiolimiter per model (start 50 RPM, 350k TPM); read x-ratelimit-remaining-requests/tokens headers via with_raw_response and adapt; on 429 honour Retry-After. Concurrency semaphore 16 per model.
Cache: diskcache keyed sha256(model|rubric_version|batch_uids|text); resumable.
Cost tracker: per-model token counts * prices from config ($/1M in/out: qwen3-235b 0.20/0.60; gpt-oss-120b 0.15/0.60; deepseek-v4-flash-0731 0.14/0.28; nemotron-3.5-lightning 0.06/0.24); hard budget env MAX_USD=15, abort when exceeded.

RUBRIC (tagging/rubric.py, RUBRIC_VERSION="v1"): system prompt =
"You are a professional chef and sensory panel lead. Rate how the DISH AS SERVED tastes to a typical UK diner on 8 independent 0-10 scales. Use the anchors exactly: 0=absent, 2-3=low anchor, 5=mid anchor, 8-9=high anchor, 10=extreme. [list all 8 dims with low/mid/high anchors]. Rules: rate each dim independently; spice = chilli heat only (not aromatic spices); sour includes tomato acidity, tamarind, lime, yoghurt tang; richness = fat/heaviness; crunch = textural crispness of main components; creaminess = smooth/creamy mouthfeel; use the full scale, avoid defaulting to 5; restaurant versions of Indian curries are usually richer than home recipes. Return only JSON."
Few-shot (6, in the user message before the batch): Margherita pizza {spice0,sweet2,sour4,bitter1,umami6,richness5,crunch4,creaminess4}; Spaghetti carbonara {0,1,1,1,7,8,1,7}; Rocket & parmesan salad {1,1,3,5,6,3,3,1}; Chicken tikka masala {4,3,3,1,5,7,0,7}; Lamb vindaloo {9,2,5,1,6,6,0,3}; Dal makhani {2,1,1,1,4,9,0,8}; (also include Papadum with mango chutney {1,6,2,0,1,3,9,0} if budget allows).
User message per batch: list of {uid, cuisine, name, description, ingredients} (truncate description 400 chars, ingredients 25 items).

Also ask the same models, in a separate pass, to tag each canonical ingredient (top ~1,500 by frequency) on the same 8 dims (20 ingredients per call) -> data/processed/ingredient_taste_llm.parquet (used for probe directions).

CONSENSUS (consensus.py): per dish per dim: median across models -> label; std -> taste_std; confidence = 1 - min(std/3,1). tagged_by = "consensus:qwen3-235b+gpt-oss-120b+deepseek-v4-flash". Flag dishes with any dim std>2 into reports/high_disagreement.csv (candidates for gold labelling = active learning). Save data/processed/labels.parquet (uid, model, 8 dims) and labels_consensus.parquet.
Per-model outputs on menu dishes are also written to the Dish schema with tagged_by "nebula:<model_id>" (alias of nebius) and verified=false.
Report: reports/tagging_summary.md with counts, failure rate, cost per model, inter-model Spearman per dim.
Prompt 3 — Train TasteNet + mini-Epicure, evaluate, export
Context: TasteLayer; inputs data/processed/corpus.parquet, labels_consensus.parquet, ingredient_taste_llm.parquet, gold/gold.json (50 human-labelled dishes; may arrive late — build everything first, eval when present). Fixed SEED. Timebox 120 min.

1) LEXICON BASELINE (lexicon.py) — "ingredient lookup": hand-written YAML of ~150 ingredients/keywords -> partial 8-dim scores (e.g. habanero spice 10, ghee richness 8, papadum crunch 9, mascarpone creaminess 9, tamarind sour 9, radicchio bitter 8, anchovy umami 9, sugar/jaggery sweet 8) + keyword rules on description (fried/crispy -> crunch+4, cream/makhani -> creaminess+3 richness+3, "hot"/"vindaloo"/"phaal" -> spice). Dish score = max-pool for spice/sour/bitter/crunch, weighted mean for others, default 2. Add Food.com nutrition features when present (fat PDV -> richness, sugar PDV -> sweet). Output predict_lookup(dish)->8 dims.

2) MINI-EPICURE (embed/cooc.py): from all corpus ingredient lists (+ What's Cooking indian/italian + optional RecipeNLG filtered by title keywords if downloaded), keep canonical ingredients with >=20 occurrences (Epicure threshold), compute PPMI/NPMI; train gensim Word2Vec skip-gram (sg=1, vector_size=128 (and 300 if time), window=10 over shuffled ingredient sets, 5 permutations per recipe, negative=5, epochs=20, seed=SEED, workers=8). Optional "Core" variant: add random walks over positive-NPMI graph (walk_length=50, 20 walks/node) to the training sentences. Save artifacts/ingredient_emb.npy + ingredient_vocab.json. Also load FlavorGraph 300-D pickle (dict node_id->vector; join input/nodes_191120.csv name/node_type, keep node_type=='ingredient'; map names with rapidfuzz to our vocab) -> artifacts/flavorgraph_emb.npy (embed/flavorgraph.py).
   PROBES (embed/probes.py): for each dim, ridge regression from ingredient embedding -> LLM ingredient taste score (ingredient_taste_llm.parquet), plus anchor constraint: report the projection of the 3 benchmark anchors (low<mid<high must hold; print a check table). Direction = normalized ridge coef. Report 5-fold Spearman per dim per embedding (ours vs FlavorGraph). Save artifacts/taste_directions.json.
   Pairings: nearest neighbours by cosine; sanity print for paneer, basil, tamarind, guanciale.

3) FEATURES (model/features.py): text = f"{cuisine} | {name} | {description} | ingredients: {', '.join(ingredients_canonical)}". 
   E_text: local sentence-transformers "BAAI/bge-base-en-v1.5" (or "intfloat/e5-base-v2"), normalized; optionally Qwen/Qwen3-Embedding-8B via Nebius /v1/embeddings (cached; ~$0.01/1M tokens) — pick whichever scores better on CV.
   E_ing: IDF-weighted mean of ingredient embeddings (ours + FlavorGraph) and projections onto the 8 taste directions (8 features each).
   L: lexicon baseline 8 scores + keyword flags. Concatenate; standardize.

4) TASTENET (model/train.py): targets = consensus labels (sample_weight = confidence). Split: hold out 10% corpus by source-stratified group; never train on gold dishes. Models: (a) RidgeCV multi-output (alphas logspace), (b) sklearn MLPRegressor(512,256, early_stopping) or a 2-layer torch MLP, (c) STRETCH if a GPU is available (Nebius AI Cloud or local): fine-tune bge-base / microsoft/deberta-v3-base with an 8-output regression head, MSE, lr 3e-5, 3 epochs, bs 32, max_len 128 — ~20-40 min. Pick best by held-out mean Spearman. Save artifacts/tastenet.joblib (+ encoder dir if c).

5) CALIBRATION (model/calibrate.py): per dim IsotonicRegression (out_of_bounds='clip') fit on gold with 5-fold cross-fitting to produce unbiased gold metrics; final calibrators fit on all 50 -> artifacts/calibrators.joblib. Apply to Claude/Qwen too for a fair "calibrated LLM" column.

6) EVALUATION (model/evaluate.py) -> reports/eval.md + reports/eval.csv:
   Rows: 8 dims + mean. Column groups: Human-human (overlap subset), Claude, each Nebius model, 3-model vote, Lookup baseline, TasteNet raw, TasteNet calibrated (cross-fitted). Metrics: Spearman rho, MAE, % within ±1, plus 1000x bootstrap 95% CI for Spearman. Extra rows: cost per 1k dishes (from token logs), runtime LLM calls (TasteNet = 0), latency per dish. Also Indian Food 101 check: AUROC of predicted spice for flavor_profile=='spicy' and sweet for 'sweet'. Plot per-dim bar chart reports/eval.png. Write a 5-line interpretation.
   Active learning output: rank remaining menu dishes by (ensemble std + TasteNet-vs-vote disagreement) -> reports/next_to_label.csv.

7) EXPORT (model/predict.py + api.py): 
   predict_taste(dish: Dish) -> {taste: {8 floats 0-10 rounded 1dp}, taste_confidence: {8}, taste_model_version: "tastenet-v1"}; batch version; FastAPI POST /taste. Write predictions for all menu dishes into data/processed/menu_dishes_tagged.json with tagged_by="model:tastenet-v1", verified=false (taste only — never touch allergens/dietary). Unit tests: schema roundtrip, range 0-10, determinism (same input -> same output).
FALLBACKS: if no gold yet, report vs held-out consensus; if mini-Epicure probes fail anchor ordering, use FlavorGraph; if MLP < ridge, ship ridge.
Prompt 4 — Gold-set labelling tool + report
Build gold/app.py: a single-file FastAPI + vanilla HTML tool (no build step) at localhost:8001 for labelling 50 dishes from gold/gold_candidates.json (select: 25 Indian/25 Italian, stratified by dish_family, plus up to 10 from reports/high_disagreement.csv). For each dish show name, description, ingredients, restaurant; 8 sliders 0-10 (integer) with the low/mid/high anchor names printed under each slider (spice: black pepper/jalapeno/habanero; sweet: roasted red pepper/caramelised onion/maple syrup; sour: natural yoghurt/lemon juice/tamarind; bitter: rocket/70% dark chocolate/radicchio; umami: tomato/parmesan/anchovy or miso; richness: olive-oil dressing/cream sauce/ghee-heavy dal makhani; crunch: soft bread/toasted nuts/fried papadum; creaminess: clear broth/hummus/mascarpone). Hide all model predictions (blind). Annotator name field; keyboard shortcuts (1-0 set value, Tab next slider, Enter save+next); autosave each label to gold/labels_<annotator>.jsonl; progress bar; "skip/unsure" flag. Mark 20 dishes as overlap (both annotators label them). Script gold/build_gold.py merges: final label = mean of annotators (rounded), writes gold/gold.json with tagged_by="human", verified=true for taste only, and reports/human_agreement.md (per-dim Spearman, MAE, within±1 on overlap = ceiling row for eval). Target labelling speed: ~45 s/dish.
Prompt 5 — Recommender integration (deterministic, filter-first)
Implement src/tastelayer/reco/ with ONE deterministic recommender. Order is mandatory:
1) filters.py (pure code, unit-tested): exclude dish if any user allergen ∈ dish.allergens; if dish not verified or the relevant dietary flag == "unknown" -> exclude for strict users, else include with warning flag; exclude if any disliked ingredient ∈ ingredients ∪ ingredients_canonical. Never read taste, allergens_suggested, or any model output here. 100% branch coverage tests including unknown-flag cases.
2) score.py: for remaining dishes: taste_fit = 1 - sqrt(sum_d w_d*(u_d - x_d)^2)/sqrt(sum_d w_d*100) with w_d = 0.5 + user.confidence_d (weights optionally from cached occasion->weights LLM mapping); price_fit from price_pref; novelty = cosine distance between dish ingredient centroid (artifacts/ingredient_emb.npy) and user ingredient_centroid; score = 0.65*taste_fit + 0.15*price_fit + 0.10*vibe_fit + 0.10*(adventurousness/10)*novelty - 0.1*(1-dish.taste_confidence_mean). Deterministic tie-break by dish id. Return top-k with per-term breakdown for explanation text (explanations generated from the breakdown by template; LLM rewording optional and cached).
3) update.py: star rating r on dish x: alpha = 0.3/(1+n_ratings*0.1); u <- clip(u + alpha*((r-3)/2)*(x - u), 0, 10) for r!=3; confidence_d += 0.05*(1 - |r-3|==0) capped 1; venue choice = implicit r=4 on the ordered dish; ingredient_centroid <- EMA(0.8) toward liked dishes' centroids. Cold start: survey -> initial u; confidence 0.3.
4) slerp.py "take me further" slider: given unit vectors a (user ingredient centroid) and b (target = normalize(a + taste_directions[dim]) or a cuisine/region centroid), theta in [0, 90°]: slerp(a,b,t)=sin((1-t)Ω)/sinΩ·a + sin(tΩ)/sinΩ·b with t=theta/Ω (clip); retrieve nearest dishes (post-filter!) by cosine to the rotated vector; also shift taste target u'_dim = u_dim + theta/90*3. Expose GET /recommend?user_id&k&theta&direction.
5) Tests: determinism (same inputs -> same ranking), filters run before scoring, unknown flags exclude/warn, theta=0 equals base ranking.
