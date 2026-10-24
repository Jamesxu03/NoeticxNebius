# PROJECT BRIEF: shared context for all three team AI assistants

You are helping one of 3 teammates build a hackathon project. Each teammate has their own AI chat, so this brief is the single source of truth. Stay inside the data contracts in section 4. If you think a contract should change, tell your human so they can sync with the team. Do not change it silently.

## 1. What we're building
**Hackathon:** Noetic x Nebius Hackathon. [TBD: theme, judging criteria, submission deadline, demo format]
**Build window:** about 4 hours. Scope everything for that. Cut features before cutting polish on the core flow.
**Team:** 3 people, each with their own AI chat. [TBD: names and strengths]

A mobile-style web app, shown inside an iPhone-shaped frame, that:
1. learns a person's food taste through a short survey,
2. shows them a shareable taste profile,
3. recommends restaurants and specific dishes for their occasion, from a curated set of 20 restaurants (10 Italian, 10 Indian) whose menus we have logged.

A real Deliveroo/Uber Eats integration isn't realistic in the time we have. The pitch is that we are the **portable taste-profile and recommendation layer** a delivery platform could plug in later. [TBD: connection to the "Epicure" food foundation model / DE3 Innovation & Entrepreneurship brief, if any.]

## 2. User journey (screens in order)
1. **Login.** Simple (fake or magic-link; don't spend time here).
2. **Profile page.** Name, surname, dietary restrictions (vegan, vegetarian, halal, kosher, allergies), and ingredients the user dislikes (used later as "flag" warnings).
3. **Survey, step 1: tolerances.** Slider bars for each taste dimension (spice, sweet, etc.). Each slider has **3 benchmark ingredients** at low/mid/high so the scale is meaningful.
4. **Survey, step 2: dish ratings.** The algorithm picks ~12-15 pre-identified dishes. The user rates each with 1-5 stars.
5. **Survey, step 3: venue picks.** 3-4 rounds. Each shows the same dish from 3 different venues (different price/vibe tiers) and the user picks one. This reveals vibe and price preference.
6. A **progress header** on every survey screen shows "what's next" in the journey.
7. **Taste profile screen.** Taste map/radar graph, adventurousness score, vibe/price preference, download/share image.
8. **Main app.** "What occasion are you eating out for?" Then restaurant + dish recommendations based on the full profile, each with a short "why we picked this" explanation.

## 3. Architecture principles
- **One recommender pipeline, with LLMs only where they add value.** We are NOT building two full competing recommenders.
- **We train our own model.** [PROPOSED, confirm with team] Claude tags the full dish set (~250-350 dishes). We then train a small taste-tagging model (text embedding of dish name + description + ingredients -> 8-dim taste vector regressor) on those labels. Validate it on the 50-dish hand-labelled gold set. This is the "own model" we can show, and it is cheap to run at inference time. The user profile update (ratings nudging the taste vector) is a second small learned/online component.
- **Hard filters are pure code and never LLM-decided or model-decided:** dietary restrictions, allergies, disliked ingredients. If a dish's flag is `"unknown"` for a restriction the user has, exclude it or show a clear warning. Never assume safe.
- **Taste matching:** every dish has a taste vector in the same space as the user's profile vector. Score by similarity, then adjust by rating history, price/vibe fit, occasion weights and an adventurousness/novelty term.
- **LLM uses:** (a) offline dish tagging (labels for training and the gold-set comparison), (b) generating the explanation text, (c) optionally mapping occasion to weights. Runtime LLM calls should be cached or minimal because credits are limited.
- **Model comparison (one focused experiment):** on the 50-dish gold set (hand-labelled by us), report per-dimension agreement for: Claude, a Nebius-hosted open-source model, and our trained regressor. Use the best option for final tagging, and use a cheaper open-source model at runtime for explanation text.
- **Credits:** teammates have free Nebius credits for open-source models and Claude Code access. Use Claude for hard, high-accuracy jobs (tagging, schema extraction) and open-source for bulk and runtime jobs.

## 4. Data contracts (source of truth)

### 4.1 Taste dimensions (`taste_dimensions.json`), all scored 0-10
| key | meaning | proposed anchors (low / mid / high) |
|---|---|---|
| spice | capsaicin heat | black pepper / jalapeño / habanero |
| sweet | sweetness | roasted red pepper / caramelised onion / maple syrup |
| sour | acidity/tang | natural yoghurt / lemon juice / tamarind |
| bitter | bitterness | rocket / 70% dark chocolate / radicchio |
| umami | savouriness | tomato / parmesan / anchovy or miso |
| richness | fat/heaviness | olive-oil dressing / cream sauce / ghee-heavy dal makhani |
| crunch | crispy/crunchy texture | soft bread / toasted nuts / fried papadum |
| creaminess | smooth/creamy texture | clear broth / hummus / mascarpone |

**The slider anchors and the dish-tagging rubric MUST use the same scale.** The data person owns this file and the frontend reads it.

### 4.2 Dish
```json
{
  "id": "d_0001",
  "restaurant_id": "r_01",
  "name": "Butter Chicken",
  "description": "string from menu",
  "cuisine": "indian | italian",
  "category": "starter | main | side | dessert | drink",
  "dish_family": "curry | pizza | pasta | biryani | ...",
  "price_gbp": 14.5,
  "ingredients": ["chicken", "tomato", "butter", "cream"],
  "allergens": ["milk", "nuts"],
  "dietary": { "vegan": "yes|no|unknown", "vegetarian": "yes|no|unknown", "halal": "yes|no|unknown", "kosher": "yes|no|unknown" },
  "taste": { "spice": 4, "sweet": 3, "sour": 2, "bitter": 0, "umami": 7, "richness": 9, "crunch": 1, "creaminess": 8 },
  "serving_temp": "hot | warm | cold",
  "adventurousness": 3,
  "tagged_by": "claude | nebius:<model> | trained-regressor | human",
  "verified": false
}
```

### 4.3 Restaurant
```json
{
  "id": "r_01",
  "name": "string",
  "cuisine": "indian | italian",
  "price_tier": 1,
  "vibe_tags": ["casual", "date-night", "trendy", "traditional", "quick", "family"],
  "area": "string",
  "lat": 0.0, "lng": 0.0,
  "halal_status": "certified | some | none | unknown",
  "menu_source_url": "string",
  "dish_ids": []
}
```
`price_tier` is 1-4 (1 = cheap/casual, 4 = premium).

### 4.4 UserProfile
```json
{
  "id": "u_001",
  "first_name": "", "last_name": "",
  "dietary": ["vegetarian", "halal"],
  "allergies": ["peanuts"],
  "disliked_ingredients": ["coriander"],
  "taste_vector": { "spice": 5, "sweet": 4, "...": 0 },
  "taste_confidence": { "spice": 0.4, "...": 0 },
  "dish_ratings": [ { "dish_id": "d_0001", "stars": 4 } ],
  "venue_choices": [ { "round_id": "v1", "options": ["r_01","r_02","r_03"], "chosen": "r_02" } ],
  "price_pref": 2.3,
  "vibe_pref": { "casual": 0.7, "date-night": 0.2 },
  "adventurousness": 6.1,
  "created_at": "ISO8601"
}
```

### 4.5 API (Recommender/Backend owns; Frontend codes against mocks first)
- `POST /profile` create from profile page
- `PUT /profile/:id/tolerances` set slider values
- `GET /survey/dishes?profile_id=` dishes to rate
- `POST /profile/:id/ratings`
- `GET /survey/venue-rounds?profile_id=`
- `POST /profile/:id/venue-choice`
- `GET /profile/:id/taste-map` returns the chart data, adventurousness, price/vibe summary
- `POST /recommend` body `{ profile_id, occasion, party_size?, budget? }` returns ranked `[{restaurant, dishes[], score, explanation, warnings[]}]`

**Important constraint on venue rounds:** the "same dish at 3 venues" rounds only work if 3 of our 20 restaurants serve the same `dish_family` at different price/vibe tiers (e.g. margherita pizza at a chain, a trendy independent and a premium restaurant; butter chicken or biryani likewise). Restaurant selection must guarantee this for at least 3 dish families (4 is better).

## 5. Scale (sized for a ~4 hour build)
- 20 restaurants (10 Italian, 10 Indian), spread across price tiers 1-4. Fallback if behind: ship 12 restaurants (6+6) that still satisfy the venue-round constraint, then add more.
- ~12-20 dishes per restaurant, so roughly 250-350 dishes. Quality of tags matters more than count.
- Survey: 8 sliders, ~12-15 rated dishes, 3-4 venue rounds
- 8-10 synthetic personas for testing (e.g. vegan spice-lover, halal adventurous eater, nut-allergy cautious eater)
- Gold set: 50 hand-labelled dishes for the model comparison

## 6. Data sources (for the data owner)
- Menus: pick the 20 restaurants, pull menus from their sites or PDFs, structure with Claude Code into the Dish schema. Check ToS for any scraping and don't present scraped delivery-platform data as a production pipeline.
- Venue metadata: Google Places API (price level, location) or manual.
- Grounding/validation: Kaggle "Indian Food 101" (flavour, diet, ingredients), RecipeNLG / Food.com recipes (typical ingredient lists), USDA FoodData Central, optionally FlavorDB/FlavorGraph.
- LLM enrichment with strict JSON output, then human verification of allergens and halal/kosher flags (menus rarely state these).

## 7. Roles (one per teammate)
**Role A: Frontend.** All screens in section 2 inside an iPhone-style frame, mobile-first. Slider component reads anchors from `taste_dimensions.json`. Taste map (radar or 2D map), share/download image, recommendation cards with warnings. Code against mock JSON until the API is live.

**Role B: Data pipeline and model training.** Own `taste_dimensions.json`, `restaurants.json`, `dishes.json`. Select the 20 restaurants (satisfying the venue-round constraint), extract menus, run LLM tagging with a strict rubric tied to section 4.1 anchors, verify allergens/dietary flags, build the 50-dish gold set, train the small taste-tagging regressor, and run the Claude vs Nebius open-source vs trained-model comparison with a per-dimension agreement table. Ship the first 30-50 dishes early so the recommender can test.

**Role C: Recommender and backend.** Profile update logic (sliders to initial vector; star ratings nudge the vector toward/away from rated dishes; venue picks update price/vibe preference; adventurousness score), hard filters, scoring, occasion weighting, explanation prompts, API endpoints in section 4.5, persona test harness, deployment. Start with mock dishes until Role B delivers.

[TBD: who owns the pitch/demo script and business framing. Suggestion: whoever has spare capacity, with pre-loaded demo users so the live demo doesn't depend on slow API calls.]

## 7b. Suggested 4-hour timeline
- 0:00-0:20 Agree contracts, create repo, everyone starts on mocks.
- 0:20-1:00 Parallel build. Data ships the first 50 tagged dishes by ~0:50.
- 1:00-2:30 Core build continues. Frontend wires to the API as endpoints go live. Data finishes tagging and starts model training.
- 2:30-3:20 Integration, persona tests, model comparison table.
- 3:20-3:50 Polish, pre-loaded demo user, rehearse the demo.
- 3:50-4:00 Feature freeze and submit.

## 8. Rules for you (the AI assistant)
1. Read this brief fully before doing anything, then confirm your role with your human.
2. Don't change contracts in section 4 without flagging it to your human.
3. Build against mocks first so no one is blocked waiting on teammates.
4. Never let an LLM or a trained model decide an allergen or dietary-restriction match at runtime. Those are code rules over verified data.
5. Be credit-conscious: prefer cached or precomputed outputs, and small models for bulk work.
6. Keep code simple and demo-ready. Judges see a polished 3-minute flow, not infrastructure.
7. When something is ambiguous, ask your human a short question rather than guessing.
8. Never commit secrets. API keys live in `.env`, which is gitignored.

## 9. Open decisions
[TBD: stack, repo/branching convention, deployment target, occasion list and weights, final restaurant list, judging criteria, who is on which role.]
