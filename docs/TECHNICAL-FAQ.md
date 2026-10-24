# Palatable technical Q&A

**Q: What does Palatable do?**  
A: It learns a food taste profile, recommends restaurants and dishes for an occasion, imports Uber Eats order history, and helps a user choose from a photographed menu.

**Q: What is the frontend stack?**  
A: React 18, JavaScript, CSS and Vite 5. The responsive UI uses reusable React components, stateful selections and browser storage. It is a web app, not a native iOS app.

**Q: What powers the backend?**  
A: Python, FastAPI, Pydantic validation and Uvicorn. NumPy supports taste-vector and embedding calculations. Vite proxies `/api` to the backend during development; production needs an equivalent proxy or a configured API URL.

**Q: What makes the UI interactive?**  
A: A five-stage guided journey, progress indicators, selectable diet/allergy chips, benchmark buttons, live sliders, star ratings, venue selection cards, animated feedback, an eight-axis radar chart, occasion cards, budget controls and expandable dish recommendations. Journey state survives navigation within the current page session.

**Q: Which profile details can users provide?**  
A: First name, optional surname, vegan/vegetarian/halal/kosher preferences, UK FSA allergens, disliked ingredients and a strict verification preference.

**Q: Which taste dimensions are modelled?**  
A: Spice, sweet, sour, bitter, umami, richness, crunch and creaminess. Each is scored from 0 to 10.

**Q: How do the slider benchmarks work?**  
A: Each dimension has low/mid/high ingredient anchors at 2.5, 5 and 8.5, read from the shared `data/taste_dimensions.json` used by the tagging rubric. Selecting an anchor sets that slider to its documented score.

**Q: How does the dish-rating survey work?**  
A: The backend selects up to 12 eligible dishes with diverse taste/flavour vectors. Users can rate known dishes from 1 to 5 stars or skip unfamiliar ones. Ratings update the taste profile; low-rated dishes are excluded from normal recommendations.

**Q: What do the venue-choice rounds learn?**  
A: Users choose among three restaurants serving the same dish family at different price tiers. The backend updates price and atmosphere preferences from those choices. The number of rounds depends on eligible catalogue coverage; restrictions can leave no rounds.

**Q: What is shown on the taste map?**  
A: An SVG radar of all eight taste dimensions, strongest dimensions, adventurousness score, price preference and learned atmosphere preferences.

**Q: Can the taste profile be shared?**  
A: The radar is rendered into a PNG using SVG and Canvas. The app uses the native Web Share API when file sharing is supported, otherwise it downloads the image. Native sharing depends on the browser/device.

**Q: Which occasions are supported?**  
A: Comfort food, date night, quick bite, family time and celebration. Users can also set a maximum restaurant price tier from 1 to 4.

**Q: How are restaurant recommendations scored?**  
A: A deterministic Python recommender combines taste similarity, Epicure flavour similarity, price preference, atmosphere fit and novelty. The configured weights are 0.5 taste, 0.15 Epicure, 0.15 price, 0.1 atmosphere and 0.1 novelty. Occasion weights affect atmosphere and price fit.

**Q: Does it recommend specific dishes?**  
A: Yes. Results include restaurant details, selected dishes, prices, explanation text and warnings. Expandable cards show ingredients and listed allergens. The backend can include one stretch pick from a less familiar dish family.

**Q: Does the app generate match percentages?**  
A: No. Backend scores are internal ranking values, not calibrated probabilities. The UI does not present invented percentage matches.

**Q: What is Epicure used for?**  
A: A committed slim Epicure-Chem ingredient embedding model supports flavour vectors and similarities without a network download or API key at runtime. Dish vectors are constructed from the active catalogue. Without the assets, recommendations fall back to taste scores.

**Q: How are allergies and dietary rules handled?**  
A: Deterministic code filters listed and text-implied allergens and incompatible dietary flags. Text rules can only exclude, not establish safety. Unknown or unverified information produces warnings or exclusion in strict mode. Restaurant confirmation is still needed.

**Q: Are disliked ingredients treated like allergies?**  
A: No. In the existing recommender they produce warnings and a ranking penalty, and affect flavour preference. They are not equivalent to an allergy exclusion. Stretch picks avoid disliked ingredients.

**Q: How does Uber Eats import work?**  
A: Users upload a delivery-data ZIP or Eats order CSV. The backend opens only supported food-order records, skips cancelled orders, and derives taste, cuisine, ingredient and price signals. The UI displays an import confirmation and summary. This is file import, not a live Uber account connection.

**Q: Is there an upload-size limit?**  
A: Menu images are limited to 20 MB by the UI. The backend also applies a 20 MB delivery-upload limit and checks the size of selected ZIP entries.

**Q: Does the app retain raw delivery exports?**  
A: The running import endpoint processes the upload in memory and retains derived signals rather than writing the raw upload. Separately, the supplied private Uber ZIP was explicitly committed to the repository at the owner's request; it remains in Git history and is accessible according to repository permissions.

**Q: How does menu-photo recognition work?**  
A: Tesseract.js performs English OCR in a browser worker. Worker code, WebAssembly and the English language model are copied from npm dependencies into locally served assets during installation/build. Menu images are not uploaded to an external OCR service.

**Q: Can users take a photo directly?**  
A: The file input requests the device's rear camera where supported. Uploading existing photos and manual text entry are also available. Actual camera capture still needs testing on a supported phone and secure origin.

**Q: What is extracted from menu text?**  
A: A reviewable parser separates dish names, explicitly listed descriptions/ingredients and prices. English OCR and line-format heuristics can make mistakes, so users can edit names, ingredients and prices, add items and remove incorrect rows before ranking or saving.

**Q: Does OCR infer missing ingredients or allergens?**  
A: No. Missing ingredients remain unknown. Scanned-menu results are preference matches and do not certify dietary or allergy safety.

**Q: How are photographed-menu dishes ranked?**  
A: The frontend scores ingredient-word overlap with the profile's derived Uber history. It ranks only the reviewed items in that menu. Without overlap, items keep input order. This path does not yet use the full taste-vector/Epicure restaurant recommender or infer taste vectors for scanned dishes.

**Q: Can menus be saved and reopened?**  
A: Reviewed menu datasets can be saved, reopened for ranking and deleted in the UI. They persist in localStorage. Signed-in users can explicitly save them with their profile to Supabase after connection.

**Q: What is stored locally?**  
A: The derived profile and reviewed menus persist in browser localStorage. Current survey progress is held in React state and does not survive a full reload. Backend profiles are in memory and reset on server restart.

**Q: Is Supabase connected?**  
A: Not yet. The code supports email/password sign-up/sign-in, session handling, restoring saved profiles and explicit cloud saves. A migration and setup guide are included, but project settings, schema application and live authentication validation are outstanding.

**Q: How is cloud data isolated between accounts?**  
A: The supplied Supabase migration enables row-level security on `palatable_profiles`, with select/insert/update/delete policies requiring `auth.uid()` to match `user_id`. Isolation must still be checked against a real project with separate accounts.

**Q: Does the app require an LLM for recommendations?**  
A: No. Optional Nebius Token Factory calls can rewrite explanation text, with caching and a template fallback. LLMs do not decide allergen eligibility. No working Nebius credential is configured in this environment.

**Q: What restaurant data is available?**  
A: The default committed demo catalogue has 6 restaurants and 36 dishes. The separate TasteLayer catalogue has 20 restaurants and 296 stand-in dishes. `DATA_DIR` selects the backend catalogue. Stand-in menus and unverified items must not be described as production-verified restaurant data.

**Q: What is the separate TasteLayer package?**  
A: The repository contains optional data ingestion, Nebius tagging, consensus labels, ingredient embeddings, a TasteNet regressor, calibration/evaluation tools and its own API. This offline training workflow was not installed or validated as part of the running app setup.

**Q: Which accessibility and feedback features are present?**  
A: Keyboard-operable controls, labelled inputs, pressed/expanded/current states, visible focus indicators, live status/error messages, loading/disabled states, focus management between journey stages and reduced-motion CSS.

**Q: What has been tested?**  
A: The existing 197 backend tests and 4 menu-parser/ranking tests pass, as does the Vite production build. Browser checks have covered desktop/mobile rendering, delivery ZIP upload, generated-image OCR, menu review and save/reload. The new guided flow is separately checked through live APIs. Phone camera capture and live Supabase operation remain unverified.

**Q: Is this deployed or a production system?**  
A: It is a development/demo application. A production deployment, persistent backend storage, real-project authentication validation, production catalogue verification and operational monitoring are separate work.
