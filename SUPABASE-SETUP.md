# Palatable cloud saving

The app works with browser-local storage without Supabase. No Supabase project has been connected or migrated automatically.

1. Apply `supabase/migrations/202610240001_palatable.sql` in your project's SQL editor.
2. Create `frontend/.env.local` (ignored) with `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` (the public anon/publishable key). Never use a service-role key in the frontend.
3. Enable email/password authentication in Supabase and configure your application's site URL and confirmation redirect URL.
4. Restart Vite, open My taste profile, and sign up. Confirm the email if required, sign in, and choose Save to cloud. Signing in restores previously saved data; saving is explicit. Only derived profiles and reviewed menu items are saved, not photos or delivery ZIPs.

The included row-level policies scope every operation to the authenticated account. Verify isolation using two accounts before production deployment. Schema execution and live authentication have not been tested without a connected project.

# Development

From repository root run `.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000`. In `frontend` run `npm ci --cache /workspace/.npm-cache`, then `npm run dev -- --host 127.0.0.1 --port 5173`. Vite proxies `/api` to the backend. Production hosting must proxy `/api` too, or build with `VITE_API_URL` pointing to your deployed API.

Menu OCR uses Tesseract.js in the browser. The install/build scripts copy its worker, WASM and English model from integrity-checked npm dependencies into generated `frontend/public/ocr/` assets, so no external OCR CDN is needed at runtime. Camera access requires a supported mobile browser and a secure HTTPS origin. Manual text entry is available if OCR assets cannot load. Extraction is a reviewable heuristic, not a verified ingredient or allergen catalogue. Menu ranking uses exact ingredient overlap with the derived Uber history; items without overlap retain their input order. Taste sliders describe the profile but are not used to infer unknown menu taste values. Reviewed menus remain private browser datasets until explicitly saved to a signed-in cloud account.
