// Every call to the food API goes through this file.
// Dev: Vite proxies /api to the backend on :8000 (vite.config.js). Production: set VITE_API_URL.
const API = import.meta.env.VITE_API_URL || '/api'

export async function request(path, body, method = 'POST') {
  const form = body instanceof FormData
  const r = await fetch(API + path, {method, headers: form ? undefined : {'Content-Type': 'application/json'}, body: body ? form ? body : JSON.stringify(body) : undefined})
  if (!r.ok) { const err = await r.json().catch(() => ({})); const e = new Error(typeof err.detail === 'string' ? err.detail : 'Could not reach the food API. Please try again.'); e.status = r.status; throw e }
  return r.json()
}

export const api = {
  tasteDimensions: () => request('/taste-dimensions', null, 'GET'),
  occasions: () => request('/occasions', null, 'GET'), // [{id, label, blurb, image}]
  createProfile: (body) => request('/profile', body),
  getProfile: (id) => request(`/profile/${id}`, null, 'GET'),
  restoreProfile: (profile) => request(`/profile/${profile.id}`, profile, 'PUT'), // browser copy -> backend after a restart
  setTolerances: (id, taste) => request(`/profile/${id}/tolerances`, taste, 'PUT'),
  surveyDishes: (id) => request(`/survey/dishes?profile_id=${id}`, null, 'GET'),
  postRatings: (id, ratings) => request(`/profile/${id}/ratings`, {ratings}),
  venueRounds: (id) => request(`/survey/venue-rounds?profile_id=${id}`, null, 'GET'),
  postVenueChoice: (id, choice) => request(`/profile/${id}/venue-choice`, choice),
  tasteMap: (id) => request(`/profile/${id}/taste-map`, null, 'GET'),
  recommend: (body) => request('/recommend', body), // {profile_id, occasion, party_size?, budget?}
  rankMenu: (profileId, items) => request('/menu/rank', {profile_id: profileId, items: items.map(({id, name, ingredients, price}) => ({id, name, ingredients: ingredients || '', price: String(price || '')}))}),
  // file: the Uber "Download your data" zip, or its Eats order CSV. Returns {profile, summary}.
  importUber: (id, file) => { const form = new FormData(); form.append('file', file); return request(`/profile/${id}/import/uber`, form) },
}

export const OCCASIONS = [['date-night', 'Date night'], ['quick-bite', 'Quick bite'], ['family', 'Family meal'], ['celebration', 'Celebration'], ['comfort', 'Comfort food']]
export const DIETARY = ['vegetarian', 'vegan', 'halal', 'kosher']
export const ALLERGENS = ['celery', 'gluten', 'crustaceans', 'egg', 'fish', 'lupin', 'milk', 'molluscs', 'mustard', 'nuts', 'peanuts', 'sesame', 'soya', 'sulphites']

// Make sure the backend knows this browser's profile and return the backend copy.
// Profiles live in backend memory, so after a restart the saved copy is restored instead of recreated.
export async function syncProfile(local, name) {
  const base = {first_name: name || local?.first_name || 'Food lover', dietary: local?.dietary || [], allergies: local?.allergies || [], disliked_ingredients: (local?.disliked_ingredients || []).map(s => s.trim()).filter(Boolean), strict: !!local?.strict}
  if (local?.id) {
    try { return await api.restoreProfile({...local, ...base}) } catch (e) { if (e.status !== 422) throw e } // 422: old local shape, start fresh
  }
  const p = await api.createProfile(base)
  return local?.taste_vector ? api.setTolerances(p.id, local.taste_vector) : p
}
