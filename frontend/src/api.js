// Every screen talks to the backend through this file only.
// Without VITE_API_URL it serves the JSON in backend/mocks, so the UI never waits on the API.
import tasteDimensions from '../../backend/mocks/taste_dimensions.json'
import profile from '../../backend/mocks/profile.json'
import surveyDishes from '../../backend/mocks/survey_dishes.json'
import venueRounds from '../../backend/mocks/venue_rounds.json'
import tasteMap from '../../backend/mocks/taste_map.json'
import recommendations from '../../backend/mocks/recommend.json'

const API = import.meta.env.VITE_API_URL

async function call(method, path, body, mock) {
  if (!API) return structuredClone(mock)
  const res = await fetch(API + path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`${method} ${path} -> ${res.status}`)
  return res.json()
}

export const api = {
  tasteDimensions: () => call('GET', '/taste-dimensions', null, tasteDimensions),
  createProfile: (body) => call('POST', '/profile', body, profile),
  setTolerances: (id, taste) => call('PUT', `/profile/${id}/tolerances`, taste, profile),
  surveyDishes: (id) => call('GET', `/survey/dishes?profile_id=${id}`, null, surveyDishes),
  postRatings: (id, ratings) => call('POST', `/profile/${id}/ratings`, { ratings }, profile),
  venueRounds: (id) => call('GET', `/survey/venue-rounds?profile_id=${id}`, null, venueRounds),
  postVenueChoice: (id, choice) => call('POST', `/profile/${id}/venue-choice`, choice, profile),
  tasteMap: (id) => call('GET', `/profile/${id}/taste-map`, null, tasteMap),
  recommend: (body) => call('POST', '/recommend', body, recommendations),
}
