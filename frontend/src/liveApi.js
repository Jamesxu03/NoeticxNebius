const API = import.meta.env.VITE_API_URL || '/api'
export async function request(path, body, method = 'POST') {
  const form = body instanceof FormData
  const response = await fetch(API + path, {
    method,
    headers: form ? undefined : { 'Content-Type': 'application/json' },
    body: body ? form ? body : JSON.stringify(body) : undefined,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    const failure = new Error(typeof error.detail === 'string' ? error.detail : 'The food API could not complete that request. Please try again.')
    failure.status = response.status
    throw failure
  }
  return response.json()
}
