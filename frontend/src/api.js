// Thin API client. In dev the Vite proxy forwards /api -> http://127.0.0.1:8000,
// so no CORS round-trip is needed; set VITE_API_BASE to talk to a remote backend.

/**
 * Normalise the configured API base.
 *
 * A trailing slash is the single most common way this deployment breaks: with
 * `VITE_API_BASE=https://api.example.com/` the client would request
 * `https://api.example.com//schema`, which the platform answers with a 308
 * redirect that carries NO CORS headers. Browsers block a cross-origin redirect
 * without them, so `fetch` rejects outright and the app reports the backend as
 * unreachable even though it is perfectly healthy.
 *
 * Stripping trailing slashes here makes the app immune to how the environment
 * variable happens to be typed.
 */
export function normalizeBase(raw) {
  const value = (raw ?? '/api').trim()
  return value.replace(/\/+$/, '')
}

const BASE = normalizeBase(import.meta.env?.VITE_API_BASE)

class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch {
    // Distinguish "backend not started locally" from "VITE_API_BASE not set on
    // the deployment", which are the two ways this realistically fails.
    const configured = Boolean(import.meta.env.VITE_API_BASE)
    throw new ApiError(
      configured
        ? `Cannot reach the VetDx API at ${BASE}. The backend may be starting up (cold start) or unreachable.`
        : 'Cannot reach the VetDx API. Locally, start it with `uvicorn backend.app.main:app --port 8000`. On a deployment, set the VITE_API_BASE environment variable to your API URL and redeploy.',
      0,
    )
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') {
        detail = body.detail
      } else if (Array.isArray(body.detail)) {
        // FastAPI/Pydantic validation errors.
        detail = body.detail
          .map((e) => `${(e.loc || []).slice(1).join('.') || 'input'}: ${e.msg}`)
          .join('; ')
      }
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(detail, response.status)
  }
  return response.json()
}

export const getSchema = () => request('/schema')
export const postChat = (message, caseState) =>
  request('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, case: caseState ?? null }),
  })
export const getBreeds = () => request('/breeds')
export const getBreed = (name) => request(`/breeds/${encodeURIComponent(name)}`)
export const postBreedRecommend = (query) =>
  request('/breeds/recommend', { method: 'POST', body: JSON.stringify(query) })
export const getModelInfo = () => request('/model-info')
export const getHealth = () => request('/health')
export const postPredict = (payload) =>
  request('/predict', { method: 'POST', body: JSON.stringify(payload) })

export { ApiError }
