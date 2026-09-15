// Thin API client. In dev the Vite proxy forwards /api -> http://127.0.0.1:8000,
// so no CORS round-trip is needed; set VITE_API_BASE to talk to a remote backend.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

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
export const getModelInfo = () => request('/model-info')
export const getHealth = () => request('/health')
export const postPredict = (payload) =>
  request('/predict', { method: 'POST', body: JSON.stringify(payload) })

export { ApiError }
