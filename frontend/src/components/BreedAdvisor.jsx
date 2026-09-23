import { useEffect, useMemo, useState } from 'react'
import { getBreed, getBreeds, postBreedRecommend } from '../api'

const CLIMATE_PRESETS = [
  'Hot and humid',
  'Hot and dry',
  'Hot, semi-arid',
  'Temperate',
  'Cool and hilly',
  'Heavy rainfall, hilly',
  'Arid, saline',
]

export default function BreedAdvisor() {
  const [mode, setMode] = useState('conditions')
  const [catalog, setCatalog] = useState(null)
  const [loadError, setLoadError] = useState(null)

  useEffect(() => {
    getBreeds().then(setCatalog).catch((e) => setLoadError(e.message))
  }, [])

  if (loadError) return <div className="card error">{loadError}</div>
  if (!catalog) return <div className="card skeleton">Loading breed catalogue…</div>

  return (
    <div className="results">
      <section className="card">
        <h2>Breed advisor</h2>
        <p className="lede">
          {catalog.facets.n_breeds} cattle and buffalo breeds, with the climate each
          one is actually suited to and its typical milk yield.
        </p>
        <div className="mode-switch" role="tablist">
          <button
            role="tab"
            aria-selected={mode === 'conditions'}
            className={mode === 'conditions' ? 'tab active' : 'tab'}
            onClick={() => setMode('conditions')}
          >
            I have conditions → find a breed
          </button>
          <button
            role="tab"
            aria-selected={mode === 'breed'}
            className={mode === 'breed' ? 'tab active' : 'tab'}
            onClick={() => setMode('breed')}
          >
            I have a breed → find its conditions
          </button>
        </div>
      </section>

      {mode === 'conditions' ? (
        <ConditionsToBreed facets={catalog.facets} />
      ) : (
        <BreedToConditions breeds={catalog.breeds} />
      )}

      <p className="disclaimer">{catalog.facets.caveat}</p>
    </div>
  )
}

function ConditionsToBreed({ facets }) {
  const [climate, setClimate] = useState('')
  const [purpose, setPurpose] = useState('dairy')
  const [type, setType] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const submit = async (e) => {
    e?.preventDefault()
    setBusy(true)
    setError(null)
    try {
      setResult(await postBreedRecommend({ climate, purpose, type, top_n: 6 }))
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <form className="card form" onSubmit={submit}>
        <div className="field">
          <label htmlFor="climate">Your climate</label>
          <input
            id="climate"
            type="text"
            value={climate}
            placeholder="e.g. hot and humid, coastal"
            onChange={(e) => setClimate(e.target.value)}
          />
          <div className="examples">
            {CLIMATE_PRESETS.map((c) => (
              <button key={c} type="button" className="chip" onClick={() => setClimate(c)}>
                {c}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label htmlFor="purpose">What do you want the animal for?</label>
          <select id="purpose" value={purpose} onChange={(e) => setPurpose(e.target.value)}>
            <option value="dairy">Milk (dairy)</option>
            <option value="draught">Work / ploughing (draught)</option>
            <option value="dual">Both</option>
            <option value="">No preference</option>
          </select>
        </div>

        <div className="field">
          <label htmlFor="type">Animal type</label>
          <select id="type" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">Any</option>
            {facets.types.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            {busy ? 'Matching…' : 'Find breeds'}
          </button>
        </div>
      </form>

      {error && <div className="card error">{error}</div>}

      {result && (
        <section className="card">
          <h2>Recommended breeds</h2>
          <p className="lede">
            {result.n_suitable} of {result.n_considered} breeds suit these
            conditions
            {result.query.climate_tags.length > 0 && (
              <> (read as: {result.query.climate_tags.join(', ').replace(/_/g, '-')})</>
            )}
            . Yield is only allowed to matter among breeds that actually fit the
            climate.
          </p>

          {result.recommendations.map((b) => (
            <article key={b.name} className={`breed ${b.suitable ? '' : 'unsuitable'}`}>
              <div className="breed-head">
                <div>
                  <h3>
                    {b.name}
                    {!b.suitable && <span className="pill warn">not recommended here</span>}
                  </h3>
                  <p className="help">
                    {b.type} · {b.region} · kept for {b.utility.toLowerCase()}
                  </p>
                </div>
                <div className="breed-yield">
                  <span className="metric-value">{b.milk_yield_l_per_day ?? '—'}</span>
                  <span className="metric-label">L/day · {b.fat_raw}</span>
                </div>
              </div>
              <div className="bar">
                <div
                  className={`bar-fill ${b.suitable ? 'tone-calm' : 'tone-alert'}`}
                  style={{ width: `${Math.max(b.score * 100, 2)}%` }}
                />
              </div>
              <ul className="reasons">
                {b.reasons.map((r, i) => (
                  <li key={i} className={r.startsWith('⚠') ? 'warn' : undefined}>
                    {r}
                  </li>
                ))}
              </ul>
              {b.features && <p className="help">{b.features}</p>}
            </article>
          ))}
        </section>
      )}
    </>
  )
}

function BreedToConditions({ breeds }) {
  const [selected, setSelected] = useState('')
  const [profile, setProfile] = useState(null)
  const [error, setError] = useState(null)
  const sorted = useMemo(
    () => [...breeds].sort((a, b) => a.name.localeCompare(b.name)),
    [breeds],
  )

  useEffect(() => {
    if (!selected) {
      setProfile(null)
      return
    }
    getBreed(selected).then(setProfile).catch((e) => setError(e.message))
  }, [selected])

  return (
    <>
      <div className="card form">
        <div className="field">
          <label htmlFor="breed">Choose a breed</label>
          <select id="breed" value={selected} onChange={(e) => setSelected(e.target.value)}>
            <option value="">Select…</option>
            {sorted.map((b) => (
              <option key={b.name} value={b.name}>
                {b.name} ({b.type})
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <div className="card error">{error}</div>}

      {profile && (
        <section className="card">
          <h2>{profile.name}</h2>
          <dl className="kv">
            <dt>Type</dt>
            <dd>{profile.type}</dd>
            <dt>Origin</dt>
            <dd>{profile.region}</dd>
            <dt>Best conditions</dt>
            <dd>
              <strong>{profile.ideal_conditions}</strong>
            </dd>
            <dt>Kept for</dt>
            <dd>{profile.utility}</dd>
            <dt>Physical traits</dt>
            <dd>{profile.traits}</dd>
            <dt>Breeding programmes</dt>
            <dd>{profile.programs}</dd>
          </dl>

          <div className="metric-row">
            <div className="metric">
              <span className="metric-value">{profile.milk_yield_l_per_day ?? '—'}</span>
              <span className="metric-label">L/day</span>
            </div>
            <div className="metric">
              <span className="metric-value">{profile.fat_percent ?? '—'}%</span>
              <span className="metric-label">milk fat</span>
            </div>
          </div>

          {profile.guidance.length > 0 && (
            <>
              <h3>Keeping this breed well</h3>
              <ul className="limits">
                {profile.guidance.map((g, i) => (
                  <li key={i}>{g}</li>
                ))}
              </ul>
            </>
          )}
          {profile.features && <p className="help">{profile.features}</p>}
        </section>
      )}
    </>
  )
}
