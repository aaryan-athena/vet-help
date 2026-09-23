import { useEffect, useState } from 'react'
import { getSchema, postPredict } from './api'
import SymptomForm from './components/SymptomForm'
import Results from './components/Results'
import ModelInfo from './components/ModelInfo'
import ChatTriage from './components/ChatTriage'
import BreedAdvisor from './components/BreedAdvisor'

export default function App() {
  const [tab, setTab] = useState('chat')
  const [schema, setSchema] = useState(null)
  const [schemaError, setSchemaError] = useState(null)
  const [result, setResult] = useState(null)
  const [predictError, setPredictError] = useState(null)
  const [busy, setBusy] = useState(false)

  const loadSchema = () => {
    setSchemaError(null)
    setSchema(null)
    getSchema().then(setSchema).catch((e) => setSchemaError(e.message))
  }

  useEffect(loadSchema, [])

  const handleSubmit = async (payload) => {
    setBusy(true)
    setPredictError(null)
    try {
      setResult(await postPredict(payload))
    } catch (e) {
      setResult(null)
      setPredictError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true">
            ◈
          </span>
          <div>
            <h1>VetDx</h1>
            <p>Early-warning triage from clinical signs</p>
          </div>
        </div>
        <nav>
          {[
            ['chat', 'Chat triage'],
            ['triage', 'Symptom form'],
            ['breeds', 'Breed advisor'],
            ['model', 'About the model'],
          ].map(([key, label]) => (
            <button
              key={key}
              className={tab === key ? 'tab active' : 'tab'}
              onClick={() => setTab(key)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      <div className="banner" role="note">
        <strong>Decision support only.</strong> VetDx does not diagnose. Anything
        it reports must be confirmed by a qualified veterinarian.
      </div>

      <main>
        {tab === 'model' ? (
          <ModelInfo />
        ) : tab === 'breeds' ? (
          <BreedAdvisor />
        ) : schemaError ? (
          <div className="card error">
            <h2>Backend unavailable</h2>
            <p>{schemaError}</p>
            <p className="help">
              Start it with <code>uvicorn backend.app.main:app --port 8000</code>,
              then retry.
            </p>
            <button className="primary" onClick={loadSchema}>
              Retry
            </button>
          </div>
        ) : !schema ? (
          <div className="card skeleton">Loading input schema…</div>
        ) : tab === 'chat' ? (
          <ChatTriage schema={schema} />
        ) : (
          <div className="layout">
            <SymptomForm schema={schema} onSubmit={handleSubmit} busy={busy} />
            <div>
              {predictError && (
                <div className="card error">
                  <h2>Could not assess this case</h2>
                  <p>{predictError}</p>
                </div>
              )}
              {busy && <div className="card skeleton">Scoring case…</div>}
              {!busy && !predictError && !result && (
                <div className="card placeholder">
                  <h2>No case assessed yet</h2>
                  <p>
                    Pick the animal type and every sign you can observe, then run
                    the assessment. You will get ranked outcomes with
                    probabilities and the signs that drove them.
                  </p>
                </div>
              )}
              {!busy && result && <Results result={result} schema={schema} />}
            </div>
          </div>
        )}
      </main>

      <footer>
        <p>
          VetDx · model artifacts served from <code>/model-info</code> ·{' '}
          {schema ? `${schema.symptom_vocabulary.length} recorded signs` : ''}
        </p>
      </footer>
    </div>
  )
}
