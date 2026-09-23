import { useEffect, useRef, useState } from 'react'
import { postChat } from '../api'
import Results from './Results'

const OPENER = {
  role: 'bot',
  text:
    "Hello. Tell me about the animal and what you're seeing — for example " +
    '"my buffalo has had a fever for 2 days and won\'t eat". ' +
    "I'll pick out the clinical signs and give you an urgency assessment.",
}

const EXAMPLES = [
  'My buffalo has had fever for 2 days and loose motions',
  'Goat is dull, off her feed, no cough',
  'Cow limping with swollen jaw',
]

export default function ChatTriage({ schema }) {
  const [messages, setMessages] = useState([OPENER])
  const [caseState, setCaseState] = useState(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const endRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  const send = async (text) => {
    const message = (text ?? input).trim()
    if (!message || busy) return

    setMessages((prev) => [...prev, { role: 'user', text: message }])
    setInput('')
    setBusy(true)
    setError(null)

    try {
      const body = await postChat(message, caseState)
      setCaseState(body.case)
      setSuggestions(body.suggestions ?? [])
      if (body.prediction) setPrediction(body.prediction)
      if (body.intent === 'reset') setPrediction(null)
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          text: body.reply,
          unmatched: body.extracted?.unmatched_terms ?? [],
        },
      ])
    } catch (e) {
      setError(e.message)
      setMessages((prev) => [
        ...prev,
        { role: 'bot', text: null, failed: true },
      ])
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  const activeCase = caseState ?? {}
  const symptoms = activeCase.symptoms ?? []
  const negated = activeCase.negated ?? []

  return (
    <div className="chat-layout">
      <div className="card chat-card">
        <div className="chat-log" role="log" aria-live="polite">
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              {m.failed ? (
                <span className="error-text">
                  That message didn&apos;t reach the server. Check your connection
                  and try again.
                </span>
              ) : (
                <span dangerouslySetInnerHTML={{ __html: renderText(m.text) }} />
              )}
              {m.unmatched?.length > 0 && (
                <p className="bubble-note">
                  Not in the model&apos;s vocabulary: {m.unmatched.join(', ')}
                </p>
              )}
            </div>
          ))}
          {busy && (
            <div className="bubble bot typing" aria-label="Assistant is typing">
              <span /> <span /> <span />
            </div>
          )}
          <div ref={endRef} />
        </div>

        {messages.length === 1 && (
          <div className="examples">
            <span className="examples-label">Try:</span>
            {EXAMPLES.map((e) => (
              <button key={e} type="button" className="chip" onClick={() => send(e)}>
                {e}
              </button>
            ))}
          </div>
        )}

        {suggestions.length > 0 && (
          <div className="examples">
            <span className="examples-label">Also seeing?</span>
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                className="chip"
                onClick={() => send(`also ${s}`)}
              >
                + {s}
              </button>
            ))}
            <button type="button" className="chip" onClick={() => send('no, that is all')}>
              That&apos;s all
            </button>
          </div>
        )}

        <form
          className="chat-input"
          onSubmit={(e) => {
            e.preventDefault()
            send()
          }}
        >
          <input
            ref={inputRef}
            type="text"
            value={input}
            placeholder="Describe the animal and what you see…"
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
            aria-label="Message"
          />
          <button type="submit" className="primary" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
        {error && <p className="error-text">{error}</p>}
      </div>

      <div>
        <section className="card case-card">
          <h2>Case so far</h2>
          <dl className="kv">
            <dt>Animal</dt>
            <dd>{activeCase.species || <span className="muted">not yet given</span>}</dd>
            {activeCase.duration && (
              <>
                <dt>Duration</dt>
                <dd>{activeCase.duration}</dd>
              </>
            )}
            {activeCase.age_years != null && (
              <>
                <dt>Age</dt>
                <dd>{activeCase.age_years} years</dd>
              </>
            )}
          </dl>

          <h3>Signs reported</h3>
          {symptoms.length ? (
            <ul className="chips">
              {symptoms.map((s) => (
                <li key={s}>
                  <button type="button" onClick={() => send(`no ${s}`)} title={`Remove ${s}`}>
                    {s} <span aria-hidden="true">×</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="help">Nothing recorded yet.</p>
          )}

          {negated.length > 0 && (
            <>
              <h3>Ruled out</h3>
              <ul className="chips muted-chips">
                {negated.map((s) => (
                  <li key={s}>
                    <button type="button" onClick={() => send(`also ${s}`)} title={`Add ${s} back`}>
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </>
          )}

          <button
            type="button"
            className="ghost full"
            onClick={() => {
              setMessages([OPENER])
              setCaseState(null)
              setPrediction(null)
              setSuggestions([])
            }}
          >
            Start a new case
          </button>
        </section>

        {prediction && <Results result={prediction} schema={schema} />}
      </div>
    </div>
  )
}

// The bot uses **bold** for the headline verdict and • for lists. Escape first,
// then apply that tiny whitelist — never inject raw server text as HTML.
function renderText(text) {
  const escaped = String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  return escaped
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br />')
}
