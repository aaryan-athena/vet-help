import { useMemo, useState } from 'react'

// Every control on this form is generated from GET /schema, so adding a column
// to the training CSV surfaces a new input without touching this file.
export default function SymptomForm({ schema, onSubmit, busy }) {
  const [values, setValues] = useState({})
  const [symptoms, setSymptoms] = useState([])
  const [customSymptom, setCustomSymptom] = useState('')
  const [filter, setFilter] = useState('')
  const [touched, setTouched] = useState(false)

  const symptomField = schema.fields.find((f) => f.name === 'symptoms')
  const otherFields = schema.fields.filter((f) => f.name !== 'symptoms')

  const options = useMemo(() => {
    const q = filter.trim().toLowerCase()
    const list = symptomField?.options ?? []
    return q ? list.filter((o) => o.value.includes(q)) : list
  }, [filter, symptomField])

  const setValue = (name, value) =>
    setValues((prev) => ({ ...prev, [name]: value }))

  const toggleSymptom = (value) =>
    setSymptoms((prev) =>
      prev.includes(value) ? prev.filter((s) => s !== value) : [...prev, value],
    )

  const addCustomSymptom = () => {
    const term = customSymptom.trim().toLowerCase()
    if (term && !symptoms.includes(term)) setSymptoms([...symptoms, term])
    setCustomSymptom('')
  }

  const speciesMissing =
    schema.fields.some((f) => f.name === 'species' && f.required) && !values.species
  const invalid = symptoms.length === 0 || speciesMissing

  const handleSubmit = (event) => {
    event.preventDefault()
    setTouched(true)
    if (invalid || busy) return

    const payload = { symptoms, top_n: Math.min(3, schema.classes.length) }
    const indicators = {}
    for (const field of otherFields) {
      const raw = values[field.name]
      if (raw === undefined || raw === '') continue
      if (field.type === 'number') payload[field.name] = Number(raw)
      else if (field.type === 'yesno') indicators[field.name] = raw
      else payload[field.name] = raw
    }
    if (Object.keys(indicators).length) payload.indicators = indicators
    onSubmit(payload)
  }

  const reset = () => {
    setValues({})
    setSymptoms([])
    setFilter('')
    setTouched(false)
  }

  return (
    <form className="card form" onSubmit={handleSubmit}>
      <h2>Case details</h2>

      {otherFields.map((field) => (
        <div className="field" key={field.name}>
          <label htmlFor={field.name}>
            {field.label}
            {field.required && <span className="required"> *</span>}
          </label>

          {field.type === 'select' && (
            <select
              id={field.name}
              value={values[field.name] ?? ''}
              onChange={(e) => setValue(field.name, e.target.value)}
            >
              <option value="">Select…</option>
              {field.options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
              {field.allow_other && <option value="other">Other / not listed</option>}
            </select>
          )}

          {field.type === 'number' && (
            <input
              id={field.name}
              type="number"
              min={field.min}
              max={field.max}
              step={field.step}
              value={values[field.name] ?? ''}
              onChange={(e) => setValue(field.name, e.target.value)}
            />
          )}

          {field.type === 'text' && (
            <input
              id={field.name}
              type="text"
              placeholder={field.placeholder}
              value={values[field.name] ?? ''}
              onChange={(e) => setValue(field.name, e.target.value)}
            />
          )}

          {field.type === 'yesno' && (
            <div className="radio-row">
              {['Yes', 'No'].map((option) => (
                <label key={option} className="radio">
                  <input
                    type="radio"
                    name={field.name}
                    value={option}
                    checked={values[field.name] === option}
                    onChange={(e) => setValue(field.name, e.target.value)}
                  />
                  {option}
                </label>
              ))}
            </div>
          )}

          {field.help && <p className="help">{field.help}</p>}
          {touched && field.name === 'species' && speciesMissing && (
            <p className="error-text">Choose an animal type.</p>
          )}
        </div>
      ))}

      {symptomField && (
        <div className="field">
          <label htmlFor="symptom-filter">
            {symptomField.label}
            <span className="required"> *</span>
          </label>
          <input
            id="symptom-filter"
            type="search"
            placeholder={`Search ${symptomField.options.length} recorded signs…`}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />

          {symptoms.length > 0 && (
            <ul className="chips">
              {symptoms.map((s) => (
                <li key={s}>
                  <button type="button" onClick={() => toggleSymptom(s)}>
                    {s} <span aria-hidden="true">×</span>
                    <span className="sr-only">remove {s}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="symptom-list" role="group" aria-label="Symptoms">
            {options.length === 0 && (
              <p className="help">No recorded sign matches “{filter}”. Add it below.</p>
            )}
            {options.map((o) => (
              <label key={o.value} className="checkbox">
                <input
                  type="checkbox"
                  checked={symptoms.includes(o.value)}
                  onChange={() => toggleSymptom(o.value)}
                />
                <span>{o.label}</span>
                <span className="count">{o.count}</span>
              </label>
            ))}
          </div>

          <div className="inline-add">
            <input
              type="text"
              placeholder="Other sign not listed…"
              value={customSymptom}
              onChange={(e) => setCustomSymptom(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  addCustomSymptom()
                }
              }}
            />
            <button type="button" className="ghost" onClick={addCustomSymptom}>
              Add
            </button>
          </div>

          {symptomField.help && <p className="help">{symptomField.help}</p>}
          {touched && symptoms.length === 0 && (
            <p className="error-text">Select at least one symptom.</p>
          )}
        </div>
      )}

      <div className="actions">
        <button type="submit" className="primary" disabled={busy}>
          {busy ? 'Assessing…' : 'Assess case'}
        </button>
        <button type="button" className="ghost" onClick={reset} disabled={busy}>
          Clear
        </button>
      </div>
    </form>
  )
}
