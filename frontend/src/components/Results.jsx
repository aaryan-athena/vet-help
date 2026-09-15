const pct = (p) => `${(p * 100).toFixed(1)}%`

function riskTone(label, kind) {
  if (kind !== 'risk') return 'neutral'
  return label.toLowerCase() === 'yes' ? 'alert' : 'calm'
}

export default function Results({ result, schema }) {
  const kind = schema.target_kind
  const heading =
    kind === 'risk' ? 'Urgency assessment' : 'Most likely conditions'
  const topLabel = result.top_prediction.label

  return (
    <div className="results">
      <section className="card">
        <h2>{heading}</h2>
        {kind === 'risk' && (
          <p className="lede">
            This model was trained on a <strong>{result.model.target_column}</strong>{' '}
            flag, so it estimates whether a case looks serious enough to warrant
            veterinary attention — not which specific disease it is.
          </p>
        )}

        <ul className="prediction-list">
          {result.predictions.map((p) => (
            <li key={p.label}>
              <div className="prediction-head">
                <span className={`label tone-${riskTone(p.label, kind)}`}>
                  {kind === 'risk'
                    ? p.label.toLowerCase() === 'yes'
                      ? 'Needs veterinary attention'
                      : 'Lower concern'
                    : p.label}
                </span>
                <span className="probability">{pct(p.probability)}</span>
              </div>
              <div className="bar" role="img" aria-label={`${pct(p.probability)} probability`}>
                <div
                  className={`bar-fill tone-${riskTone(p.label, kind)}`}
                  style={{ width: `${Math.max(p.probability * 100, 1.5)}%` }}
                />
              </div>
            </li>
          ))}
        </ul>

        {result.unrecognized_symptoms.length > 0 && (
          <p className="notice">
            Not in the training vocabulary:{' '}
            <strong>{result.unrecognized_symptoms.join(', ')}</strong>. These
            still count towards overall symptom burden, but the model has no
            specific evidence about them.
          </p>
        )}
      </section>

      <section className="card">
        <h2>Why this result</h2>
        {result.explanation.length === 0 ? (
          <p className="help">
            No individual input moved this prediction appreciably.
          </p>
        ) : (
          <>
            <p className="lede">
              SHAP contributions for the “{topLabel}” outcome. Bars to the right
              pushed the score up; bars to the left pushed it down.
            </p>
            <ul className="explanation-list">
              {result.explanation.map((item) => {
                const max = Math.max(
                  ...result.explanation.map((e) => Math.abs(e.contribution)),
                )
                const width = (Math.abs(item.contribution) / max) * 50
                return (
                  <li key={item.feature}>
                    <span className="explain-label">{item.label}</span>
                    <span className="diverging">
                      <span className="axis" />
                      <span
                        className={`diverging-fill ${item.direction}`}
                        style={{
                          width: `${width}%`,
                          [item.direction === 'increases' ? 'left' : 'right']: '50%',
                        }}
                      />
                    </span>
                    <span className="explain-value">
                      {item.contribution > 0 ? '+' : ''}
                      {item.contribution.toFixed(2)}
                    </span>
                  </li>
                )
              })}
            </ul>
          </>
        )}
      </section>

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  )
}
