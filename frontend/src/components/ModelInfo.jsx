import { useEffect, useState } from 'react'
import { getModelInfo } from '../api'

const fmt = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(3))

export default function ModelInfo() {
  const [info, setInfo] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getModelInfo().then(setInfo).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="card error">{error}</div>
  if (!info) return <div className="card">Loading model details…</div>

  const test = info.test_metrics ?? {}
  const cm = info.confusion_matrix ?? {}

  return (
    <div className="results">
      <section className="card">
        <h2>Deployed model</h2>
        <dl className="kv">
          <dt>Algorithm</dt>
          <dd>{info.model_name}</dd>
          <dt>Trained</dt>
          <dd>{new Date(info.trained_at).toLocaleString()}</dd>
          <dt>Predicting</dt>
          <dd>
            <code>{info.target_column}</code> ({info.classes.join(' / ')})
          </dd>
          <dt>Training rows</dt>
          <dd>{info.n_training_rows}</dd>
          <dt>Features</dt>
          <dd>{info.n_features}</dd>
          <dt>Selected on</dt>
          <dd>{info.selection_metric}</dd>
        </dl>

        <h3>Held-out test performance</h3>
        <div className="metric-row">
          <div className="metric">
            <span className="metric-value">{fmt(test.accuracy)}</span>
            <span className="metric-label">Accuracy</span>
          </div>
          <div className="metric">
            <span className="metric-value">{fmt(test.macro_f1)}</span>
            <span className="metric-label">Macro F1</span>
          </div>
          <div className="metric">
            <span className="metric-value">{fmt(test.balanced_accuracy)}</span>
            <span className="metric-label">Balanced acc.</span>
          </div>
          <div className="metric">
            <span className="metric-value">{fmt(test.roc_auc)}</span>
            <span className="metric-label">ROC AUC</span>
          </div>
        </div>
        <p className="help">
          Accuracy is flattered by class imbalance ({' '}
          {Object.entries(info.class_counts ?? {})
            .map(([k, v]) => `${k}: ${v}`)
            .join(', ')}
          ). Macro F1 and balanced accuracy are the honest numbers.
        </p>
      </section>

      <section className="card">
        <h2>Model comparison</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Macro F1</th>
                <th>Balanced acc.</th>
                <th>ROC AUC</th>
                <th>CV macro F1</th>
              </tr>
            </thead>
            <tbody>
              {(info.comparison ?? []).map((row) => (
                <tr
                  key={row.model}
                  className={row.model === info.model_name ? 'selected' : undefined}
                >
                  <td>
                    {row.model}
                    {row.model === info.model_name && <span className="pill">deployed</span>}
                  </td>
                  <td>{fmt(row.accuracy)}</td>
                  <td>{fmt(row.macro_f1)}</td>
                  <td>{fmt(row.balanced_accuracy)}</td>
                  <td>{fmt(row.roc_auc)}</td>
                  <td>{fmt(row.cv_macro_f1_mean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="help">Scores on the validation split; CV is 5-fold on train+validation.</p>
      </section>

      {cm.matrix && (
        <section className="card">
          <h2>Test confusion matrix</h2>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>actual \ predicted</th>
                  {cm.labels.map((l) => (
                    <th key={l}>{l}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cm.matrix.map((row, i) => (
                  <tr key={cm.labels[i]}>
                    <th>{cm.labels[i]}</th>
                    {row.map((n, j) => (
                      <td key={j} className={i === j ? 'diag' : undefined}>
                        {n}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="card">
        <h2>What the model leans on</h2>
        <p className="lede">
          Global importance ({info.explanation_method}) across the training set.
        </p>
        <ul className="explanation-list global">
          {(info.global_importance ?? []).slice(0, 12).map((f) => {
            const max = info.global_importance[0]?.importance || 1
            return (
              <li key={f.feature}>
                <span className="explain-label">{f.label}</span>
                <span className="bar">
                  <span
                    className="bar-fill tone-neutral"
                    style={{ width: `${(f.importance / max) * 100}%` }}
                  />
                </span>
              </li>
            )
          })}
        </ul>
      </section>

      <section className="card">
        <h2>Known limitations</h2>
        <ul className="limits">
          <li>
            Trained on {info.dataset?.n_clean_rows ?? '—'} cleaned case records —
            small for a clinical model.
          </li>
          <li>
            Severe class imbalance ({info.dataset?.imbalance_ratio ?? '—'}× between
            the largest and smallest class), so minority-class estimates rest on
            very few examples.
          </li>
          {(info.dataset?.fields_absent_from_dataset ?? []).length > 0 && (
            <li>
              The source CSV carries no{' '}
              {info.dataset.fields_absent_from_dataset.join(', ')} column, so those
              clinically relevant inputs are not used.
            </li>
          )}
          <li>
            Symptoms were recorded as free text with inconsistent spelling; the
            cleaning step merges obvious variants but cannot fix label noise.
          </li>
          <li>Species coverage and geography of the source data are unknown.</li>
        </ul>
        <p className="disclaimer">{info.disclaimer}</p>
      </section>
    </div>
  )
}
