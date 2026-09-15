# VetDx — ML-driven early-warning triage for animals

VetDx takes an animal's profile and the clinical signs someone can actually
observe, and returns a ranked, probability-weighted assessment with an
explanation of which signs drove it. It is aimed at the case the proposal
describes: rural and resource-limited settings where a vet is not immediately
reachable and early signs get missed.

> **Decision support only — not a diagnostic tool.** VetDx predictions are
> statistical associations learned from a small, noisy, publicly sourced
> dataset. They are not a substitute for examination and diagnosis by a
> qualified veterinarian. Always seek professional care for a sick animal.

---

## ⚠️ Read this first: what the shipped dataset actually contains

The proposal assumes a dataset with animal profile, breed, age, weight, symptom
duration and a **diagnosed disease** label. The CSV at `data/data.csv` has
none of that. Its real schema is:

```
AnimalName, symptoms1, symptoms2, symptoms3, symptoms4, symptoms5, Dangerous
```

- **No disease label.** The only target column is `Dangerous` (Yes/No), so the
  deployed model answers *"does this case look serious enough to need a vet?"* —
  a severity triage — rather than *"which disease is this?"*.
- **Severely imbalanced.** 821 `Yes` to 20 `No`. Always answering "Yes" scores
  97.6% accuracy, so accuracy is a meaningless metric here; macro F1 and
  balanced accuracy are used for model selection and reporting.
- **No breed / age / weight / duration columns.**

The pipeline is written to adapt rather than to assume. It discovers the role of
every column at runtime (`ml/cleaning.py::discover_schema`) and prefers a
`disease` / `diagnosis` / `label` column when one exists, falling back to
`Dangerous`. Cleaning, parsing and feature code for breed, age, weight and
free-text duration is implemented and unit-tested — it activates automatically
the moment those columns appear in the CSV. **To switch to disease prediction,
drop in a CSV with a `Disease` column and re-run `python -m ml.train`.** No code
changes are needed; the API `/schema` and the React form both re-render from the
new artifacts.

Run `python -m ml.inspect_data` on any CSV to see exactly what the pipeline
inferred before training on it.

---

## Repo structure

```
vet-help/
├── data/
│   ├── data.csv                  # source CSV — drop your own here
│   └── processed/                # versioned outputs of the pipeline (gitignored)
│       ├── cases_clean.csv       #   cleaned case records
│       ├── features_trainval.csv #   engineered feature matrix (train+val)
│       └── features_test.csv     #   engineered feature matrix (test)
├── ml/
│   ├── config.py                 # every path, threshold and knob
│   ├── inspect_data.py           # step 0: what is actually in the CSV
│   ├── cleaning.py               # schema discovery, text canonicalisation, duration parsing
│   ├── features.py               # record frame + SymptomFeatureBuilder transformer
│   ├── eda.py                    # class balance, symptom frequency, co-occurrence
│   ├── train.py                  # model comparison, selection, artifact writing
│   ├── explain.py                # SHAP global + per-prediction explanations
│   ├── report.py                 # renders docs/evaluation_report.md from artifacts
│   └── artifacts/                # model.joblib, feature_schema.json, metrics, runs.jsonl
├── backend/app/
│   ├── main.py                   # FastAPI routes, CORS, structured logging
│   ├── model_service.py          # loads artifacts once, serves predictions
│   └── schemas.py                # Pydantic request/response contracts
├── frontend/                     # React + Vite single-page app
│   └── src/
│       ├── api.js                # fetch client with typed error handling
│       ├── App.jsx               # tabs, loading and error states
│       └── components/           # SymptomForm, Results, ModelInfo
├── tests/                        # unit + integration + deployment-guard tests
├── reports/                      # EDA figures and eda_summary.json (gitignored)
├── docs/evaluation_report.md     # generated model comparison write-up
├── requirements.txt              # SERVING deps only (what Vercel installs)
├── requirements-dev.txt          # + training, explainability, tests
├── pyproject.toml                # Vercel entrypoint + serving deps
├── vercel.json / .vercelignore   # keep the function bundle small
└── .python-version               # 3.12 on Vercel
```

---

## Quick start

### 0. Prerequisites

Python 3.10+ and Node 18+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
```

There are two dependency sets, and the split is deliberate:

| File | Contents | Installed by |
|---|---|---|
| `requirements.txt` | fastapi, pydantic, numpy, pandas, scikit-learn, joblib — **279 MB** | production / Vercel |
| `requirements-dev.txt` | the above **plus** xgboost, shap, matplotlib, uvicorn, pytest — **611 MB** | you, locally |

`shap` alone drags in `numba` + `llvmlite` (~150 MB). Keeping the training-only
packages out of the serving set is what brings the deployed bundle under
Vercel's limit — see [Deploying to Vercel](#deploying-to-vercel).

### 1. Look at your data first

```bash
python -m ml.inspect_data
```

Prints shape, dtypes, null counts, target value counts, the inferred column
roles, the symptom vocabulary before and after canonicalisation, and a warning
listing any proposal fields the CSV is missing.

### 2. Run the EDA

```bash
python -m ml.eda
```

Writes `reports/class_balance.png`, `reports/symptom_frequency.png`,
`reports/cooccurrence.png` and `reports/eda_summary.json`.

### 3. Train

```bash
python -m ml.train              # full run, including SHAP
python -m ml.train --no-shap    # skip the explainability pass (faster)
python -m ml.train --csv path/to/other.csv
```

This trains six models on identical folds with a small macro-F1 grid search per
model, picks the winner on validation macro F1, refits it on train+validation,
scores it once on the untouched test split, and writes:

| Artifact | Purpose |
|---|---|
| `ml/artifacts/model.joblib` | fitted feature builder + classifier, plus label encoder and schema |
| `ml/artifacts/feature_schema.json` | the input contract `GET /schema` serves |
| `ml/artifacts/evaluation_report.json` | full comparison, per-class metrics, confusion matrix |
| `ml/artifacts/model_info.json` | deployed-model summary for `GET /model-info` |
| `ml/artifacts/shap_global.json` | global feature importance |
| `ml/artifacts/runs.jsonl` | append-only experiment log (params + metrics per run) |

Then regenerate the write-up:

```bash
python -m ml.report     # -> docs/evaluation_report.md
```

### 4. Run the backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

| Endpoint | What it does |
|---|---|
| `GET /` | service descriptor — confirms the API is up and lists the routes |
| `GET /health` | liveness plus whether a model is loaded |
| `GET /schema` | field definitions, symptom vocabulary and valid values, so the form is generated not hardcoded |
| `GET /model-info` | deployed model, metrics, comparison table, confusion matrix, global importance |
| `POST /predict` | top-N outcomes with probabilities + SHAP drivers for the top one |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"species":"buffaloes","symptoms":["fever","lethargy","diarrhea"],"top_n":2}'
```

Malformed input returns `422` with a per-field reason. If the API is started
before training, `/health` reports `degraded` and the other routes return `503`
with instructions rather than crashing.

Every route is also served under an `/api` prefix (`/api/health`,
`/api/predict`, …), so the API works whether it has its own domain or sits
behind an `/api` path — and `VITE_API_BASE` is forgiving about a trailing
`/api`.

> **`{"detail":"Not Found"}` in the browser?** That is FastAPI's 404: the
> backend is running, the URL just has no route. The API root now answers with a
> service descriptor instead, so if you still see it, check the path — and
> remember the API is not the web interface, which is the separate frontend
> deployment.

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000`, so no CORS round-trip
is needed locally. To point at a backend elsewhere, copy `frontend/.env.example`
to `frontend/.env` and set `VITE_API_BASE`. For a non-proxied or deployed setup,
allow the frontend origin on the API with the `VETDX_CORS_ORIGINS` environment
variable (see [Deploying to Vercel](#deploying-to-vercel)).

`npm run build` produces a static bundle in `frontend/dist/`.

### 6. Run the tests

```bash
python -m pytest
```

84 tests: unit coverage of schema discovery, text canonicalisation, Yes/No
encoding, duration parsing and every branch of the feature builder, plus
integration tests that drive the real trained pipeline through the FastAPI app
(ranking, explanation, unknown-symptom handling, input validation, CORS). Two
suites guard the deployment specifically:

- `tests/test_explain.py` asserts the numpy SHAP closed form agrees with the
  `shap` package to float32 precision, and that contributions are additive.
- `tests/test_serving_deps.py` boots the API in a subprocess with `shap`,
  `xgboost` and `matplotlib` made unimportable — exactly the production
  environment — and asserts predictions and explanations are byte-identical.

Tests needing a trained model **skip** rather than fail if
`ml/artifacts/model.joblib` is absent, so a fresh clone can run the unit tests
before training.

---

## How it works

### Cleaning

`data/data.csv` records symptoms as free text with inconsistent casing, double
spaces, U+FFFD mojibake and clinical synonyms (`Pains`/`Pain`,
`Anorexia`/`Poor Appetite`/`stop eating`, `Dyspnea`/`Difficulty breathing`).
`normalize_text` strips the noise and a curated synonym table collapses the
equivalents, taking 935 distinct raw strings down to 826 canonical terms.
Species names get the same treatment (`Dogs`→`dog`, `cow`→`cattle`,
`other birds`→`birds`). Rows with a null target and exact duplicate rows are
dropped.

### Features

Training rows and live API requests are funnelled through one intermediate
**record frame** (species, symptom set, age, weight, duration, indicators), then
through a single scikit-learn transformer. That is what guarantees train/serve
parity — a request saying `"Pains"` lands on exactly the feature a training row
saying `"Pain"` did.

The transformer emits: multi-hot symptom flags (terms seen ≥3 times in the
training split), co-occurrence flags for the 25 most common symptom pairs,
symptom-burden counts including out-of-vocabulary terms, one-hot species/breed
with an `__other` bucket for unseen values, median-imputed numerics, and binned
age/duration. 196 features on the shipped dataset.

The vocabulary is fitted on the **training split only**, so validation and test
scores are not inflated by leakage from terms the model would not have seen.

### Models

Logistic regression and Bernoulli naive Bayes as baselines, random forest and
XGBoost as the stronger models, a small MLP for comparison, and a
majority-class dummy so it is obvious when a model has learnt nothing. All are
class-weighted, all get a small macro-F1 grid search, all are scored on
accuracy, macro F1, weighted F1, balanced accuracy and ROC AUC plus 5-fold CV.

### Explainability

SHAP, with a fast exact path (`TreeExplainer` for the ensembles,
`LinearExplainer` for logistic regression) and a permutation fallback.
`/predict` returns the top contributing features for the specific case,
restricted to inputs that are actually **present** — telling someone that "not
having seizures" moved the score is noise in a triage context. Global importance
is computed at the end of training and surfaced on the frontend's model page.

For a **linear** model — which is what is deployed — SHAP has a closed form,
`φᵢ = wᵢ·(xᵢ − E[xᵢ])`, so `ml/explain.py::linear_contributions` computes exact
values in numpy with no sampling and no `shap` import. That is both more
accurate than the sampling explainers and what keeps the serving bundle small.
The `shap` package is still used at training time for non-linear models, and
`tests/test_explain.py` pins the two implementations together.

---

## Results

Full write-up in [`docs/evaluation_report.md`](docs/evaluation_report.md),
regenerated from the artifacts by `python -m ml.report`.

Validation split, shipped `Dangerous` target:

| Model | Accuracy | Macro F1 | Balanced acc. | ROC AUC | CV macro F1 |
|---|---|---|---|---|---|
| `majority_baseline` | 0.978 | 0.494 | 0.500 | 0.500 | 0.494 |
| **`logistic_regression`** (deployed) | 0.993 | **0.927** | 0.996 | 0.998 | 0.822 |
| `naive_bayes` | 0.978 | 0.494 | 0.500 | 0.896 | 0.743 |
| `random_forest` | 0.970 | 0.492 | 0.496 | 0.977 | 0.745 |
| `neural_net` | 0.963 | 0.491 | 0.492 | 0.318 | 0.564 |
| `xgboost` | 0.985 | 0.746 | 0.667 | 0.990 | 0.614 |

Held-out test: accuracy 0.970, macro F1 0.715, balanced accuracy 0.741,
ROC AUC 0.976.

Three things worth reading carefully:

- Naive Bayes, random forest and the neural net all land on the majority-class
  baseline's macro F1 — on this data they learned to always answer "Yes".
  Heavily regularised logistic regression (`C=0.003`) is the only model that
  meaningfully separates the classes.
- The test macro F1 (0.715) is well below the validation figure (0.927) because
  the test split contains only **4** minority-class cases. The 5-fold CV macro
  F1 of 0.822 is the more stable estimate.
- Species is one of the strongest features. That reflects how the data was
  collected — minority-class cases cluster in a couple of species — not biology.

## Model artifacts and version control

`.gitignore` excludes virtualenvs, `node_modules`, `data/processed/` and
`reports/` — all regenerable by re-running the pipeline.

**`ml/artifacts/` is tracked in full, `model.joblib` included** (~22 KB, the
whole directory is ~92 KB). Vercel builds from the git repo and there is no
training step in the build, so a committed model is the only way the deployed
function can find one. The JSON artifacts also let the repo document what is
deployed without a retrain.

If you later switch to a large model — a tuned random forest can be tens of MB —
stop tracking the binary (`ml/artifacts/*.joblib`) and fetch it at build time or
from object storage instead.

---

## Deploying to Vercel

The repo deploys as **two Vercel projects from the same git repo**: a static
frontend and a Python API.

### Why two projects, not one

Vercel's Python runtime gives a detected framework preset precedence over
file-based `/api` functions — *"When Vercel detects a framework preset, the
framework application handles all requests"*. Because `requirements.txt` lists
FastAPI, a single project would route **every** request to Python and never
serve the React bundle. Two projects keeps each build honest. (Vercel's newer
[Services](https://vercel.com/docs/services) feature can combine them under one
domain if you would rather have that later.)

### Bundle size — the thing that would otherwise break the deploy

Vercel's Python bundle limit is **500 MB** uncompressed. The dependency split
above exists because of it:

| | Size | Fits? |
|---|---|---|
| Everything (`requirements-dev.txt`) | 611 MB | ✗ over the limit |
| Serving only (`requirements.txt`) | 279 MB | ✓ |

`shap` (+`numba`+`llvmlite`) and `xgboost` are 330 MB of that, and neither is
needed to serve: explanations use the exact numpy closed form, and the trained
model is already pickled. `tests/test_serving_deps.py` enforces this by booting
the API with those packages unimportable.

### 1. Deploy the API

New Project → import the repo → **Root Directory: repository root**.

Vercel reads `pyproject.toml` for the entrypoint
(`[tool.vercel] entrypoint = "backend.app.main:app"`), `requirements.txt` for
dependencies, `.python-version` for Python 3.12, and `vercel.json` +
`.vercelignore` to keep the frontend, tests and figures out of the bundle.

Set one environment variable **after** you know the frontend URL:

```
VETDX_CORS_ORIGINS = https://your-frontend.vercel.app
```

Preview deployments get generated `*.vercel.app` origins, which are allowed by a
regex; set `VETDX_CORS_ALLOW_VERCEL_PREVIEWS=0` to disable that and rely only on
the explicit list.

Verify: `curl https://your-api.vercel.app/health` → `{"status":"ok", ...}`.

### 2. Deploy the frontend

New Project → same repo → **Root Directory: `frontend`**. Vercel detects Vite
(`frontend/vercel.json` pins it). Set:

```
VITE_API_BASE = https://your-api.vercel.app
```

**Vite inlines env vars at build time**, so setting this requires a redeploy to
take effect — the single most common way this deployment goes wrong. If it is
unset, the app falls back to `/api` on its own domain, gets a 404, and shows an
error naming `VITE_API_BASE`.

Trailing slashes and an `/api` suffix are both tolerated: `src/api.js` strips
trailing slashes, and the API serves every route under `/api` as well. Worth
knowing *why* the slash mattered — with `https://host/` the client requested
`https://host//schema`, which Vercel answers with a **308 redirect that carries
no CORS headers**. Browsers block cross-origin redirects without them, so
`fetch` rejects before any status code is visible and the app reports
"backend unavailable" while the API is in fact perfectly healthy. Diagnose this
class of bug with:

```bash
curl -i -H "Origin: https://your-frontend.vercel.app" https://your-api.vercel.app/schema
# expect: 200 + Access-Control-Allow-Origin
# a 3xx, or a 200 with no Access-Control-Allow-Origin, is what the browser blocks
```

### 3. Then set `VETDX_CORS_ORIGINS` on the API and redeploy it

Chicken-and-egg: the API needs the frontend's final URL. Deploy the frontend
first if you want to do it in one pass.

### Serverless caveats worth knowing

- **Cold starts.** Importing pandas + scikit-learn and unpickling the model
  takes a few seconds on a cold function. The model loads once per instance via
  an `lru_cache` singleton and `/health` warms it.
- **Read-only filesystem.** Nothing is written at runtime — the background means
  needed for explanations are baked into `model.joblib` at training time rather
  than read from `data/processed/`.
- **Logs are ephemeral.** The structured per-request logs go to stdout and are
  visible in Vercel's log drain, not persisted. Attach a log drain if you want
  prediction history.
- **Training does not run on Vercel.** Retrain locally, commit the updated
  `ml/artifacts/`, and push.

## Limitations

Summarised here, in full in [`docs/evaluation_report.md`](docs/evaluation_report.md):

- **No disease labels in the shipped data** — the model predicts severity, not
  diagnosis.
- **841 cleaned rows**, with only 20 minority-class examples. Small for anything
  clinical.
- **No breed, age, weight or duration** columns, so genuinely predictive signals
  are unavailable.
- **Free-text label and symptom noise** that canonicalisation reduces but cannot
  eliminate.
- **Unknown provenance** — geographic and breed coverage of the source records
  is undocumented, so nothing here should be assumed to generalise.
- **Species confounding**, as above.

VetDx is a triage and decision-support aid. It does not diagnose, and its output
must be confirmed by a qualified veterinarian.
