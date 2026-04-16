# DST Sector Health Forecaster
**MSc BDS — MLOps Exam | Statistics Denmark**

An end-to-end MLOps pipeline that fetches live data from the **Statistics Denmark (DST) StatBank API** and forecasts a **Sector Vitality Score (SVS)** for each of the 10 Danish industry sectors one quarter ahead. The pipeline can be adjusted to run automatically on a quarterly cron schedule via GitHub Actions and exposes predictions through a FastAPI REST endpoint.

> **Data source:** [Statistics Denmark StatBank API](https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api) — free, no authentication required, CC 4.0 BY licence.

---

## Repository Structure

```
MLOps-Exam-Assignment/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml          ← GitHub Actions: (https://github.com/Ory999/MLOps-Exam-Assignment/actions/runs/24525069469)
│
├── artifacts/                    ← All pipeline outputs (contents gitignored)
│   ├── raw/                      ← Timestamped raw CSVs from DST API
│   ├── processed/                ← Quarterly panel + feature datasets
│   ├── models/                   ← Trained model .joblib files
│   ├── metrics/                  ← JSON metrics per training run
│   ├── reports/                  ← Monitoring reports + visualisation PNGs
│   └── mlflow/                   ← MLflow experiment tracking database
│
├── notebooks/
│   ├── Workbook_1_data_ingestion.ipynb         ← Explore DST API, fetch raw data
│   ├── Workbook_2_preprocessing_features.ipynb ← Build panel, vitality score, features
│   ├── Workbook_3_model_training.ipynb         ← Train Naive/Ridge/XGBoost, MLflow
│   └── Workbook_4_monitoring.ipynb             ← Drift detection, performance monitoring
│
├── src/
│   ├── __init__.py               ← Makes src/ a Python package
│   ├── dst_client.py             ← DST StatBank API client (KONK4, DEMO14)
│   ├── preprocessing.py          ← Monthly→quarterly aggregation, vitality score
│   ├── features.py               ← Lag/rolling features, time-based train/test split
│   ├── model.py                  ← Naive baseline, Ridge, XGBoost, MLflow logging
│   ├── monitoring.py             ← KS-test drift detection, MAE performance monitoring
│   └── api.py                    ← FastAPI: /predict, /monitoring, /pipeline/run
│
├── .gitignore                    ← Ignores artifact contents, keeps folder structure
├── config.yaml                   ← All pipeline configuration in one place
├── Dockerfile                    ← Python 3.11-slim image for pipeline + API
├── docker-compose.yml            ← Services: pipeline, api, mlflow
├── requirements.txt              ← All Python dependencies with pinned versions
└── run_pipeline.py               ← End-to-end runner: ingest → preprocess → train → monitor
```

---

## What the Pipeline Does

The pipeline forecasts a **Sector Vitality Score (SVS)** — a normalised [0, 1] index measuring the balance between enterprise births and bankruptcies in each sector. A score near 1 indicates sector expansion; near 0 indicates contraction.

**Five steps run sequentially:**

```
┌────────────────────────────────────────────────────────────┐
│  GitHub Actions (cron: Feb/May/Aug/Nov 10th at 09:00 UTC)  │
│  OR: POST /pipeline/run  OR: python run_pipeline.py        │
└────────────────────────┬───────────────────────────────────┘
                         │
      ┌──────────────────▼─────────────────────┐
      │  1. DATA INGESTION                      │
      │  DST StatBank API (free, no auth)       │
      │  • KONK4: bankruptcies (monthly)        │
      │  • DEMO14: enterprise births (annual)   │
      │  Output: artifacts/raw/*.csv            │
      └──────────────────┬─────────────────────┘
                         │
      ┌──────────────────▼─────────────────────┐
      │  2. PREPROCESSING                       │
      │  • Aggregate monthly → quarterly        │
      │  • Build (sector × quarter) panel       │
      │  • Forward-fill annual DEMO14 data      │
      │  • Filter to 2019–2023 (valid window)   │
      │  • Compute Sector Vitality Score [0,1]  │
      │  Output: artifacts/processed/panel_*   │
      └──────────────────┬─────────────────────┘
                         │
      ┌──────────────────▼─────────────────────┐
      │  3. FEATURE ENGINEERING                 │
      │  • Lag features t-1 … t-4              │
      │  • Rolling mean/std (4Q, 8Q windows)   │
      │  • Net growth rate, momentum            │
      │  • Sector dummies (10), seasonality     │
      │  • Time-based train/test split (4Q)     │
      │  Output: artifacts/processed/features_*│
      └──────────────────┬─────────────────────┘
                         │
      ┌──────────────────▼─────────────────────┐
      │  4. MODEL TRAINING                      │
      │  • Naive baseline (persistence)         │
      │  • Ridge regression (linear baseline)   │
      │  • XGBoost (primary model)              │
      │  • All runs logged to MLflow            │
      │  Output: artifacts/models/model_latest  │
      └──────────────────┬─────────────────────┘
                         │
      ┌──────────────────▼─────────────────────┐
      │  5. MONITORING                          │
      │  • KS-test: data drift per feature      │
      │  • MAE degradation vs training baseline │
      │  • Prediction distribution shift        │
      │  Output: artifacts/reports/monitoring_* │
      └─────────────────────────────────────────┘
```

---

## Data Sources

**Statistics Denmark (Danmarks Statistik) StatBank API**
- Base URL: `https://api.statbank.dk/v1`
- No authentication required — CC 4.0 BY licence
- Documentation: https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api

| Table | Description | Frequency | Coverage |
|-------|-------------|-----------|----------|
| `KONK4` | Bankruptcies by industry (DB07), active companies only (K02) | Monthly | 2009–present |
| `DEMO14` | Enterprise births by industry (DB07 10-grouping) | Annual | 2019–2023 |

**Important:** DEMO14 has a ~2-year publication lag. Data for 2024+ is not yet published, which is why the pipeline restricts the panel to 2019–2023. Pre-2019 rows have zero enterprise births (DEMO14 does not cover that period), which would distort the vitality score normalisation if included.

---

## Model Results (2019–2023, test set = 2023)

| Model | MAE | RMSE | R² | Directional Accuracy |
|-------|-----|------|----|----------------------|
| **Naive (persistence)** | **0.0952** | **0.1354** | -9.87 | **90%** |
| Ridge | 0.1123 | 0.1443 | -11.34 | 87% |
| XGBoost | 0.2473 | 0.2614 | -39.48 | 40% |

The **Naive model wins**. With only ~10 training quarters per sector after lagging, XGBoost cannot generalise — it overfits the 2019–2022 growth phase and fails to extrapolate the 2022–2023 decline. Negative R² is expected at this data volume; the pipeline is designed to improve automatically as DST publishes new DEMO14 data each year.

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Ory999/MLOps-Exam-Assignment.git
cd MLOps-Exam-Assignment
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
Fetches fresh data from DST, preprocesses, trains all three models, and saves a monitoring report to `artifacts/reports/`.

### 3. Explore notebooks step by step
```bash
jupyter notebook notebooks/
```
Open in order: Workbook_1 → Workbook_2 → Workbook_3 → Workbook_4

### 4. Start the API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Monitoring report: http://localhost:8000/monitoring

### 5. View MLflow experiments
```bash
mlflow ui --backend-store-uri ./artifacts/mlflow
```
Open http://localhost:5000 to compare runs.

---

## Docker

### Run pipeline then start services
```bash
# Step 1: train the model
docker compose run pipeline

# Step 2: start API + MLflow UI
docker compose up api mlflow
```

### Individual commands
```bash
docker build -t dst-sector-health .

# Run pipeline
docker run -v $(pwd)/artifacts:/app/artifacts dst-sector-health python run_pipeline.py

# Run API
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts dst-sector-health
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| GET | `/model/info` | Model metadata and training metrics |
| POST | `/predict` | Predict next-quarter vitality score for a sector |
| GET | `/monitoring` | Latest monitoring report (JSON) |
| POST | `/pipeline/run` | Trigger full pipeline run in background |

### Example prediction request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sector": 5,
    "vitality_score_lag1": 0.62,
    "vitality_score_lag2": 0.58,
    "vitality_score_lag3": 0.55,
    "vitality_score_lag4": 0.60,
    "bankruptcy_rate_lag1": 0.018,
    "birth_rate_lag1": 0.045,
    "quarter_of_year": 2,
    "year": 2025
  }'
```

**Sector codes (DB07 10-grouping):**

| Code | Sector |
|------|--------|
| 1 | Agriculture, forestry and fishing |
| 2 | Manufacturing, mining and quarrying |
| 3 | Energy, water supply, sewerage and waste |
| 4 | Construction |
| 5 | Trade and transport |
| 6 | Information and communication |
| 7 | Financial and insurance activities |
| 8 | Real estate |
| 9 | Other business services |
| 10 | Public administration, education, health and culture |

---

## GitHub Actions — Automated Pipeline

The pipeline runs automatically via `.github/workflows/pipeline.yml`:

| Trigger | When |
|---------|------|
| Scheduled cron | 10th of Feb, May, Aug, Nov at 09:00 UTC (aligned to DST release calendar) |
| Push to `main` | When `src/**`, `config.yaml`, or `run_pipeline.py` changes |
| Manual | Via the "Run workflow" button in the Actions tab |

Each run fetches fresh DST data, retrains the model, and uploads artifacts (model, metrics, monitoring report, charts) as downloadable bundles retained for 90 days.

---

## Monitoring Strategy

The pipeline monitors two things after every run:

**Data drift (KS-test):** Each feature is compared between the training distribution and the new incoming data using the Kolmogorov-Smirnov test. If more than 30% of features have a p-value below 0.15, an overall drift flag is raised.

**Performance degradation:** The current MAE is compared to the MAE at training time. If the current MAE exceeds the baseline by more than 20%, a retrain flag is raised.

Results are saved to `artifacts/reports/monitoring_latest.json` and served via the `/monitoring` endpoint.

---

## Artifact Storage

| Artifact | Location | Gitignored? |
|----------|----------|-------------|
| Raw CSVs from DST | `artifacts/raw/` | Yes (contents) |
| Processed panel | `artifacts/processed/` | Yes (contents) |
| Trained models | `artifacts/models/` | Yes (contents) |
| Metrics JSON | `artifacts/metrics/` | Yes (contents) |
| Monitoring reports + charts | `artifacts/reports/` | Yes (contents) |
| MLflow tracking database | `artifacts/mlflow/` | Yes (entirely) |
| Folder structure | `artifacts/*/.gitkeep` | No — tracked to preserve structure on clone |

Artifact contents are not stored in the repository. On a GitHub Actions run they are uploaded as downloadable bundles. Locally they are written to disk and read by the API.

---

## Configuration

All pipeline parameters are in `config.yaml`:

```yaml
model:
  test_quarters: 4        # hold out last 4 quarters (1 year) for evaluation
  lag_periods: [1,2,3,4]  # autoregressive lag features
  rolling_windows: [4,8]  # rolling mean/std window sizes

monitoring:
  drift_threshold: 0.15   # KS p-value below which a feature is flagged as drifted
  performance_threshold: 0.20  # MAE increase above which retraining is triggered
```
