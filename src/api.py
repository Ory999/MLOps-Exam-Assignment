"""
api.py
------
FastAPI deployment: exposes the trained model via HTTP endpoints.

Endpoints
---------
GET  /health          → service health check
GET  /model/info      → model metadata and training metrics
POST /predict         → predict vitality score for a sector next quarter
GET  /monitoring      → latest monitoring report
POST /pipeline/run    → trigger full pipeline (ingest → preprocess → train)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
app = FastAPI(
    title="DST Sector Health API",
    description="Forecasts Danish sector vitality scores using Statistics Denmark data",
    version="1.0.0",
)

# Lazy-load model on first request
_model_artifact = None


def get_model():
    global _model_artifact
    if _model_artifact is None:
        from src.model import load_model
        model_path = "./artifacts/models/model_latest.joblib"
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail="Model not yet trained. Run POST /pipeline/run first."
            )
        _model_artifact = load_model(model_path)
    return _model_artifact


# Schemas
class PredictRequest(BaseModel):
    sector: int = Field(..., ge=1, le=10, description="DB07 sector code (1–10). "
                        "1=Agriculture, 2=Manufacturing, 3=Energy/water, "
                        "4=Construction, 5=Trade/transport, 6=Finance/insurance, "
                        "7=Real estate, 8=Business services, "
                        "9=Public/education/health, 10=Culture/other")
    vitality_score_lag1: float = Field(..., ge=0, le=1, description="Vitality score last quarter")
    vitality_score_lag2: float = Field(..., ge=0, le=1, description="Vitality score 2 quarters ago")
    vitality_score_lag3: float = Field(..., ge=0, le=1, description="Vitality score 3 quarters ago")
    vitality_score_lag4: float = Field(..., ge=0, le=1, description="Vitality score 4 quarters ago")
    bankruptcy_rate_lag1: float = Field(0.02, ge=0, description="Bankruptcy rate last quarter")
    birth_rate_lag1: float = Field(0.05, ge=0, description="New enterprise rate last quarter")
    quarter_of_year: int = Field(..., ge=1, le=4, description="Quarter being forecast (1–4)")
    year: int = Field(..., ge=2015, description="Year being forecast")


class PredictResponse(BaseModel):
    sector: int
    predicted_vitality_score: float
    interpretation: str
    model_name: str


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_artifact is not None}


@app.get("/model/info")
def model_info():
    artifact = get_model()
    return {
        "model_name": artifact["model_name"],
        "trained_at": artifact["trained_at"],
        "n_features": len(artifact["feature_cols"]),
        "metrics": artifact["metrics"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    artifact = get_model()
    feature_cols = artifact["feature_cols"]

    # Build a one-row DataFrame matching the training feature schema.
    # Features not supplied by the caller default to 0.0 (e.g. lag2–4 rates,
    # rolling features, momentum). The most important features, vitality lags
    # and sector dummy, are filled from the request.
    row = {col: 0.0 for col in feature_cols}

    # Fill supplied values
    row["vitality_score_lag1"] = req.vitality_score_lag1
    row["vitality_score_lag2"] = req.vitality_score_lag2
    row["vitality_score_lag3"] = req.vitality_score_lag3
    row["vitality_score_lag4"] = req.vitality_score_lag4
    row["bankruptcy_rate_lag1"] = req.bankruptcy_rate_lag1
    row["birth_rate_lag1"] = req.birth_rate_lag1
    row["quarter_of_year"] = req.quarter_of_year
    row["year"] = req.year

    # training columns sector_1 … sector_10. Previously used a letter code
    # ("sector_F") that was never found in row, silently ignoring sector.
    sector_col = f"sector_{req.sector}"
    if sector_col in row:
        row[sector_col] = 1.0

    df_input = pd.DataFrame([row])
    from src.model import predict as model_predict
    prediction = float(model_predict(artifact, df_input)[0])
    prediction = float(np.clip(prediction, 0, 1))

    # Human-readable interpretation
    if prediction >= 0.65:
        interpretation = "Strong growth — sector is expanding with more births than deaths."
    elif prediction >= 0.45:
        interpretation = "Stable — sector is in equilibrium."
    elif prediction >= 0.25:
        interpretation = "Mild stress — sector facing slight contraction."
    else:
        interpretation = "High stress — sector under significant pressure."

    return PredictResponse(
        sector=req.sector,
        predicted_vitality_score=round(prediction, 4),
        interpretation=interpretation,
        model_name=artifact["model_name"],
    )


@app.get("/monitoring")
def get_monitoring():
    report_path = "./artifacts/reports/monitoring_latest.json"
    if not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="No monitoring report available yet.")
    with open(report_path) as f:
        return json.load(f)


@app.post("/pipeline/run")
async def run_pipeline(background_tasks: BackgroundTasks):
    """
    Trigger a full pipeline run in the background:
    ingest → preprocess → feature engineering → train → monitor.
    In production this would be triggered by a GitHub Actions cron job.
    """
    def _run():
        logger.info("Pipeline triggered via API...")
        try:
            subprocess.run(
                ["python", "run_pipeline.py"],
                check=True,
                capture_output=True,
            )
            logger.info("Pipeline completed successfully.")
            # Reset the cached model so next /predict loads the fresh one
            global _model_artifact
            _model_artifact = None
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed: {e.stderr.decode()}")

    background_tasks.add_task(_run)
    return {"status": "Pipeline triggered in background. Check /health for status."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)