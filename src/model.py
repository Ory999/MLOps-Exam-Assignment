"""
model.py
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import logging
import json
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Metrics helper

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sectors: np.ndarray | None = None,
) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: was the direction of change correct? (up/down/no change)
    # The dataset is a panel (multiple sectors × time). np.diff across all
    # rows would cross sector boundaries, comparing the last quarter of one
    # sector with the first quarter of the next, which is meaningless.
    # When sector labels are supplied, compute per sector and average.
    if sectors is not None:
        per_sector = []
        for s in np.unique(sectors):
            mask = sectors == s
            yt, yp = y_true[mask], y_pred[mask]
            if len(yt) > 1:
                per_sector.append(
                    np.mean(np.sign(np.diff(yt)) == np.sign(np.diff(yp)))
                )
        dir_acc = float(np.mean(per_sector)) if per_sector else float("nan")
    elif len(y_true) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))))
    else:
        dir_acc = float("nan")

    return {
        "mae": round(float(mae), 5),
        "rmse": round(float(rmse), 5),
        "r2": round(float(r2), 5),
        "directional_accuracy": round(float(dir_acc), 5),
    }


# Baseline: Naive persistence

def naive_baseline(df_test: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Naive forecast: predict next quarter's vitality = this quarter's vitality.
    This is the floor, if the model doesn't beat it, then there is a problem.
    The lag-1 feature IS the naive prediction.
    """
    lag1_col = next(c for c in feature_cols if c.endswith("_lag1") and "vitality" in c)
    y_true = df_test["target"].values
    y_pred = df_test[lag1_col].values
    metrics = regression_metrics(y_true, y_pred, sectors=df_test["sector"].values)
    logger.info(f"Naive baseline test metrics: {metrics}")
    return {"model": "naive_persistence", "metrics": metrics, "predictions": y_pred}


# Ridge Regression (linear baseline)

def train_ridge(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """
    Ridge regression as a linear baseline. Simpler than XGBoost
    and useful to check if linear relationships explain most of
    the variance.
    """
    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train["target"].values
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test["target"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = regression_metrics(y_test, y_pred, sectors=df_test["sector"].values)
    logger.info(f"Ridge test metrics: {metrics}")
    return {
        "model": "ridge",
        "estimator": model,
        "scaler": scaler,
        "metrics": metrics,
        "predictions": y_pred,
    }


# XGBoost

def train_xgboost(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> dict:
    """
    Train XGBoost regressor.

    Default hyperparameters are conservative and work well for
    small quarterly datasets. No grid search here to keep the
    pipeline fast, the point is operational correctness, not
    optimal performance.

    Key parameters explained:
        n_estimators=300     : number of trees
        max_depth=4          : shallow trees → avoids overfitting on small data
        learning_rate=0.05   : slow learning → more stable on limited data
        subsample=0.8        : row sampling per tree → regularisation
        colsample_bytree=0.8 : feature sampling per tree → regularisation
        early_stopping_rounds: stop if validation loss doesn't improve
    """
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,    # L1 regularisation
            "reg_lambda": 1.0,   # L2 regularisation
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
            "eval_metric": "mae",
        }

    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train["target"].values
    X_test = df_test[feature_cols].fillna(0).values
    y_test = df_test["target"].values

    # Use a small validation slice from the END of train (time-ordered)
    val_size = max(1, int(len(X_train) * 0.1))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    model = XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, sectors=df_test["sector"].values)
    logger.info(f"XGBoost test metrics: {metrics}")
    logger.info(f"  Best iteration: {model.best_iteration}")

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    top10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"  Top features: {top10}")

    return {
        "model": "xgboost",
        "estimator": model,
        "metrics": metrics,
        "predictions": y_pred,
        "feature_importances": importances,
        "params": params,
    }


# MLflow logging

def log_to_mlflow(
    result: dict,
    feature_cols: list[str],
    run_name: str | None = None,
    tracking_uri: str = "./artifacts/mlflow",
    experiment_name: str = "dst_sector_health",
):
    """
    Log a model run to MLflow — parameters, metrics, and the model artifact.
    MLflow creates a local SQLite + file store under ./artifacts/mlflow.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = run_name or f"{result['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        if "params" in result:
            mlflow.log_params(result["params"])
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("model_type", result["model"])

        # Log metrics
        for metric_name, value in result["metrics"].items():
            if not np.isnan(value):
                mlflow.log_metric(metric_name, value)

        # Log model
        if "estimator" in result:
            mlflow.sklearn.log_model(
                result["estimator"],
                artifact_path="model",
                registered_model_name=f"sector_health_{result['model']}",
            )

        # Log feature importances as JSON
        if "feature_importances" in result:
            mlflow.log_dict({k: float(v) for k, v in result["feature_importances"].items()}, "feature_importances.json")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run logged: {run_id}")
        return run_id


# Save / load model artifact

def save_model(result: dict, feature_cols: list[str], model_dir: str = "./artifacts/models") -> str:
    """Save model, scaler (if present), feature list, and metrics as artifacts."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = result["model"]

    artifact = {
        "estimator": result["estimator"],
        "feature_cols": feature_cols,
        "metrics": result["metrics"],
        "model_name": model_name,
        "trained_at": timestamp,
    }
    if "scaler" in result:
        artifact["scaler"] = result["scaler"]

    path = f"{model_dir}/{model_name}_{timestamp}.joblib"
    joblib.dump(artifact, path)
    # Always update 'latest'
    joblib.dump(artifact, f"{model_dir}/model_latest.joblib")

    # Save metrics as human-readable JSON
    metrics_path = f"./artifacts/metrics/{model_name}_{timestamp}_metrics.json"
    Path("./artifacts/metrics").mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"metrics": result["metrics"], "params": result.get("params", {})}, f, indent=2)

    logger.info(f"Model saved → {path}")
    return path


def load_model(model_path: str = "./artifacts/models/model_latest.joblib") -> dict:
    """Load a saved model artifact."""
    return joblib.load(model_path)


def predict(model_artifact: dict, df: pd.DataFrame) -> np.ndarray:
    """Run inference using a loaded model artifact."""
    feature_cols = model_artifact["feature_cols"]
    X = df[feature_cols].fillna(0).values
    estimator = model_artifact["estimator"]
    if "scaler" in model_artifact:
        X = model_artifact["scaler"].transform(X)
    return estimator.predict(X)
