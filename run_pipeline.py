"""
run_pipeline.py
---------------
End-to-end pipeline runner. Called by:
  - GitHub Actions (scheduled cron)
  - FastAPI /pipeline/run endpoint
  - Manual execution: python run_pipeline.py
"""

import sys
import logging
import json
import yaml
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=" * 60)
    logger.info("DST Sector Health Pipeline — Starting")
    logger.info("=" * 60)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1: Data Ingestion
    logger.info("\n[1/5] Data Ingestion")
    from src.dst_client import DSTClient, fetch_bankruptcies, fetch_new_enterprises, save_raw
    client = DSTClient()
    df_bankrupt = fetch_bankruptcies(client)
    df_new = fetch_new_enterprises(client)
    save_raw(df_bankrupt, "bankruptcies")
    save_raw(df_new, "new_enterprises")

    # Step 2: Preprocessing
    logger.info("\n[2/5] Preprocessing")
    from src.preprocessing import (
        bankruptcies_to_quarterly, build_panel,
        compute_vitality_score, quality_report, save_processed
    )
    df_bankrupt_q = bankruptcies_to_quarterly(df_bankrupt)
    panel = build_panel(df_new, df_bankrupt_q)

    # Filter to the period where both data sources have complete coverage.
    # KONK4 (bankruptcies) runs from 2009; DEMO14 (enterprise births) only
    # from 2019, with a ~2-year publication lag meaning 2024+ has no births
    # yet. Rows outside 2019–2023 have new_enterprises=0, which distorts the
    # per-sector min-max normalisation of the vitality score.
    panel = panel[
        (panel["period"] >= "2019-01-01") &
        (panel["period"] < "2024-01-01")
    ].reset_index(drop=True)

    panel = compute_vitality_score(panel)
    qr = quality_report(panel)
    logger.info(f"Quality report: {json.dumps(qr, indent=2)}")
    save_processed(panel)

    # Step 3: Feature Engineering
    logger.info("\n[3/5] Feature Engineering")
    from src.features import build_features, get_feature_columns, time_split, save_features
    cfg = config["model"]
    df_features = build_features(
        panel,
        lag_periods=cfg["lag_periods"],
        rolling_windows=cfg["rolling_windows"],
        target_col=cfg["target"],
    )
    feature_cols = get_feature_columns(df_features)
    df_train, df_test = time_split(df_features, test_quarters=cfg["test_quarters"])
    save_features(df_features)

    # Step 4: Model Training
    logger.info("\n[4/5] Model Training")
    from src.model import (
        naive_baseline, train_ridge, train_xgboost,
        log_to_mlflow, save_model
    )

    # Baseline
    baseline = naive_baseline(df_test, feature_cols)

    # Ridge
    ridge_result = train_ridge(df_train, df_test, feature_cols)
    log_to_mlflow(ridge_result, feature_cols,
                  tracking_uri=config["mlflow"]["tracking_uri"],
                  experiment_name=config["mlflow"]["experiment_name"])
    save_model(ridge_result, feature_cols, config["paths"]["models"])

    # XGBoost (primary)
    xgb_result = train_xgboost(df_train, df_test, feature_cols)
    log_to_mlflow(xgb_result, feature_cols,
                  tracking_uri=config["mlflow"]["tracking_uri"],
                  experiment_name=config["mlflow"]["experiment_name"])
    save_model(xgb_result, feature_cols, config["paths"]["models"])

    # Step 5: Monitoring
    logger.info("\n[5/5] Monitoring")
    from src.monitoring import (
        detect_data_drift,
        monitor_performance, monitor_prediction_distribution,
        save_monitoring_report
    )

    data_drift = detect_data_drift(
        df_train, df_test, feature_cols,
        threshold=config["monitoring"]["drift_threshold"]
    )

    # Use XGBoost's own test MAE as the performance baseline so that
    # monitor_performance detects degradation relative to what the model
    # achieved at training time. The Naive MAE is a different metric and
    # is reported separately in model_metrics for full context.
    baseline_mae = xgb_result["metrics"]["mae"]
    y_test = df_test["target"].values
    y_pred_train = xgb_result["estimator"].predict(df_train[feature_cols].fillna(0).values)
    y_pred_test = xgb_result["predictions"]

    perf_monitor = monitor_performance(
        y_test, y_pred_test, baseline_mae,
        threshold_pct=config["monitoring"]["performance_threshold"]
    )
    pred_dist = monitor_prediction_distribution(y_pred_train, y_pred_test)

    full_report = {
        "data_drift": data_drift,
        "performance": perf_monitor,
        "prediction_distribution": pred_dist,
        "model_metrics": {
            "naive": baseline["metrics"],
            "ridge": ridge_result["metrics"],
            "xgboost": xgb_result["metrics"],
        }
    }
    save_monitoring_report(full_report)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully ✅")
    logger.info(f"  Naive MAE:   {baseline['metrics']['mae']:.4f}")
    logger.info(f"  Ridge MAE:   {ridge_result['metrics']['mae']:.4f}")
    logger.info(f"  XGBoost MAE: {xgb_result['metrics']['mae']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
