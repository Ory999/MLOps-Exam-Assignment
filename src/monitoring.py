"""
monitoring.py
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)

# Data drift detection

def compute_feature_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute summary statistics for each feature column."""
    stats_dict = {}
    for col in feature_cols:
        if col in df.columns:
            s = df[col].dropna()
            stats_dict[col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "n": int(len(s)),
            }
    return stats_dict


def detect_data_drift(
    df_reference: pd.DataFrame,   # training data distribution
    df_current: pd.DataFrame,     # new incoming data
    feature_cols: list[str],
    threshold: float = 0.15,
) -> dict:
    """
    Run KS-test on each numeric feature between reference and current data.

    A KS p-value < threshold suggests the distributions are significantly
    different → the model may be operating outside its training domain.

    Parameters
    ----------
    df_reference : reference distribution (training data)
    df_current   : new incoming data batch
    feature_cols : features to test
    threshold    : KS p-value threshold (default 0.15)

    Returns
    -------
    dict with per-feature drift results and an overall drift flag
    """
    results = {}
    drifted_features = []

    for col in feature_cols:
        if col not in df_reference.columns or col not in df_current.columns:
            continue
        ref_vals = df_reference[col].dropna().values
        cur_vals = df_current[col].dropna().values

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue  # not enough data for a meaningful test

        ks_stat, p_value = stats.ks_2samp(ref_vals, cur_vals)
        is_drifted = p_value < threshold

        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": is_drifted,
        }
        if is_drifted:
            drifted_features.append(col)

    overall_drift = len(drifted_features) / max(len(results), 1) > 0.3

    report = {
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "n_features_tested": len(results),
        "n_features_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "overall_drift_flag": overall_drift,
        "per_feature": results,
    }

    if overall_drift:
        logger.warning(
            f"⚠️  DATA DRIFT DETECTED: {len(drifted_features)}/{len(results)} features drifted. "
            f"Consider retraining the model."
        )
    else:
        logger.info(
            f"✅ No significant data drift detected "
            f"({len(drifted_features)}/{len(results)} features flagged)."
        )

    return report


# Model performance monitoring

def monitor_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_mae: float,
    threshold_pct: float = 0.20,
) -> dict:
    """
    Compare recent prediction error against training baseline.

    If recent MAE > baseline_mae * (1 + threshold_pct) → flag for retraining.

    Parameters
    ----------
    y_true        : true vitality scores for the new period
    y_pred        : model predictions for the new period
    baseline_mae  : MAE achieved during training evaluation
    threshold_pct : tolerated degradation fraction (default: 20%)

    Returns
    -------
    dict with current metrics and a retrain flag
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    current_mae = mean_absolute_error(y_true, y_pred)
    current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    degradation_pct = (current_mae - baseline_mae) / max(baseline_mae, 1e-9)
    needs_retrain = current_mae > baseline_mae * (1 + threshold_pct)

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_mae": round(float(baseline_mae), 5),
        "current_mae": round(float(current_mae), 5),
        "current_rmse": round(float(current_rmse), 5),
        "degradation_pct": round(float(degradation_pct * 100), 2),
        "retrain_flag": needs_retrain,
    }

    if needs_retrain:
        logger.warning(
            f"⚠️  MODEL PERFORMANCE DEGRADED: MAE={current_mae:.4f} "
            f"(baseline={baseline_mae:.4f}, +{degradation_pct*100:.1f}%). "
            f"Retraining recommended."
        )
    else:
        logger.info(
            f"✅ Model performance OK: MAE={current_mae:.4f} "
            f"(baseline={baseline_mae:.4f}, {degradation_pct*100:+.1f}%)"
        )

    return report

# Prediction distribution monitoring

def monitor_prediction_distribution(
    y_pred_train: np.ndarray,
    y_pred_new: np.ndarray,
    threshold: float = 0.10,
) -> dict:
    """
    Check if model predictions have shifted in distribution.
    Extreme prediction drift (e.g. all predictions near 0 or 1)
    can indicate a problem with incoming data rather than model drift.
    """
    ks_stat, p_value = stats.ks_2samp(y_pred_train, y_pred_new)
    is_drifted = p_value < threshold

    report = {
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_value), 4),
        "prediction_drift": is_drifted,
        "new_pred_mean": round(float(y_pred_new.mean()), 4),
        "new_pred_std": round(float(y_pred_new.std()), 4),
        "train_pred_mean": round(float(y_pred_train.mean()), 4),
        "train_pred_std": round(float(y_pred_train.std()), 4),
    }
    return report

# Save monitoring report

def save_monitoring_report(report: dict, report_dir: str = "./artifacts/reports") -> str:
    """Save a monitoring report as a timestamped JSON file."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{report_dir}/monitoring_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    # Keep a 'latest' copy for the API / dashboard to read
    with open(f"{report_dir}/monitoring_latest.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Monitoring report saved → {path}")
    return path
