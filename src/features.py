"""
features.py
-----------
Transform the quarterly panel into a supervised ML dataset.

Strategy: convert each sector's time series into a tabular format
where each row is (sector, quarter) and features are lagged values
and rolling statistics of the vitality score and its components.

Target
------
  vitality_score at time t+1  (one-quarter-ahead forecast)
  → column: "target"
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Core feature builder

def build_features(
    panel: pd.DataFrame,
    lag_periods: list[int] = [1, 2, 3, 4],
    rolling_windows: list[int] = [4, 8],
    target_col: str = "vitality_score",
) -> pd.DataFrame:
    """
    For each (sector, quarter), generate:
        - Lag features of vitality_score
        - Rolling mean / std of vitality_score
        - Lag features of bankruptcy rate and new enterprise rate
        - Sector dummy (one-hot)
        - Quarter-of-year (seasonality signal)
        - Target: vitality_score at t+1

    Parameters
    ----------
    panel : pd.DataFrame
        Output of preprocessing.compute_vitality_score()
    lag_periods : list[int]
        Quarters to lag, e.g. [1, 2, 3, 4]
    rolling_windows : list[int]
        Rolling window sizes in quarters for mean/std features
    target_col : str
        Column to use as target (default: "vitality_score")

    Returns
    -------
    pd.DataFrame  – ML-ready, NaN rows from lagging are dropped
    """
    df = panel.copy().sort_values(["sector", "period"])

    # Sector-level normalised rates
    df["bankruptcy_rate"] = df["bankruptcies"] / df["active_estimate"].clip(lower=1)
    df["birth_rate"] = df["new_enterprises"] / df["active_estimate"].clip(lower=1)
    df["net_growth_rate"] = df["birth_rate"] - df["bankruptcy_rate"]

    feature_cols = []

    for sector, grp in df.groupby("sector"):
        grp = grp.sort_values("period").copy()

        # Lag features
        for lag in lag_periods:
            col = f"{target_col}_lag{lag}"
            grp[col] = grp[target_col].shift(lag)
            feature_cols.append(col)

            for rate_col in ["bankruptcy_rate", "birth_rate", "net_growth_rate"]:
                c = f"{rate_col}_lag{lag}"
                grp[c] = grp[rate_col].shift(lag)
                feature_cols.append(c)

        # Rolling statistics
        for window in rolling_windows:
            col_mean = f"{target_col}_roll{window}_mean"
            col_std = f"{target_col}_roll{window}_std"
            grp[col_mean] = grp[target_col].shift(1).rolling(window).mean()
            grp[col_std] = grp[target_col].shift(1).rolling(window).std()
            feature_cols.append(col_mean)
            feature_cols.append(col_std)

        # Momentum: change vs previous quarter
        grp[f"{target_col}_momentum"] = grp[target_col].diff().shift(1)
        feature_cols.append(f"{target_col}_momentum")

        # Target: t+1 vitality score
        grp["target"] = grp[target_col].shift(-1)

        df.loc[df["sector"] == sector, grp.columns] = grp.values

    # Time features (seasonality)
    df["quarter_of_year"] = df["period"].dt.quarter
    df["year"] = df["period"].dt.year
    feature_cols += ["quarter_of_year", "year"]

    # Sector encoding
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", drop_first=False)
    df = pd.concat([df, sector_dummies], axis=1)
    feature_cols += list(sector_dummies.columns)

    # Drop rows without target or with NaN in critical lags
    before = len(df)
    df = df.dropna(subset=["target", f"{target_col}_lag1"])
    after = len(df)
    logger.info(
        f"Features built: {after:,} rows ({before - after} dropped for NaN lags/target)"
    )

    # Deduplicate feature list
    feature_cols = list(dict.fromkeys(feature_cols))  # preserve order, remove dups

    # Store feature list as attribute for easy retrieval
    df.attrs["feature_cols"] = feature_cols

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns stored on the DataFrame."""
    return df.attrs.get("feature_cols", [c for c in df.columns if c not in
                                         ["sector", "period", "target",
                                          "vitality_score", "vitality_raw",
                                          "new_enterprises", "bankruptcies",
                                          "active_estimate", "employment",
                                          "bankruptcy_rate", "birth_rate",
                                          "net_growth_rate",
                                          "mængde4", "indhold", "region",
                                          "virktyp1", "branche"]])


# Train / test split, time-based for time series

def time_split(
    df: pd.DataFrame,
    test_quarters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature DataFrame into train and test sets.

    It is split by time, NOT randomly. Random splitting would
    cause data leakage, future data leaking into the training set.
    The most recent `test_quarters` periods form the test set.

    Parameters
    ----------
    test_quarters : int
        Number of most-recent quarters to hold out as test set.
        Default is 4 (one calendar year), matching the project configuration.

    Returns
    -------
    (df_train, df_test)
    """
    periods = sorted(df["period"].unique())
    cutoff = periods[-test_quarters]

    df_train = df[df["period"] < cutoff].copy()
    df_test = df[df["period"] >= cutoff].copy()

    logger.info(
        f"Time split: train={len(df_train):,} rows "
        f"({periods[0].strftime('%Y-%m')} → {df_train['period'].max().strftime('%Y-%m')}), "
        f"test={len(df_test):,} rows "
        f"({df_test['period'].min().strftime('%Y-%m')} → {periods[-1].strftime('%Y-%m')})"
    )
    return df_train, df_test


def save_features(df: pd.DataFrame, feature_dir: str = "./artifacts/processed") -> str:
    """Save the feature DataFrame as a CSV artifact."""
    from datetime import datetime
    Path(feature_dir).mkdir(parents=True, exist_ok=True)
    path = f"{feature_dir}/features_latest.csv"
    df.to_csv(path, index=False)
    logger.info(f"Feature dataset saved → {path}")
    return path