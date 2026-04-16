"""
preprocessing.py
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# Align monthly bankruptcies → quarterly

def bankruptcies_to_quarterly(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    DST KONK4 is published monthly. Aggregated to quarters so it
    aligns with the new-enterprise registration data (annual, forward-filled to quarterly).
    Keeps only active companies (VIRKTYP1 == K02) which is already filtered in dst_client.
    """
    df_monthly = df_monthly.copy()
    df_monthly["quarter"] = df_monthly["period"].dt.to_period("Q").dt.to_timestamp()

    agg = (
        df_monthly.groupby(["branche10", "quarter"])["bankruptcies"]
        .sum()
        .reset_index()
        .rename(columns={"branche10": "sector", "quarter": "period"})
    )
    return agg


# Align annual employment data → quarterly (forward-fill)

def employment_to_quarterly(df_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Employment is annual in DST. Forward-filled to quarterly so it
    can join with the quarterly panel.
    """
    df = df_annual.copy()
    df["period"] = df["period"].dt.to_period("Q").dt.to_timestamp()

    records = []
    for _, row in df.iterrows():
        year = row["period"].year
        for q in range(1, 5):
            records.append({
                "sector": row.get("branche", row.get("sector", "Unknown")),
                # pd.Period is the correct Pandas 2.x way to build a quarter
                # timestamp. pd.Timestamp(..., freq="Q") was removed in 2.x.
                "period": pd.Period(f"{year}Q{q}").to_timestamp(),
                "employment": row["employment"],
            })
    return pd.DataFrame(records)


# Build the quarterly panel

def build_panel(
    df_new: pd.DataFrame,
    df_bankrupt: pd.DataFrame,
    df_employment: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge new enterprises and bankruptcies on (sector, period) to build
    a balanced quarterly panel.

    DEMO14 is annual data (one row per sector per year with period = Jan 1).
    After outer-joining with quarterly bankruptcy data, only Q1 of each year
    has new_enterprises > 0. forward-fill within each (sector, year) so
    all 4 quarters carry the annual enterprise birth count.

    Returns a DataFrame with columns:
        sector, period, new_enterprises, bankruptcies, [employment]
    """
    df_new = df_new.copy()
    df_bankrupt = df_bankrupt.copy()
    df_new["branche"] = df_new["branche"].astype(str)
    df_bankrupt["sector"] = df_bankrupt["sector"].astype(str)

    panel = pd.merge(
        df_new.rename(columns={"branche": "sector"}),
        df_bankrupt,
        on=["sector", "period"],
        how="outer",
    ).fillna(0)

    panel = panel.sort_values(["sector", "period"]).reset_index(drop=True)

    # Forward-fill annual enterprise births across all 4 quarters of each year.
    # After the outer merge, only Q1 (January) has new_enterprises > 0.
    # The other 3 quarters get 0 from fillna, which is incorrect, the annual count should apply to all quarters of that year.
    panel["year"] = panel["period"].dt.year
    panel["new_enterprises"] = panel.groupby(["sector", "year"])["new_enterprises"].transform(
        lambda x: x.where(x > 0).ffill().bfill().fillna(0)
    )
    panel = panel.drop(columns=["year"])

    if df_employment is not None:
        panel = panel.merge(df_employment, on=["sector", "period"], how="left")
        panel["employment"] = panel["employment"].ffill()

    panel = panel.sort_values(["sector", "period"]).reset_index(drop=True)
    logger.info(
        f"Panel built: {panel['sector'].nunique()} sectors × "
        f"{panel['period'].nunique()} quarters = {len(panel):,} rows"
    )
    return panel


# Compute Sector Vitality Score

def compute_vitality_score(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Sector Vitality Score (SVS) for each sector-quarter.

    Steps:
        1. Estimate active enterprise count via cumulative net growth
        2. SVS = (new_enterprises - bankruptcies) / active_estimate
        3. Normalise SVS to [0, 1] per sector for cross-sector comparability

    The normalised SVS is the TARGET VARIABLE for the forecasting model.
    Raw SVS is kept for interpretability.
    """
    df = panel.copy().sort_values(["sector", "period"])

    # Step 1: Estimate active enterprise count
    records = []
    for sector, grp in df.groupby("sector"):
        grp = grp.sort_values("period").copy()
        # Estimate realistic stock: ~5% annual birth rate → stock ≈ births × 20
        first_births = grp['new_enterprises'].iloc[0]
        active = max(float(first_births) * 20, 100.0)
        actives = []
        for _, row in grp.iterrows():
            actives.append(active)
            net = row["new_enterprises"] - row["bankruptcies"]
            active = max(active + net, 1)
        grp["active_estimate"] = actives
        records.append(grp)

    df = pd.concat(records).reset_index(drop=True)

    # Step 2: Raw SVS
    df["vitality_raw"] = (
        (df["new_enterprises"] - df["bankruptcies"]) / df["active_estimate"]
    )

    # Step 3: Min-max normalise per sector → vitality_score ∈ [0, 1]
    def minmax(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else x * 0 + 0.5

    df["vitality_score"] = df.groupby("sector")["vitality_raw"].transform(minmax)

    logger.info(
        f"Vitality score computed. "
        f"Mean={df['vitality_score'].mean():.3f}, "
        f"Std={df['vitality_score'].std():.3f}"
    )
    return df


# Quality checks

def quality_report(df: pd.DataFrame) -> dict:
    """Return a dict of data quality metrics for logging / artifact storage."""
    report = {
        "n_rows": len(df),
        "n_sectors": df["sector"].nunique(),
        "n_quarters": df["period"].nunique(),
        "missing_pct": df.isnull().mean().to_dict(),
        "period_min": str(df["period"].min()),
        "period_max": str(df["period"].max()),
        "vitality_mean": round(df["vitality_score"].mean(), 4) if "vitality_score" in df.columns else None,
        "vitality_std": round(df["vitality_score"].std(), 4) if "vitality_score" in df.columns else None,
        "zero_bankruptcies_pct": round((df["bankruptcies"] == 0).mean(), 4),
    }
    return report


def save_processed(df: pd.DataFrame, processed_dir: str = "./artifacts/processed") -> str:
    """Save the processed panel as a CSV artifact."""
    from datetime import datetime
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{processed_dir}/panel_{timestamp}.csv"
    df.to_csv(path, index=False)
    df.to_csv(f"{processed_dir}/panel_latest.csv", index=False)
    logger.info(f"Processed panel saved → {path}")
    return path