"""
dst_client.py
-------------
Client for the Statistics Denmark (DST) StatBank API.
Documentation: https://www.dst.dk/en/Statistik/hjaelp-til-statistikbanken/api
"""

import requests
import pandas as pd
import io
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DSTClient:
    """
    Thin wrapper around the DST StatBank REST API.

    Key endpoints:
        /v1/tables    - list all available tables
        /v1/tableinfo - metadata + variable codes for a specific table
        /v1/data      - fetch actual data (CSV or JSON-stat)
    """

    BASE_URL = "https://api.statbank.dk/v1"

    def __init__(self, language: str = "en"):
        self.language = language
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # Discovery helpers

    def search_tables(self, keyword: str) -> pd.DataFrame:
        """Search available tables by keyword. Useful for finding table IDs."""
        resp = self.session.get(
            f"{self.BASE_URL}/tables",
            params={"lang": self.language, "format": "JSON"},
        )
        resp.raise_for_status()
        tables = resp.json()
        df = pd.DataFrame(tables)[["id", "text", "updated", "firstPeriod", "latestPeriod"]]
        mask = df["text"].str.lower().str.contains(keyword.lower(), na=False)
        return df[mask].reset_index(drop=True)

    def get_table_info(self, table_id: str) -> dict:
        """
        Return full metadata for a table: description, variables, and
        all possible values for each variable. Always run this first
        before fetching data, it shows you what filters you can use.
        """
        resp = self.session.post(
            f"{self.BASE_URL}/tableinfo",
            json={"table": table_id, "lang": self.language},
        )
        resp.raise_for_status()
        return resp.json()

    def print_table_variables(self, table_id: str):
        """Pretty-print available variables and their values for a table."""
        info = self.get_table_info(table_id)
        print(f"\n{'='*60}")
        print(f"Table: {table_id} — {info.get('text', '')}")
        print(f"Unit: {info.get('unit', 'N/A')} | Updated: {info.get('updated', 'N/A')}")
        print(f"Period: {info.get('firstPeriod', '')} → {info.get('latestPeriod', '')}")
        print(f"{'='*60}")
        for var in info.get("variables", []):
            print(f"\n  Variable: {var['id']} — {var['text']}")
            vals = var.get("values", [])[:10]  # show first 10
            for v in vals:
                print(f"    {v['id']:>12}  {v['text']}")
            if len(var.get("values", [])) > 10:
                print(f"    ... ({len(var['values'])} total values)")

    # Data fetching
    def fetch(
        self,
        table_id: str,
        variables: dict,
        value_presentation: str = "Code",
    ) -> pd.DataFrame:
        """
        Fetch data from a DST table and return a tidy DataFrame.

        Parameters
        ----------
        table_id : str
            DST table identifier, e.g. "KONK4"
        variables : dict
            Mapping of variable code → list of values to filter on.
            Use ["*"] to get all values for that variable.
            Example: {"BRANCHE": ["*"], "Tid": ["2015M01", "2015M02"]}
        value_presentation : str
            "Code" returns coded values (e.g. "1"), "Value" returns labels.

        Returns
        -------
        pd.DataFrame  (tidy / long format)
        """
        payload = {
            "table": table_id,
            "format": "CSV",
            "lang": self.language,
            "valuePresentation": value_presentation,
            "variables": [
                {"code": code, "values": vals}
                for code, vals in variables.items()
            ],
        }
        logger.info(f"Fetching table {table_id} ...")
        resp = self.session.post(f"{self.BASE_URL}/data", json=payload)
        resp.raise_for_status()

        # DST returns semicolon-separated CSV
        df = pd.read_csv(io.StringIO(resp.text), sep=";", decimal=",")
        logger.info(f"  → {len(df):,} rows fetched")
        return df

    def fetch_all_periods(self, table_id: str, variables: dict) -> pd.DataFrame:
        """Convenience wrapper: always use '*' for the time dimension."""
        info = self.get_table_info(table_id)
        # Find time variable by 'time' flag first, fall back to 'Tid' by name
        time_var = next(
            (v["id"] for v in info["variables"] if v.get("time")),
            next((v["id"] for v in info["variables"] if v["id"] == "Tid"), None)
        )
        if time_var:
            variables[time_var] = ["*"]
        return self.fetch(table_id, variables)


# Table-specific fetchers

def fetch_bankruptcies(client: DSTClient) -> pd.DataFrame:
    """
    KONK4: Monthly bankruptcies by industry (DB07) and enterprise type.
    Variables (confirmed from API):
        BRANCHE  — industry (000=total, 1=Agriculture, 2=Manufacturing, etc.)
        VIRKTYP1 — K01=total, K02=active companies, K03=inactive companies
        Tid      — period in format 2009M01
    Keep K02 (active companies) only, these are economically meaningful.
    Keep only top-level sector codes 1-10 to match DEMO14 industry grouping.
    """
    df = client.fetch_all_periods(
        "KONK4",
        variables={
            "BRANCHE":  ["*"],    # all industry groups
            "VIRKTYP1": ["K02"],  # active companies only
        },
    )
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Rename value column (DST names it after the unit, e.g. "antal")
    value_col = [c for c in df.columns if c not in ("branche", "virktyp1", "tid")][0]
    df = df.rename(columns={value_col: "bankruptcies", "tid": "period", "branche": "branche10"})

    # Parse period — format is "2020M01"
    df["period"] = pd.to_datetime(df["period"], format="%YM%m", errors="coerce")
    df["bankruptcies"] = pd.to_numeric(df["bankruptcies"], errors="coerce").fillna(0)

    # Drop the total row (BRANCHE code "000")
    df = df[df["branche10"] != "000"].reset_index(drop=True)

    # Keep only top-level sector codes (1-10) to match DEMO14 industry grouping.
    # KONK4 also contains sub-codes (G01, G02, G03, H, I, 101, 102, 11)
    # which have no equivalent in DEMO14 and would cause a panel mismatch.
    top_level = [str(i) for i in range(1, 11)]
    df = df[df["branche10"].isin(top_level)].reset_index(drop=True)

    return df


def fetch_new_enterprises(client: DSTClient) -> pd.DataFrame:
    """
    DEMO14: Business Demography — annual enterprise births by industry.
    Variables (confirmed from API):
        REGION        — 000=All Denmark
        BRANCHEDB0710 — industry (DB07 10-grouping), TOT=total, 1-10=sectors
        MÆNGDE4       — NYE=new enterprises, AFU=closures, OPH=discontinued, OMS=turnover
        Tid           — year (2019, 2020, 2021, 2022, 2023)
    Note: annual data — forward-filled to quarterly frequency in preprocessing.
    """
    df = client.fetch_all_periods(
        "DEMO14",
        variables={
            "REGION":        ["000"],  # All Denmark
            "BRANCHEDB0710": ["*"],    # all industry groups
            "MÆNGDE4":       ["NYE"],  # NYE = new enterprises only
        },
    )
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    value_col = [c for c in df.columns
                 if c not in ("region", "branchedb0710", "mængde4", "tid")][0]
    df = df.rename(columns={
        value_col: "new_enterprises",
        "tid": "period",
        "branchedb0710": "branche",
    })
    # Keep only top-level sector codes 1-10.
    # DEMO14 also includes sector 11 ("Activity not stated") which has no
    # equivalent in KONK4 and would create unmatched rows in the panel merge.
    top_level = [str(i) for i in range(1, 11)]
    df = df[df["branche"].isin(top_level)].reset_index(drop=True)

    # Parse annual period → January 1st of that year
    df["period"] = pd.to_datetime(df["period"].astype(str), format="%Y")
    df["new_enterprises"] = pd.to_numeric(
        df["new_enterprises"], errors="coerce"
    ).fillna(0)

    # Drop helper columns not needed downstream
    df = df.drop(columns=["region", "mængde4"], errors="ignore")

    return df


def fetch_employment(client: DSTClient) -> pd.DataFrame:
    """
    RAS200: Register-based employment by industry (annual).
    Used as a supplementary macro feature.
    """
    df = client.fetch_all_periods(
        "RAS200",
        variables={
            "BRANCHE": ["*"],
            "STATSEKT": ["TOT"],   # all sectors combined
        },
    )
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    value_col = [c for c in df.columns if c not in ("branche", "statsekt", "tid")][0]
    df = df.rename(columns={value_col: "employment", "tid": "period"})
    df["period"] = pd.to_datetime(df["period"], format="%Y")
    df["employment"] = pd.to_numeric(df["employment"], errors="coerce")
    return df


def save_raw(df: pd.DataFrame, name: str, raw_dir: str = "./artifacts/raw") -> str:
    """Save a raw DataFrame as a timestamped CSV artifact."""
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{raw_dir}/{name}_{timestamp}.csv"
    df.to_csv(path, index=False)
    # Also save as 'latest' for easy pipeline access
    latest_path = f"{raw_dir}/{name}_latest.csv"
    df.to_csv(latest_path, index=False)
    logger.info(f"Saved → {path}")
    return path