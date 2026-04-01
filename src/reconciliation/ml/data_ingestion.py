"""
data_ingestion.py
-----------------
Load and normalise raw Shopee and Bank CSVs into clean DataFrames.
"""

import os
import pandas as pd


def load_shopee_batches(path: str) -> pd.DataFrame:
    """Load Shopee settlement batches and parse dates."""
    df = pd.read_csv(path)
    for col in ["batch_start_date", "batch_end_date", "settlement_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def load_bank_transactions(path: str) -> pd.DataFrame:
    """Load bank transactions, parse dates, normalise description text."""
    df = pd.read_csv(path)
    df["credit_date"] = pd.to_datetime(df["credit_date"])

    # --- text normalisation ---
    df["description_clean"] = (
        df["description"]
        .str.upper()
        .str.replace(r"[^A-Z0-9 ]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def load_match_mapping(path: str) -> pd.DataFrame:
    """Load ground-truth mapping table."""
    return pd.read_csv(path)


def load_all(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: load all three datasets from *data_dir*."""
    batches = load_shopee_batches(os.path.join(data_dir, "Shopee_batches.csv"))
    bank = load_bank_transactions(os.path.join(data_dir, "Bank_transactions.csv"))
    mapping = load_match_mapping(os.path.join(data_dir, "Match_mapping.csv"))
    return batches, bank, mapping
