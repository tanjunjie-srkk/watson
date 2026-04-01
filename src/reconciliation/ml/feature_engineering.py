"""
feature_engineering.py
---------------------
Compute numeric, date, and text features for every candidate pair.
All features are designed to be explainable and useful for tree-based
models (XGBoost / LightGBM) as well as logistic regression baselines.
"""

import re

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


# ---------------------------------------------------------------------------
# Shopee-related keywords for keyword-flag features
# ---------------------------------------------------------------------------
SHOPEE_KEYWORDS = {"SHOPEE", "PAYOUT", "SETTLEMENT", "SP", "DIGITAL"}


# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set[str]:
    return set(re.sub(r"[^A-Z0-9 ]", " ", text.upper()).split())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of candidate pairs (with columns produced by
    candidate_generation.py), compute ML features and return a copy
    with the new columns appended.

    Input columns expected
    ----------------------
    settlement_date, credit_date, batch_net, bank_amount,
    batch_gross, batch_refund, batch_order_count,
    bank_description, bank_description_clean

    Output columns added
    --------------------
    amount_diff, amount_abs_diff, amount_pct_diff, amount_log_ratio,
    fee_adjusted_diff, date_diff_days, date_diff_abs,
    is_weekend_credit, desc_jaccard, desc_seq_ratio,
    keyword_shopee_flag, batch_size_log, refund_ratio,
    group_size  (if present)
    """
    df = candidates.copy()

    # ---- Amount features ----
    df["amount_diff"] = df["batch_net"] - df["bank_amount"]
    df["amount_abs_diff"] = df["amount_diff"].abs()
    df["amount_pct_diff"] = np.where(
        df["batch_net"] != 0,
        df["amount_abs_diff"] / df["batch_net"].abs(),
        0.0,
    )
    df["amount_log_ratio"] = np.log1p(df["batch_net"].abs()) - np.log1p(df["bank_amount"].abs())

    # Fee-adjusted: compare gross minus known deductions to bank amount
    df["fee_adjusted_diff"] = (df["batch_gross"] - df["batch_refund"] - df["bank_amount"]).abs()

    # ---- Date features ----
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    df["credit_date"] = pd.to_datetime(df["credit_date"])
    df["date_diff_days"] = (df["settlement_date"] - df["credit_date"]).dt.days
    df["date_diff_abs"] = df["date_diff_days"].abs()
    df["is_weekend_credit"] = df["credit_date"].dt.dayofweek.isin([5, 6]).astype(int)

    # ---- Text features ----
    desc = df["bank_description_clean"].fillna("")
    batch_label = "SHOPEE SETTLEMENT"  # canonical label for batch side

    df["desc_jaccard"] = desc.apply(lambda d: _jaccard(_token_set(batch_label), _token_set(d)))
    df["desc_seq_ratio"] = desc.apply(lambda d: _sequence_ratio(batch_label, d))
    df["keyword_shopee_flag"] = desc.apply(
        lambda d: int(bool(SHOPEE_KEYWORDS & _token_set(d)))
    )

    # ---- Batch-level features ----
    df["batch_size_log"] = np.log1p(df["batch_order_count"])
    df["refund_ratio"] = np.where(
        df["batch_gross"] != 0,
        df["batch_refund"] / df["batch_gross"],
        0.0,
    )

    # ---- Group size (many-to-one) ----
    if "group_size" not in df.columns:
        df["group_size"] = 1

    return df


# ---------------------------------------------------------------------------
# Feature column list (for model input)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "amount_abs_diff",
    "amount_pct_diff",
    "amount_log_ratio",
    "fee_adjusted_diff",
    "date_diff_days",
    "date_diff_abs",
    "is_weekend_credit",
    "desc_jaccard",
    "desc_seq_ratio",
    "keyword_shopee_flag",
    "batch_size_log",
    "refund_ratio",
    "group_size",
]
