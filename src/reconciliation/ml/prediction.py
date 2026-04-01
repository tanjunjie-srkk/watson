"""
prediction.py
-------------
Load a trained model and score new candidate pairs.
Applies threshold logic:
    >= 0.90  → auto_match
    0.60–0.90 → review
    <  0.60  → unmatched
"""

import pickle
from typing import Optional

import numpy as np
import pandas as pd

from feature_engineering import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Thresholds (configurable)
# ---------------------------------------------------------------------------
THRESHOLD_AUTO = 0.90
THRESHOLD_REVIEW = 0.60


def load_model(model_path: str) -> dict:
    """Load a persisted model artefact (model + scaler + feature list)."""
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    return artefact


def predict(
    df: pd.DataFrame,
    artefact: dict,
    threshold_auto: float = THRESHOLD_AUTO,
    threshold_review: float = THRESHOLD_REVIEW,
) -> pd.DataFrame:
    """
    Score candidate pairs and assign reconciliation status.

    Parameters
    ----------
    df : DataFrame with FEATURE_COLUMNS already computed
    artefact : dict returned by load_model()
    threshold_auto : probability above which pair is auto-matched
    threshold_review : probability above which pair is flagged for review

    Returns
    -------
    Copy of df with extra columns: confidence, match_status
    """
    model = artefact["model"]
    scaler = artefact.get("scaler")
    features = artefact.get("features", FEATURE_COLUMNS)

    X = df[features].copy().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Use scaler only for linear models (LogisticRegression)
    model_type = type(model).__name__
    if model_type == "LogisticRegression" and scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["confidence"] = proba
    out["match_status"] = np.select(
        [proba >= threshold_auto, proba >= threshold_review],
        ["auto_match", "review"],
        default="unmatched",
    )
    return out


# ---------------------------------------------------------------------------
# Conflict resolution: pick best match per bank_trx_id
# ---------------------------------------------------------------------------

def resolve_conflicts(scored: pd.DataFrame) -> pd.DataFrame:
    """
    When multiple batches score as 'auto_match' for the same bank_trx_id,
    keep only the highest-confidence match; demote duplicates to 'review'.
    """
    df = scored.copy()
    df = df.sort_values("confidence", ascending=False)

    seen_bank = set()
    new_status = []
    for _, row in df.iterrows():
        bank_id = row["bank_trx_id"]
        status = row["match_status"]
        if status == "auto_match":
            if bank_id in seen_bank:
                new_status.append("review")
            else:
                seen_bank.add(bank_id)
                new_status.append("auto_match")
        else:
            new_status.append(status)

    df["match_status"] = new_status
    return df.sort_index()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def reconciliation_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Return a human-readable summary grouped by match_status."""
    summary = (
        scored.groupby("match_status")
        .agg(
            count=("confidence", "size"),
            avg_confidence=("confidence", "mean"),
            total_bank_amount=("bank_amount", "sum"),
        )
        .reset_index()
    )
    return summary
