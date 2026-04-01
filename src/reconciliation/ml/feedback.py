"""
feedback.py
-----------
Store human feedback (accept / reject / correct) and support periodic
retraining of the reconciliation model.

Feedback is stored as a JSONL file so it can be appended to cheaply
and loaded into Pandas for retraining.
"""

import json
import os
from datetime import datetime

import pandas as pd


FEEDBACK_FILE = "feedback_log.jsonl"


def record_feedback(
    batch_id: str,
    bank_trx_id: str,
    predicted_label: int,
    human_label: int,
    confidence: float,
    reviewer: str = "system",
    feedback_dir: str = ".",
) -> None:
    """
    Append one feedback record.

    Parameters
    ----------
    batch_id : settlement batch (or group_key for many-to-one)
    bank_trx_id : bank transaction ID
    predicted_label : what the model said (0 or 1)
    human_label : what the human decided (0 or 1)
    confidence : model's probability
    reviewer : who reviewed
    feedback_dir : folder to store the JSONL file
    """
    os.makedirs(feedback_dir, exist_ok=True)
    path = os.path.join(feedback_dir, FEEDBACK_FILE)
    record = {
        "timestamp": datetime.now().isoformat(),
        "batch_id": batch_id,
        "bank_trx_id": bank_trx_id,
        "predicted_label": int(predicted_label),
        "human_label": int(human_label),
        "confidence": float(confidence),
        "reviewer": reviewer,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_feedback(feedback_dir: str = ".") -> pd.DataFrame:
    """Load all feedback records into a DataFrame."""
    path = os.path.join(feedback_dir, FEEDBACK_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def merge_feedback_into_training(
    training_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge human feedback into the training set.
    - If a (batch_id, bank_trx_id) pair has feedback, use human_label.
    - Otherwise keep original match_label.

    Returns a new DataFrame ready for retraining.
    """
    if feedback_df.empty:
        return training_df

    # Identify the key columns
    key_col = "group_key" if "group_key" in training_df.columns else "batch_id"

    fb = feedback_df.rename(columns={"human_label": "feedback_label", "batch_id": key_col})
    fb = fb[[key_col, "bank_trx_id", "feedback_label"]].drop_duplicates(
        subset=[key_col, "bank_trx_id"], keep="last"
    )

    merged = training_df.merge(fb, on=[key_col, "bank_trx_id"], how="left")
    merged["match_label"] = merged["feedback_label"].fillna(merged["match_label"]).astype(int)
    merged = merged.drop(columns=["feedback_label"])

    return merged
