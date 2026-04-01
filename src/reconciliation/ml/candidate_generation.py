"""
candidate_generation.py
-----------------------
Generate plausible (batch, bank_trx) candidate pairs using rule-based
blocking on date window and amount tolerance.  This dramatically reduces
the search space compared to a full cross-join.
"""

import itertools
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 1-to-1 candidates
# ---------------------------------------------------------------------------

def generate_one_to_one_candidates(
    batches: pd.DataFrame,
    bank: pd.DataFrame,
    date_window_days: int = 10,
    amount_tolerance_pct: float = 0.15,
) -> pd.DataFrame:
    """
    For every (batch, bank_trx) pair where:
      - |settlement_date - credit_date| <= date_window_days
      - |total_net - amount| / total_net  <= amount_tolerance_pct
    create a candidate row.
    """
    pairs = []
    for _, b in batches.iterrows():
        for _, t in bank.iterrows():
            day_diff = abs((b["settlement_date"] - t["credit_date"]).days)
            if day_diff > date_window_days:
                continue
            if b["total_net"] == 0:
                continue
            amt_diff_pct = abs(b["total_net"] - t["amount"]) / abs(b["total_net"])
            if amt_diff_pct > amount_tolerance_pct:
                continue
            pairs.append(
                {
                    "batch_id": b["batch_id"],
                    "bank_trx_id": t["bank_trx_id"],
                    "settlement_date": b["settlement_date"],
                    "credit_date": t["credit_date"],
                    "batch_net": b["total_net"],
                    "bank_amount": t["amount"],
                    "batch_gross": b["total_gross"],
                    "batch_refund": b["total_refund"],
                    "batch_order_count": b["number_of_orders"],
                    "bank_description": t.get("description", ""),
                    "bank_description_clean": t.get("description_clean", ""),
                }
            )
    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Many-to-one candidates  (group N consecutive batches → 1 bank deposit)
# ---------------------------------------------------------------------------

def generate_many_to_one_candidates(
    batches: pd.DataFrame,
    bank: pd.DataFrame,
    max_group_size: int = 3,
    date_window_days: int = 10,
    amount_tolerance_pct: float = 0.15,
) -> pd.DataFrame:
    """
    Try grouping 2..max_group_size consecutive batches (by settlement_date)
    and see if their combined net matches a single bank transaction.
    Returns one row per (group_key, bank_trx_id).
    """
    sorted_batches = batches.sort_values("settlement_date").reset_index(drop=True)
    pairs = []

    for size in range(2, max_group_size + 1):
        for start_idx in range(len(sorted_batches) - size + 1):
            group = sorted_batches.iloc[start_idx : start_idx + size]
            group_net = group["total_net"].sum()
            group_last_date = group["settlement_date"].max()
            group_key = "+".join(group["batch_id"].tolist())

            for _, t in bank.iterrows():
                day_diff = abs((group_last_date - t["credit_date"]).days)
                if day_diff > date_window_days:
                    continue
                if group_net == 0:
                    continue
                amt_diff_pct = abs(group_net - t["amount"]) / abs(group_net)
                if amt_diff_pct > amount_tolerance_pct:
                    continue
                pairs.append(
                    {
                        "group_key": group_key,
                        "bank_trx_id": t["bank_trx_id"],
                        "settlement_date": group_last_date,
                        "credit_date": t["credit_date"],
                        "batch_net": round(group_net, 2),
                        "bank_amount": t["amount"],
                        "batch_gross": round(group["total_gross"].sum(), 2),
                        "batch_refund": round(group["total_refund"].sum(), 2),
                        "batch_order_count": int(group["number_of_orders"].sum()),
                        "bank_description": t.get("description", ""),
                        "bank_description_clean": t.get("description_clean", ""),
                        "group_size": size,
                    }
                )

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Merge with ground-truth labels
# ---------------------------------------------------------------------------

def label_candidates(
    candidates: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join candidate pairs with the ground-truth mapping to assign
    match_label.  Unmatched pairs get label 0.
    """
    if "group_key" in candidates.columns:
        # For many-to-one candidates, expand group_key into individual batch_ids
        # and check if ALL constituent batches map to the same bank_trx_id.
        positive_set = set(
            zip(mapping.loc[mapping["match_label"] == 1, "batch_id"],
                mapping.loc[mapping["match_label"] == 1, "bank_trx_id"])
        )
        labels = []
        for _, row in candidates.iterrows():
            batch_ids = row["group_key"].split("+")
            bank_id = row["bank_trx_id"]
            if all((bid, bank_id) in positive_set for bid in batch_ids):
                labels.append(1)
            else:
                labels.append(0)
        candidates = candidates.copy()
        candidates["match_label"] = labels
    else:
        candidates = candidates.merge(
            mapping[["batch_id", "bank_trx_id", "match_label"]],
            on=["batch_id", "bank_trx_id"],
            how="left",
        )
        candidates["match_label"] = candidates["match_label"].fillna(0).astype(int)

    return candidates
