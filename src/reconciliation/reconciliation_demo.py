"""
Shopee Seller Reconciliation Demo System
==========================================
Generates realistic sample data and performs two-level reconciliation:
  1. Order → Seller Balance Release
  2. Withdrawal → Bank Statement

Dependencies: pandas, rapidfuzz
"""

import os
import random
import datetime
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "demo_output")
SEED = 42  # reproducible results

random.seed(SEED)


# ===================================================================
# STEP 1 — GENERATE SAMPLE DATA
# ===================================================================

def generate_sample_data(num_orders: int = 50, num_withdrawals: int = 5) -> dict:
    """
    Create synthetic Shopee seller demo data.

    Returns a dict of DataFrames:
        orders, seller_balance_release, withdrawals, bank_statement
    """
    base_date = datetime.date(2025, 6, 1)

    # ------------------------------------------------------------------
    # 1. orders.csv
    # ------------------------------------------------------------------
    orders = []
    for i in range(num_orders):
        order_id = f"ORD{1001 + i}"
        completed_date = base_date + datetime.timedelta(days=random.randint(0, 29))
        gross_amount = round(random.uniform(50, 500), 2)
        commission_fee = round(gross_amount * 0.10, 2)
        net_amount = round(gross_amount - commission_fee, 2)
        orders.append({
            "order_id": order_id,
            "completed_date": completed_date,
            "gross_amount": gross_amount,
            "commission_fee": commission_fee,
            "net_amount": net_amount,
        })
    df_orders = pd.DataFrame(orders)

    # ------------------------------------------------------------------
    # 2. seller_balance_release.csv
    #    Intentional anomalies:
    #      - 3 missing releases
    #      - 3 amount mismatches
    #      - 3 delayed releases (5 days instead of 2–3)
    # ------------------------------------------------------------------
    missing_idx = random.sample(range(num_orders), 3)          # indices to skip
    mismatch_idx = random.sample(
        [i for i in range(num_orders) if i not in missing_idx], 3
    )
    delay_idx = random.sample(
        [i for i in range(num_orders) if i not in missing_idx and i not in mismatch_idx], 3
    )

    releases = []
    for i, row in df_orders.iterrows():
        if i in missing_idx:
            continue  # simulate missing release

        delay_days = 5 if i in delay_idx else random.randint(2, 3)
        release_date = row["completed_date"] + datetime.timedelta(days=delay_days)

        release_amount = row["net_amount"]
        if i in mismatch_idx:
            # introduce a small random discrepancy (±1–5 %)
            factor = random.choice([0.95, 0.97, 1.03, 1.05])
            release_amount = round(release_amount * factor, 2)

        releases.append({
            "order_id": row["order_id"],
            "release_date": release_date,
            "release_amount": release_amount,
        })
    df_releases = pd.DataFrame(releases)

    # ------------------------------------------------------------------
    # 3. withdrawals.csv
    #    Group consecutive releases into withdrawal batches.
    # ------------------------------------------------------------------
    df_releases_sorted = df_releases.sort_values("release_date").reset_index(drop=True)
    batch_size = len(df_releases_sorted) // num_withdrawals

    withdrawals = []
    wd_release_map = {}  # withdrawal_id → list of release rows
    for w in range(num_withdrawals):
        start = w * batch_size
        end = start + batch_size if w < num_withdrawals - 1 else len(df_releases_sorted)
        batch = df_releases_sorted.iloc[start:end]

        withdrawal_id = f"WD{2001 + w}"
        withdrawal_date = batch["release_date"].max() + datetime.timedelta(days=1)
        withdrawal_amount = round(batch["release_amount"].sum(), 2)
        withdrawal_fee = random.choice([0.00, 0.12])

        withdrawals.append({
            "withdrawal_id": withdrawal_id,
            "withdrawal_date": withdrawal_date,
            "withdrawal_amount": withdrawal_amount,
            "withdrawal_fee": withdrawal_fee,
        })
        wd_release_map[withdrawal_id] = batch

    df_withdrawals = pd.DataFrame(withdrawals)

    # ------------------------------------------------------------------
    # 4. bank_statement.csv
    #    Intentional anomalies:
    #      - 1 missing bank entry
    #      - 1 wrong amount
    #      - 1 description without Shopee keyword
    # ------------------------------------------------------------------
    shopee_descs = [
        "Shopee Seller Payout",
        "SPMY WD",
        "Shopee MY",
        "Shopee Payout",
        "Shopee Settlement",
    ]
    non_shopee_descs = [
        "General Transfer",
        "Merchant Payment",
        "E-Comm Settlement",
    ]

    missing_bank_idx = random.choice(range(num_withdrawals))
    wrong_amt_idx = random.choice(
        [i for i in range(num_withdrawals) if i != missing_bank_idx]
    )
    bad_desc_idx = random.choice(
        [i for i in range(num_withdrawals) if i not in (missing_bank_idx, wrong_amt_idx)]
    )

    bank_rows = []
    for i, wd in df_withdrawals.iterrows():
        if i == missing_bank_idx:
            continue  # simulate missing bank entry

        bank_date = wd["withdrawal_date"] + datetime.timedelta(days=random.randint(1, 3))
        bank_amount = round(wd["withdrawal_amount"] - wd["withdrawal_fee"], 2)

        if i == wrong_amt_idx:
            bank_amount = round(bank_amount * random.choice([0.95, 1.07]), 2)

        description = (
            random.choice(non_shopee_descs)
            if i == bad_desc_idx
            else random.choice(shopee_descs)
        )

        bank_rows.append({
            "withdrawal_id": wd["withdrawal_id"],   # reference for verification
            "bank_date": bank_date,
            "bank_amount": bank_amount,
            "description": description,
        })
    df_bank = pd.DataFrame(bank_rows)

    return {
        "orders": df_orders,
        "seller_balance_release": df_releases,
        "withdrawals": df_withdrawals,
        "bank_statement": df_bank,
    }


# ===================================================================
# STEP 2 — ORDER → SELLER BALANCE RECONCILIATION
# ===================================================================

def reconcile_orders(df_orders: pd.DataFrame, df_releases: pd.DataFrame) -> pd.DataFrame:
    """
    Match orders to balance releases by order_id.

    Checks:
      - Amount equality
      - Release date within completed_date +2 to +4 days

    Returns a DataFrame with reconciliation results.
    """
    merged = df_orders.merge(df_releases, on="order_id", how="left")

    results = []
    for _, row in merged.iterrows():
        order_id = row["order_id"]
        completed = row["completed_date"]
        net = row["net_amount"]
        release_date = row.get("release_date")
        release_amt = row.get("release_amount")

        # --- determine date_difference ---
        if pd.isna(release_date):
            date_diff = None
        else:
            date_diff = (pd.Timestamp(release_date) - pd.Timestamp(completed)).days

        # --- determine amount_difference ---
        if pd.isna(release_amt):
            amt_diff = None
        else:
            amt_diff = round(release_amt - net, 2)

        # --- classify status ---
        if pd.isna(release_date) or pd.isna(release_amt):
            status = "missing_release"
        elif amt_diff != 0:
            status = "amount_difference"
        elif date_diff is not None and not (2 <= date_diff <= 4):
            status = "timing_difference"
        else:
            status = "matched"

        results.append({
            "order_id": order_id,
            "completed_date": completed,
            "net_amount": net,
            "release_date": release_date if not pd.isna(release_date) else None,
            "release_amount": release_amt if not pd.isna(release_amt) else None,
            "date_difference": date_diff,
            "amount_difference_value": amt_diff,
            "status": status,
        })

    return pd.DataFrame(results)


# ===================================================================
# STEP 3 — WITHDRAWAL → BANK RECONCILIATION (scoring model)
# ===================================================================

def _score_amount(wd_net: float, bank_amt: float) -> int:
    """Exact match → 50 pts, otherwise 0."""
    return 50 if abs(wd_net - bank_amt) < 0.01 else 0


def _score_date(wd_date, bank_date, tolerance: int = 4) -> int:
    """Within tolerance days → 30 pts, otherwise 0."""
    diff = abs((pd.Timestamp(bank_date) - pd.Timestamp(wd_date)).days)
    return 30 if diff <= tolerance else 0


def _score_description(description: str, keyword: str = "Shopee") -> int:
    """Fuzzy match of description against keyword → up to 20 pts."""
    ratio = fuzz.partial_ratio(keyword.lower(), description.lower())
    return int(20 * ratio / 100)


def reconcile_withdrawals(
    df_withdrawals: pd.DataFrame,
    df_bank: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score every (withdrawal, bank_entry) pair and pick the best match.

    Scoring:
      - Amount exact match → 50 pts
      - Date within 4-day tolerance → 30 pts
      - Fuzzy description match with 'Shopee' → up to 20 pts

    Status:
      - auto_match   (>= 90)
      - suggested_match (70-89)
      - manual_review  (< 70)
    """
    results = []

    for _, wd in df_withdrawals.iterrows():
        wd_id = wd["withdrawal_id"]
        wd_date = wd["withdrawal_date"]
        wd_net = round(wd["withdrawal_amount"] - wd["withdrawal_fee"], 2)

        best_score = 0
        best_bank = None

        for _, bk in df_bank.iterrows():
            score = (
                _score_amount(wd_net, bk["bank_amount"])
                + _score_date(wd_date, bk["bank_date"])
                + _score_description(bk["description"])
            )
            if score > best_score:
                best_score = score
                best_bank = bk

        # classify
        if best_score >= 90:
            status = "auto_match"
        elif best_score >= 70:
            status = "suggested_match"
        else:
            status = "manual_review"

        results.append({
            "withdrawal_id": wd_id,
            "withdrawal_date": wd_date,
            "withdrawal_net": wd_net,
            "matched_bank_date": best_bank["bank_date"] if best_bank is not None else None,
            "matched_bank_amount": best_bank["bank_amount"] if best_bank is not None else None,
            "matched_description": best_bank["description"] if best_bank is not None else None,
            "match_score": best_score,
            "status": status,
        })

    return pd.DataFrame(results)


# ===================================================================
# SUMMARY
# ===================================================================

def print_summary(df_order_recon: pd.DataFrame, df_wd_recon: pd.DataFrame) -> None:
    """Print a clear reconciliation summary to the console."""

    sep = "=" * 60

    # --- Order → Release summary ---
    print(f"\n{sep}")
    print("  ORDER → SELLER BALANCE RECONCILIATION SUMMARY")
    print(sep)
    total = len(df_order_recon)
    counts = df_order_recon["status"].value_counts()
    print(f"  Total orders           : {total}")
    print(f"  Matched                : {counts.get('matched', 0)}")
    print(f"  Timing differences     : {counts.get('timing_difference', 0)}")
    print(f"  Amount differences     : {counts.get('amount_difference', 0)}")
    print(f"  Missing releases       : {counts.get('missing_release', 0)}")
    print(sep)

    # --- Withdrawal → Bank summary ---
    print(f"\n{sep}")
    print("  WITHDRAWAL → BANK RECONCILIATION SUMMARY")
    print(sep)
    total_wd = len(df_wd_recon)
    wd_counts = df_wd_recon["status"].value_counts()
    print(f"  Total withdrawals      : {total_wd}")
    print(f"  Auto-matched           : {wd_counts.get('auto_match', 0)}")
    print(f"  Suggested matches      : {wd_counts.get('suggested_match', 0)}")
    print(f"  Manual review needed   : {wd_counts.get('manual_review', 0)}")
    print(sep)
    print()


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Generate sample data ---
    print("▶ Generating sample data …")
    data = generate_sample_data(num_orders=50, num_withdrawals=5)

    for name, df in data.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved  {path}")

    # --- Step 2: Order → Release reconciliation ---
    print("\n▶ Running Order → Seller Balance reconciliation …")
    df_order_recon = reconcile_orders(data["orders"], data["seller_balance_release"])
    order_recon_path = os.path.join(OUTPUT_DIR, "reconciliation_order_result.csv")
    df_order_recon.to_csv(order_recon_path, index=False)
    print(f"  Saved  {order_recon_path}")

    # --- Step 3: Withdrawal → Bank reconciliation ---
    print("\n▶ Running Withdrawal → Bank reconciliation …")
    df_wd_recon = reconcile_withdrawals(data["withdrawals"], data["bank_statement"])
    wd_recon_path = os.path.join(OUTPUT_DIR, "reconciliation_withdrawal_result.csv")
    df_wd_recon.to_csv(wd_recon_path, index=False)
    print(f"  Saved  {wd_recon_path}")

    # --- Summary ---
    print_summary(df_order_recon, df_wd_recon)
    print("✅ Done — all files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
