import random
from datetime import datetime

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def generate_orders(rng: np.random.Generator, start_date: str, months: int) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    all_dates = pd.date_range(start, end, freq="D")

    orders = []
    order_id = 1

    for date in all_dates:
        daily_count = rng.integers(50, 201)
        gross = rng.uniform(20, 500, size=daily_count).round(2)
        commission_rate = rng.uniform(0.02, 0.05, size=daily_count)
        fee_rate = rng.uniform(0.01, 0.02, size=daily_count)
        shipping_subsidy = rng.uniform(0, 10, size=daily_count).round(2)
        refund_flag = rng.random(size=daily_count) < rng.uniform(0.05, 0.10)
        refund_amount = np.where(refund_flag, rng.uniform(1, 200, size=daily_count), 0).round(2)

        commission_fee = (gross * commission_rate).round(2)
        transaction_fee = (gross * fee_rate).round(2)
        net_amount = (gross - commission_fee - transaction_fee - refund_amount + shipping_subsidy).round(2)

        for i in range(daily_count):
            orders.append(
                {
                    "order_id": f"ORD{order_id:07d}",
                    "order_date": date.date(),
                    "gross_amount": float(gross[i]),
                    "commission_fee": float(commission_fee[i]),
                    "transaction_fee": float(transaction_fee[i]),
                    "shipping_subsidy": float(shipping_subsidy[i]),
                    "refund_flag": int(refund_flag[i]),
                    "refund_amount": float(refund_amount[i]),
                    "net_amount": float(net_amount[i]),
                }
            )
            order_id += 1

    return pd.DataFrame(orders)


def generate_batches(rng: np.random.Generator, orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.sort_values("order_date").copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    batches = []
    batch_id = 1
    i = 0
    n = len(orders)

    while i < n:
        window_days = int(rng.integers(2, 4))
        batch_start = orders.iloc[i]["order_date"]
        batch_end = batch_start + pd.Timedelta(days=window_days - 1)

        mask = (orders["order_date"] >= batch_start) & (orders["order_date"] <= batch_end)
        batch_orders = orders[mask]

        if batch_orders.empty:
            i += 1
            continue

        settlement_date = batch_orders["order_date"].max() + pd.Timedelta(days=int(rng.integers(3, 8)))

        batches.append(
            {
                "batch_id": f"BATCH{batch_id:06d}",
                "batch_start_date": batch_orders["order_date"].min().date(),
                "batch_end_date": batch_orders["order_date"].max().date(),
                "settlement_date": settlement_date.date(),
                "total_gross": float(batch_orders["gross_amount"].sum().round(2)),
                "total_net": float(batch_orders["net_amount"].sum().round(2)),
                "total_refund": float(batch_orders["refund_amount"].sum().round(2)),
                "number_of_orders": int(batch_orders.shape[0]),
            }
        )
        batch_id += 1
        i += batch_orders.shape[0]

    return pd.DataFrame(batches)


def noisy_description(rng: np.random.Generator) -> str:
    base = rng.choice(
        [
            "SHOPEE DIGITAL",
            "Shopee Payout",
            "SP Settlement",
            "Shopee Digital MY",
            "SHOPEE-PAYOUT",
        ]
    )
    ref = f"{rng.integers(100000, 999999)}"
    noise = rng.choice(["", "  ", "-", " ", "  "])
    text = f"{base}{noise}{ref}"

    if rng.random() < 0.3:
        text = text.swapcase()
    if rng.random() < 0.2:
        text = text.replace(" ", rng.choice(["  ", " "]))
    if rng.random() < 0.1:
        text = text.replace("O", "0")

    return text.strip()


def generate_bank_transactions(
    rng: np.random.Generator, batches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bank_rows = []
    mapping_rows = []

    batch_ids = list(batches["batch_id"])
    rng.shuffle(batch_ids)

    combined_queue = []
    bank_id = 1

    for batch_id in batch_ids:
        batch = batches.loc[batches["batch_id"] == batch_id].iloc[0]
        payout_type = rng.choice(
            ["normal", "split", "combined", "delayed", "missing"],
            p=[0.75, 0.10, 0.08, 0.05, 0.02],
        )

        if payout_type == "missing":
            continue

        if payout_type == "combined":
            combined_queue.append(batch)
            if len(combined_queue) < 2:
                continue
            batch_a, batch_b = combined_queue[:2]
            combined_queue = combined_queue[2:]
            amount = round(float(batch_a["total_net"]) + float(batch_b["total_net"]), 2)
            credit_date = pd.to_datetime(batch_b["settlement_date"]) + pd.Timedelta(
                days=int(rng.integers(0, 2))
            )
            bank_trx_id = f"BANK{bank_id:07d}"
            bank_rows.append(
                {
                    "bank_trx_id": bank_trx_id,
                    "credit_date": credit_date.date(),
                    "amount": amount,
                    "description": noisy_description(rng),
                }
            )
            mapping_rows.append({"batch_id": batch_a["batch_id"], "bank_trx_id": bank_trx_id, "match_label": 1})
            mapping_rows.append({"batch_id": batch_b["batch_id"], "bank_trx_id": bank_trx_id, "match_label": 1})
            bank_id += 1
            continue

        if payout_type == "split":
            split_ratio = rng.uniform(0.4, 0.6)
            amount_total = float(batch["total_net"])
            amount_1 = round(amount_total * split_ratio, 2)
            amount_2 = round(amount_total - amount_1, 2)
            for amount in [amount_1, amount_2]:
                bank_trx_id = f"BANK{bank_id:07d}"
                credit_date = pd.to_datetime(batch["settlement_date"]) + pd.Timedelta(
                    days=int(rng.integers(0, 2))
                )
                bank_rows.append(
                    {
                        "bank_trx_id": bank_trx_id,
                        "credit_date": credit_date.date(),
                        "amount": amount,
                        "description": noisy_description(rng),
                    }
                )
                mapping_rows.append({"batch_id": batch["batch_id"], "bank_trx_id": bank_trx_id, "match_label": 1})
                bank_id += 1
            continue

        credit_date = pd.to_datetime(batch["settlement_date"]) + pd.Timedelta(
            days=int(rng.integers(0, 2))
        )
        if payout_type == "delayed":
            credit_date += pd.Timedelta(days=int(rng.integers(3, 6)))

        bank_trx_id = f"BANK{bank_id:07d}"
        bank_rows.append(
            {
                "bank_trx_id": bank_trx_id,
                "credit_date": credit_date.date(),
                "amount": float(batch["total_net"]),
                "description": noisy_description(rng),
            }
        )
        mapping_rows.append({"batch_id": batch["batch_id"], "bank_trx_id": bank_trx_id, "match_label": 1})
        bank_id += 1

    bank_df = pd.DataFrame(bank_rows)
    mapping_df = pd.DataFrame(mapping_rows)

    return bank_df, mapping_df


def inject_noise(rng: np.random.Generator, bank_df: pd.DataFrame) -> pd.DataFrame:
    bank_df = bank_df.copy()
    n = len(bank_df)

    rounding_idx = rng.choice(bank_df.index, size=max(1, int(0.10 * n)), replace=False)
    bank_df.loc[rounding_idx, "amount"] += rng.integers(-5, 6, size=len(rounding_idx))

    incorrect_idx = rng.choice(bank_df.index, size=max(1, int(0.05 * n)), replace=False)
    bank_df.loc[incorrect_idx, "amount"] *= rng.uniform(0.98, 1.02, size=len(incorrect_idx))
    bank_df["amount"] = bank_df["amount"].round(2)

    dup_count = max(1, int(0.03 * n))
    dup_rows = bank_df.sample(dup_count, random_state=42)
    dup_rows = dup_rows.assign(bank_trx_id=dup_rows["bank_trx_id"].apply(lambda x: f"{x}_DUP"))

    bank_df = pd.concat([bank_df, dup_rows], ignore_index=True)

    if rng.random() < 0.2:
        corrupt_idx = rng.choice(bank_df.index, size=max(1, int(0.10 * n)), replace=False)
        bank_df.loc[corrupt_idx, "description"] = bank_df.loc[corrupt_idx, "description"].str.replace(
            " ", "", regex=False
        )

    return bank_df


def generate_negative_pairs(
    rng: np.random.Generator, batches: pd.DataFrame, bank_df: pd.DataFrame, mapping_df: pd.DataFrame
) -> pd.DataFrame:
    positives = set(zip(mapping_df["batch_id"], mapping_df["bank_trx_id"]))
    target_neg = len(mapping_df) * 5

    batch_ids = batches["batch_id"].tolist()
    bank_ids = bank_df["bank_trx_id"].tolist()

    negative_rows = []
    while len(negative_rows) < target_neg:
        batch_id = rng.choice(batch_ids)
        bank_id = rng.choice(bank_ids)
        if (batch_id, bank_id) in positives:
            continue
        negative_rows.append({"batch_id": batch_id, "bank_trx_id": bank_id, "match_label": 0})

    return pd.DataFrame(negative_rows)


def main() -> None:
    rng = set_seed(42)

    orders = generate_orders(rng, start_date="2025-09-01", months=6)
    if len(orders) < 20000:
        raise ValueError("Generated orders are fewer than 20,000. Adjust parameters.")

    batches = generate_batches(rng, orders)
    bank_df, mapping_df = generate_bank_transactions(rng, batches)
    bank_df = inject_noise(rng, bank_df)

    negative_df = generate_negative_pairs(rng, batches, bank_df, mapping_df)
    mapping_df = pd.concat([mapping_df, negative_df], ignore_index=True)

    orders.to_csv("Shopee_orders.csv", index=False)
    batches.to_csv("Shopee_batches.csv", index=False)
    bank_df.to_csv("Bank_transactions.csv", index=False)
    mapping_df.to_csv("Match_mapping.csv", index=False)


if __name__ == "__main__":
    main()
