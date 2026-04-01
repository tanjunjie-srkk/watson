"""
compare_ml_vs_rules.py
----------------------
Head-to-head comparison of:
    1. Rule-based reconciliation (traditional approach)
    2. ML-based reconciliation   (trained models)

Run from the ml/ folder:
    python compare_ml_vs_rules.py
"""

import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))

from data_ingestion import load_all
from candidate_generation import generate_one_to_one_candidates, label_candidates
from feature_engineering import build_features, FEATURE_COLUMNS

DATA_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(DATA_DIR, "models")


# ======================================================================
# RULE-BASED RECONCILIATION ENGINE
# ======================================================================

def rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Traditional rule-based matching logic:
      1. Amount must be within 2% of batch net
      2. Date difference must be <= 7 days
      3. Bank description must contain a Shopee-related keyword

    If ALL three rules pass → predict Match (1)
    Otherwise → predict Not Match (0)
    """
    rule_amount = df["amount_pct_diff"] <= 0.02
    rule_date = df["date_diff_abs"] <= 7
    rule_keyword = df["keyword_shopee_flag"] == 1

    return (rule_amount & rule_date & rule_keyword).astype(int).values


def rule_based_predict_relaxed(df: pd.DataFrame) -> np.ndarray:
    """
    Relaxed rule-based matching (looser tolerances):
      1. Amount within 5%
      2. Date difference <= 10 days
      3. Keyword flag OR text similarity > 0.3
    """
    rule_amount = df["amount_pct_diff"] <= 0.05
    rule_date = df["date_diff_abs"] <= 10
    rule_text = (df["keyword_shopee_flag"] == 1) | (df["desc_seq_ratio"] > 0.3)

    return (rule_amount & rule_date & rule_text).astype(int).values


def rule_based_predict_strict(df: pd.DataFrame) -> np.ndarray:
    """
    Strict rule-based matching (tight tolerances):
      1. Amount must be exact (within 0.5%)
      2. Date difference <= 3 days
      3. Keyword flag AND text similarity > 0.4
    """
    rule_amount = df["amount_pct_diff"] <= 0.005
    rule_date = df["date_diff_abs"] <= 3
    rule_text = (df["keyword_shopee_flag"] == 1) & (df["desc_seq_ratio"] > 0.4)

    return (rule_amount & rule_date & rule_text).astype(int).values


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_predictions(y_true, y_pred, y_prob, name: str) -> dict:
    """Compute P / R / F1 / AUC for one approach."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0
    pr = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0

    result = {
        "approach": name,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "false_positive_rate": fp / max(fp + tn, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
    }
    return result


def print_comparison(results: list[dict]) -> None:
    """Pretty-print a comparison table."""
    df = pd.DataFrame(results)

    print("\n" + "=" * 100)
    print("  HEAD-TO-HEAD COMPARISON:  Rule-Based  vs  Machine Learning")
    print("=" * 100)

    cols = [
        "approach", "precision", "recall", "f1",
        "roc_auc", "pr_auc",
        "tp", "fp", "fn", "tn",
        "false_positive_rate", "false_negative_rate",
    ]
    print(df[cols].to_string(index=False, float_format="%.4f"))

    # ---- Interpretation ----
    best_f1 = df.loc[df["f1"].idxmax()]
    best_prauc = df.loc[df["pr_auc"].idxmax()]
    lowest_fp = df.loc[df["false_positive_rate"].idxmin()]
    lowest_fn = df.loc[df["false_negative_rate"].idxmin()]

    print("\n  Key insights:")
    print(f"    Best F1 score        → {best_f1['approach']}  (F1={best_f1['f1']:.4f})")
    print(f"    Best PR-AUC          → {best_prauc['approach']}  (PR-AUC={best_prauc['pr_auc']:.4f})")
    print(f"    Lowest false positives → {lowest_fp['approach']}  (FPR={lowest_fp['false_positive_rate']:.4f})")
    print(f"    Lowest false negatives → {lowest_fn['approach']}  (FNR={lowest_fn['false_negative_rate']:.4f})")

    # ---- Error analysis ----
    print("\n  Error analysis (what each approach gets WRONG):")
    for _, row in df.iterrows():
        name = row["approach"]
        print(f"\n    {name}:")
        print(f"      False Positives (wrongly matched)  : {row['fp']}")
        print(f"      False Negatives (missed real match) : {row['fn']}")
        total_errors = row["fp"] + row["fn"]
        total = row["tp"] + row["fp"] + row["fn"] + row["tn"]
        print(f"      Total errors                       : {total_errors} / {total}  "
              f"({total_errors/total*100:.1f}%)")


# ======================================================================
# MAIN
# ======================================================================

def main():
    # ---- Load & prepare data (same as training pipeline) ----
    batches, bank, mapping = load_all(DATA_DIR)
    candidates = generate_one_to_one_candidates(batches, bank)
    candidates = label_candidates(candidates, mapping)
    candidates = build_features(candidates)

    y_true = candidates["match_label"].values
    X = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)

    results = []

    # ---- Rule-based approaches ----
    for name, predict_fn in [
        ("Rule-Based (Strict)", rule_based_predict_strict),
        ("Rule-Based (Standard)", rule_based_predict),
        ("Rule-Based (Relaxed)", rule_based_predict_relaxed),
    ]:
        y_pred = predict_fn(candidates)
        # Rules produce binary; use as probability too (0 or 1)
        results.append(evaluate_predictions(y_true, y_pred, y_pred.astype(float), name))

    # ---- ML approaches ----
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    for mf in sorted(model_files):
        path = os.path.join(MODEL_DIR, mf)
        with open(path, "rb") as f:
            artefact = pickle.load(f)
        model = artefact["model"]
        scaler = artefact.get("scaler")
        model_type = type(model).__name__

        X_input = X.copy()
        if model_type == "LogisticRegression" and scaler is not None:
            X_input = scaler.transform(X_input)

        y_prob = model.predict_proba(X_input)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        label = f"ML: {model_type}"
        results.append(evaluate_predictions(y_true, y_pred, y_prob, label))

    # ---- Print comparison ----
    print_comparison(results)

    # ---- Save to CSV ----
    out_path = os.path.join(DATA_DIR, "output", "ml_vs_rules_comparison.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n  Comparison saved → {out_path}")


if __name__ == "__main__":
    main()
