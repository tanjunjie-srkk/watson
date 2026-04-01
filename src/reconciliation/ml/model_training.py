"""
model_training.py
-----------------
Train, evaluate, and persist reconciliation models.
Supports:
  - Logistic Regression (baseline)
  - XGBoost  (primary)
  - LightGBM (optional comparison)

Produces classification report, ROC-AUC, PR-AUC and feature importances.
"""

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Optional: XGBoost / LightGBM (graceful fallback)
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_model(model, scaler, model_dir: str, model_name: str) -> str:
    _ensure_dir(model_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{model_name}_{ts}.pkl"
    fpath = os.path.join(model_dir, fname)
    with open(fpath, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": FEATURE_COLUMNS}, f)
    print(f"  Model saved → {fpath}")
    return fpath


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict:
    """Compute and print metrics. Returns dict of scores."""
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)

    roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"  {model_name}  evaluation")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_1": report.get("1", {}).get("precision", 0),
        "recall_1": report.get("1", {}).get("recall", 0),
        "f1_1": report.get("1", {}).get("f1-score", 0),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    model_dir: str = "models",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Train all available models, evaluate on hold-out set, and persist
    the best one.

    Parameters
    ----------
    df : DataFrame with FEATURE_COLUMNS + 'match_label'
    model_dir : where to save .pkl files
    test_size : hold-out fraction
    seed : random state

    Returns
    -------
    dict with keys: best_model_name, best_model_path, metrics (list of dicts)
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["match_label"].values

    # Handle any remaining NaN / inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Scaler (needed for Logistic Regression; tree models don't need it but
    # we keep it uniform in the saved artefact)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---- 1. Logistic Regression baseline ----
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    lr.fit(X_train_scaled, y_train)
    prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, prob_lr, "LogisticRegression"))
    _save_model(lr, scaler, model_dir, "logistic_regression")

    # ---- 2. XGBoost ----
    if HAS_XGB:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / max(n_pos, 1)
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        prob_xgb = xgb.predict_proba(X_test)[:, 1]
        results.append(evaluate(y_test, prob_xgb, "XGBoost"))
        _save_model(xgb, scaler, model_dir, "xgboost")

        # Feature importances
        imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\n  XGBoost feature importances:")
        for feat, val in imp.items():
            print(f"    {feat:30s} {val:.4f}")
    else:
        print("\n  [SKIP] XGBoost not installed — pip install xgboost")

    # ---- 3. LightGBM ----
    if HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            is_unbalance=True,
            random_state=seed,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
        results.append(evaluate(y_test, prob_lgbm, "LightGBM"))
        _save_model(lgbm, scaler, model_dir, "lightgbm")

        imp = pd.Series(lgbm.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\n  LightGBM feature importances:")
        for feat, val in imp.items():
            print(f"    {feat:30s} {val:.4f}")
    else:
        print("\n  [SKIP] LightGBM not installed — pip install lightgbm")

    # ---- Pick best model by PR-AUC (better for imbalanced data) ----
    best = max(results, key=lambda r: r["pr_auc"])
    print(f"\n  ★ Best model: {best['model']}  (PR-AUC={best['pr_auc']:.4f})")

    return {
        "best_model_name": best["model"],
        "metrics": results,
    }
