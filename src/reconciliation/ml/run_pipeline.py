"""
run_pipeline.py
---------------
End-to-end orchestrator that ties every module together:

  1. Load data
  2. Generate candidate pairs  (1-to-1 + many-to-one)
  3. Label candidates with ground truth
  4. Engineer features
  5. Train & evaluate models
  6. Score candidates and apply threshold logic
  7. Print reconciliation summary

Run from the ml/ folder:
    python run_pipeline.py
"""

import os
import sys

import pandas as pd

# Ensure local imports resolve when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from data_ingestion import load_all
from candidate_generation import (
    generate_one_to_one_candidates,
    generate_many_to_one_candidates,
    label_candidates,
)
from feature_engineering import build_features, FEATURE_COLUMNS
from model_training import train_and_evaluate
from prediction import predict, resolve_conflicts, reconciliation_summary, load_model

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = os.path.dirname(__file__)          # same folder as the CSVs
MODEL_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data …")
    batches, bank, mapping = load_all(DATA_DIR)
    print(f"  Batches : {len(batches)}")
    print(f"  Bank txn: {len(bank)}")
    print(f"  Mapping : {len(mapping)}  (pos={int((mapping.match_label==1).sum())}, "
          f"neg={int((mapping.match_label==0).sum())})")

    # ------------------------------------------------------------------
    # 2. Generate candidate pairs
    # ------------------------------------------------------------------
    print("\n[2/6] Generating candidate pairs …")
    cand_1to1 = generate_one_to_one_candidates(batches, bank)
    print(f"  1-to-1 candidates : {len(cand_1to1)}")

    cand_mto1 = generate_many_to_one_candidates(batches, bank, max_group_size=3)
    print(f"  Many-to-1 candidates: {len(cand_mto1)}")

    # Label candidates with ground truth
    cand_1to1 = label_candidates(cand_1to1, mapping)
    cand_mto1 = label_candidates(cand_mto1, mapping)

    print(f"  1-to-1   pos/neg: {int(cand_1to1.match_label.sum())} / "
          f"{int((cand_1to1.match_label==0).sum())}")
    print(f"  Many-to-1 pos/neg: {int(cand_mto1.match_label.sum())} / "
          f"{int((cand_mto1.match_label==0).sum())}")

    # For now train on 1-to-1 candidates (most data, cleaner signal).
    # Many-to-one candidates can be merged in once the 1-to-1 model is stable.
    candidates = cand_1to1.copy()

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("\n[3/6] Engineering features …")
    candidates = build_features(candidates)
    print(f"  Feature columns: {FEATURE_COLUMNS}")
    print(f"  Dataset shape  : {candidates.shape}")

    # Save feature matrix for inspection
    candidates.to_csv(os.path.join(OUTPUT_DIR, "feature_matrix.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Train & evaluate models
    # ------------------------------------------------------------------
    print("\n[4/6] Training models …")
    result = train_and_evaluate(candidates, model_dir=MODEL_DIR)

    # ------------------------------------------------------------------
    # 5. Score with best model & apply thresholds
    # ------------------------------------------------------------------
    print("\n[5/6] Scoring candidates with best model …")
    # Find best model file (most recent pkl starting with preferred name)
    best_name = result["best_model_name"].lower().replace(" ", "_")
    model_files = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.startswith(best_name) and f.endswith(".pkl")]
    )
    if not model_files:
        # fallback: any pkl
        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    best_path = os.path.join(MODEL_DIR, model_files[-1])
    print(f"  Using model: {best_path}")

    artefact = load_model(best_path)
    scored = predict(candidates, artefact)
    scored = resolve_conflicts(scored)

    scored.to_csv(os.path.join(OUTPUT_DIR, "scored_candidates.csv"), index=False)
    print(f"  Scored candidates saved → {os.path.join(OUTPUT_DIR, 'scored_candidates.csv')}")

    # ------------------------------------------------------------------
    # 6. Reconciliation summary
    # ------------------------------------------------------------------
    print("\n[6/6] Reconciliation summary")
    summary = reconciliation_summary(scored)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "reconciliation_summary.csv"), index=False)

    # Print sample auto-matches
    auto = scored[scored["match_status"] == "auto_match"].sort_values("confidence", ascending=False)
    if not auto.empty:
        print(f"\n  Top auto-matches ({len(auto)} total):")
        cols = ["batch_id", "bank_trx_id", "batch_net", "bank_amount", "confidence"]
        print(auto[cols].head(10).to_string(index=False))

    # Print metrics summary
    print("\n  Model comparison:")
    for m in result["metrics"]:
        print(f"    {m['model']:25s}  ROC-AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"F1={m['f1_1']:.4f}")

    print("\n✓ Pipeline complete.  Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
