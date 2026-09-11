"""
Milestone 6 — Train and Benchmark Intent Classifiers for Resolv.

Models trained:
1. Baseline 1: Majority Class Classifier
2. Baseline 2: Standard Unigram TF-IDF + Logistic Regression
3. Main Model: Resolv Calibrated Feature Union (Word 1-3 ngrams + Char 3-5 ngrams + Class-Balanced Calibrated LogReg)

Evaluated on:
- Validation Split (held-out validation components)
- Golden Evaluation Set (196 hand-annotated examples)

Output:
- evaluation/classifier_metrics.json
- evaluation/confusion_matrix.csv
- data/processed/intent_classifier.pkl

Run: python scripts/train_classifier.py
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTENT_TAXONOMY, EVALUATION_DIR, PROCESSED_DATA_PATH
from src.data_utils import get_training_data, get_validation_data, load_golden_set
from src.classifier.baselines import MajorityClassBaseline, SimpleTfidfLogisticBaseline
from src.classifier.model import ResolvIntentClassifier
from src.classifier.evaluate import compute_classification_metrics, evaluate_on_golden_set


def main():
    print("=" * 70)
    print("RESOLV — MILESTONE 6: INTENT CLASSIFIER BENCHMARKING")
    print("=" * 70)
    
    # 1. Load Datasets
    print("\n[1/4] Loading training and evaluation splits...")
    t0 = time.time()
    train_df = get_training_data()
    val_df = get_validation_data()
    golden_df = load_golden_set()
    print(f"  Train set (train_retrieval only): {len(train_df):,} examples")
    print(f"  Validation set: {len(val_df):,} examples")
    print(f"  Golden evaluation benchmark: {len(golden_df)} examples (159 in-scope, 37 out-of-scope/ambiguous)")
    print(f"  Data loaded in {time.time() - t0:.2f}s")
    
    X_train, y_train = train_df["text"], train_df["intent"]
    X_val, y_val = val_df["text"], val_df["intent"]
    
    # 2. Train Models
    print("\n[2/4] Training Baseline and Main Models...")
    
    # Baseline 1: Majority Class
    print("  Training Baseline 1 (Majority Class)...")
    b1 = MajorityClassBaseline()
    b1.fit(X_train, y_train)
    
    # Baseline 2: Simple TF-IDF LogReg
    print("  Training Baseline 2 (Simple TF-IDF + LogReg)...")
    b2 = SimpleTfidfLogisticBaseline()
    b2.fit(X_train, y_train)
    
    # Main Model: Resolv Calibrated Feature Union
    print("  Training Main Model (Resolv Calibrated Word+Char Feature Union)...")
    t_start = time.time()
    main_model = ResolvIntentClassifier()
    main_model.fit(X_train, y_train)
    train_duration = time.time() - t_start
    print(f"  Main model trained in {train_duration:.2f}s")
    
    # Save main model
    model_save_path = Path("data/processed/intent_classifier.pkl")
    main_model.save(str(model_save_path))
    print(f"  Saved trained model to {model_save_path}")

    # 3. Evaluate on Validation Set
    print("\n[3/4] Evaluating on Validation Split...")
    val_metrics = {
        "baseline_1_majority": compute_classification_metrics(y_val.tolist(), b1.predict(X_val).tolist()),
        "baseline_2_simple_tfidf": compute_classification_metrics(y_val.tolist(), b2.predict(X_val).tolist()),
        "main_resolv_model": compute_classification_metrics(y_val.tolist(), main_model.predict(X_val).tolist()),
    }
    
    print("\n=== Validation Split Results ===")
    print(f"{'Model':<32} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 72)
    for model_name, m in val_metrics.items():
        print(f"{model_name:<32} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {m['weighted_f1']:<12.4f}")

    # 4. Evaluate on Golden Benchmark
    print("\n[4/4] Evaluating on 196-row Golden Evaluation Benchmark...")
    golden_eval_results = {
        "baseline_1_majority": evaluate_on_golden_set(b1, golden_df),
        "baseline_2_simple_tfidf": evaluate_on_golden_set(b2, golden_df),
        "main_resolv_model": evaluate_on_golden_set(main_model, golden_df),
    }

    print("\n=== Golden Benchmark Results (In-Scope N=159) ===")
    print(f"{'Model':<32} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<12}")
    print("-" * 72)
    for model_name, res in golden_eval_results.items():
        m = res["in_scope_metrics"]
        print(f"{model_name:<32} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {m['weighted_f1']:<12.4f}")

    print("\n=== Main Model Per-Intent Performance on Golden Set ===")
    print(f"{'Intent':<42} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'Support':<8}")
    print("-" * 80)
    for intent, pm in golden_eval_results["main_resolv_model"]["in_scope_metrics"]["per_class"].items():
        print(f"{intent:<42} | {pm['precision']:<8.4f} | {pm['recall']:<8.4f} | {pm['f1']:<8.4f} | {pm['support']:<8}")

    print("\n=== Main Model Performance by Difficulty Tier ===")
    for diff, dm in golden_eval_results["main_resolv_model"]["difficulty_breakdown"].items():
        print(f"  {diff:<8} (N={dm['count']:<2}): Accuracy = {dm['accuracy']:.4f}, Macro F1 = {dm['macro_f1']:.4f}")

    # Confusion Matrix on In-Scope Golden Set
    in_scope_df = golden_df[golden_df["human_intent"].isin(INTENT_TAXONOMY)]
    main_preds = main_model.predict(in_scope_df["text"])
    cm = confusion_matrix(in_scope_df["human_intent"], main_preds, labels=INTENT_TAXONOMY)
    cm_df = pd.DataFrame(cm, index=INTENT_TAXONOMY, columns=INTENT_TAXONOMY)
    cm_save_path = EVALUATION_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_save_path)
    print(f"\nSaved Confusion Matrix to {cm_save_path}")

    # Save Full Metrics JSON
    full_output = {
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "golden_samples_total": len(golden_df),
        "golden_in_scope_count": len(in_scope_df),
        "training_duration_seconds": round(train_duration, 2),
        "validation_metrics": val_metrics,
        "golden_benchmark_metrics": golden_eval_results,
    }
    
    metrics_path = EVALUATION_DIR / "classifier_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved complete metrics to {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
