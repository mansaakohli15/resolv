"""
Evaluation and benchmarking utilities for Resolv Intent Classifiers.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from src.config import INTENT_TAXONOMY, ALL_EVAL_CLASSES


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    target_names: List[str] = INTENT_TAXONOMY,
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=target_names,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    
    per_class_metrics = {}
    for intent in target_names:
        if intent in report:
            per_class_metrics[intent] = {
                "precision": round(float(report[intent]["precision"]), 4),
                "recall": round(float(report[intent]["recall"]), 4),
                "f1": round(float(report[intent]["f1-score"]), 4),
                "support": int(report[intent]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "per_class": per_class_metrics,
    }


def evaluate_on_golden_set(
    model,
    golden_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates intent prediction on the 196-row golden evaluation set.
    Separates in-scope intent evaluation (159 examples) from out-of-scope/ambiguous examples (37 examples).
    """
    # 1. In-scope evaluation (examples whose true human intent is in the 8-intent taxonomy)
    in_scope_df = golden_df[golden_df["human_intent"].isin(INTENT_TAXONOMY)].copy()
    in_scope_preds = model.predict(in_scope_df["text"])
    in_scope_metrics = compute_classification_metrics(
        y_true=in_scope_df["human_intent"].tolist(),
        y_pred=in_scope_preds.tolist(),
        target_names=INTENT_TAXONOMY,
    )
    
    # 2. Per-difficulty breakdown on in-scope examples
    difficulty_breakdown = {}
    for diff in ["Easy", "Medium", "Hard"]:
        sub = in_scope_df[in_scope_df["difficulty"] == diff]
        if len(sub) > 0:
            preds = model.predict(sub["text"])
            difficulty_breakdown[diff] = {
                "count": len(sub),
                "accuracy": round(float(accuracy_score(sub["human_intent"], preds)), 4),
                "macro_f1": round(float(f1_score(sub["human_intent"], preds, average="macro", zero_division=0)), 4),
            }

    # 3. Overall golden evaluation (with confidence / threshold evaluation)
    conf_preds = model.predict_with_confidence(golden_df["text"].tolist()) if hasattr(model, "predict_with_confidence") else []
    
    return {
        "in_scope_metrics": in_scope_metrics,
        "difficulty_breakdown": difficulty_breakdown,
        "total_golden_examples": len(golden_df),
        "in_scope_count": len(in_scope_df),
        "out_of_scope_count": len(golden_df) - len(in_scope_df),
    }
