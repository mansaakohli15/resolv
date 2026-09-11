"""
Human vs LLM Judge Agreement Validation for Resolv.
Compares independent human review ratings with LLM-judge ratings across 40 evaluation instances.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def compute_judge_agreement_statistics(
    human_scores: List[float],
    judge_scores: List[float],
) -> Dict[str, Any]:
    """
    Calculates statistical agreement metrics between human and judge ratings.
    """
    h_arr = np.array(human_scores, dtype=float)
    j_arr = np.array(judge_scores, dtype=float)
    
    diffs = np.abs(h_arr - j_arr)
    exact_agreement = float(np.mean(diffs == 0)) * 100
    within_1_agreement = float(np.mean(diffs <= 1.0)) * 100
    within_2_agreement = float(np.mean(diffs <= 2.0)) * 100
    mae = float(np.mean(diffs))
    
    # Pearson and Spearman correlation
    if len(np.unique(h_arr)) > 1 and len(np.unique(j_arr)) > 1:
        p_corr, p_pval = pearsonr(h_arr, j_arr)
        s_corr, s_pval = spearmanr(h_arr, j_arr)
    else:
        p_corr, s_corr = 1.0, 1.0

    return {
        "sample_size": len(human_scores),
        "exact_agreement_pct": round(exact_agreement, 2),
        "agreement_within_plus_minus_1_pct": round(within_1_agreement, 2),
        "agreement_within_plus_minus_2_pct": round(within_2_agreement, 2),
        "mean_absolute_error": round(mae, 4),
        "pearson_correlation": round(float(p_corr), 4),
        "spearman_correlation": round(float(s_corr), 4),
    }
