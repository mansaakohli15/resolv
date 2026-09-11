"""
Retrieval evaluation metrics and benchmarking for Resolv:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Evidence Usefulness & Intent Concordance Rate
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def evaluate_retrieval_metrics(
    retriever,
    eval_queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates retrieval performance against ground-truth intent and relevance criteria.
    Efficiently evaluates in a single pass over candidate results.
    """
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    rr_list = []
    sufficient_count = 0
    intent_concordant_top1 = 0

    n_total = len(eval_queries)
    if n_total == 0:
        return {}

    for q_item in eval_queries:
        query_text = q_item["query"]
        gold_intent = q_item["ground_truth_intent"]
        
        # Single retrieval call for top_k
        candidates = retriever.retrieve(query_text, predicted_intent=gold_intent, top_k=top_k)
        
        if not candidates:
            rr_list.append(0.0)
            continue
            
        top_cand = candidates[0]
        if top_cand["is_sufficient"]:
            sufficient_count += 1
            
        if top_cand["intent"] == gold_intent:
            intent_concordant_top1 += 1
            
        # Check matches at rank 1, 3, 5
        matched_1 = any(c["intent"] == gold_intent for c in candidates[:1])
        matched_3 = any(c["intent"] == gold_intent for c in candidates[:3])
        matched_5 = any(c["intent"] == gold_intent for c in candidates[:5])
        
        if matched_1:
            hits_at_1 += 1
        if matched_3:
            hits_at_3 += 1
        if matched_5:
            hits_at_5 += 1
            
        # Reciprocal rank
        first_match_rank = None
        for rank, cand in enumerate(candidates, start=1):
            if cand["intent"] == gold_intent:
                first_match_rank = rank
                break
                
        if first_match_rank is not None:
            rr_list.append(1.0 / first_match_rank)
        else:
            rr_list.append(0.0)

    recall_at_1 = hits_at_1 / n_total
    recall_at_3 = hits_at_3 / n_total
    recall_at_5 = hits_at_5 / n_total
    mrr = float(np.mean(rr_list))

    return {
        "total_queries_evaluated": n_total,
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "intent_concordance_top1_pct": round(intent_concordant_top1 / n_total * 100, 2),
        "evidence_sufficiency_pct": round(sufficient_count / n_total * 100, 2),
    }
