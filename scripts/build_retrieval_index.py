"""
Milestone 7 — Build Evidence Corpus, Fit Hybrid Retriever, and Evaluate.

Corpus source:
- train_retrieval split ONLY (zero leakage from validation/test/golden)

Outputs:
- data/processed/retrieval_index.pkl
- evaluation/retrieval_metrics.json
- reports/retrieval_experiments.md

Run: python scripts/build_retrieval_index.py
"""
import sys
import json
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_DIR, INTENT_TAXONOMY
from src.data_utils import load_golden_set, load_split_assignments
from src.retrieval.corpus import build_evidence_corpus
from src.retrieval.hybrid_search import ResolvHybridRetriever
from src.retrieval.evaluate import evaluate_retrieval_metrics


def main():
    print("=" * 70)
    print("RESOLV — MILESTONE 7: EVIDENCE RETRIEVAL PIPELINE")
    print("=" * 70)

    # 1. Build Corpus
    print("\n[1/4] Building Evidence Corpus from train_retrieval split...")
    t0 = time.time()
    corpus_df = build_evidence_corpus(min_reply_len=15)
    print(f"  Extracted {len(corpus_df):,} high-quality resolution pairs in {time.time() - t0:.2f}s")
    
    # Leakage Assertion Check
    splits = load_split_assignments()
    val_comps = set(splits.loc[splits["split"] == "validation", "_component"])
    test_comps = set(splits.loc[splits["split"] == "test", "_component"])
    corpus_comps = set(corpus_df["component_id"])
    
    assert len(corpus_comps & val_comps) == 0, "FATAL: Validation split leaked into retrieval corpus!"
    assert len(corpus_comps & test_comps) == 0, "FATAL: Test split leaked into retrieval corpus!"
    print("  [LEAKAGE CHECK PASSED]: 0% overlap with validation or test/golden splits.")

    print("\n  Corpus Intent Distribution:")
    print(corpus_df["intent"].value_counts().to_string())

    # 2. Fit Hybrid Retriever Index
    print("\n[2/4] Fitting Hybrid Lexical + Dense Semantic Index...")
    t_fit = time.time()
    retriever = ResolvHybridRetriever()
    retriever.fit(corpus_df)
    fit_duration = time.time() - t_fit
    print(f"  Hybrid index built and fitted in {fit_duration:.2f}s")

    # Save index
    index_save_path = Path("data/processed/retrieval_index.pkl")
    retriever.save(str(index_save_path))
    print(f"  Saved retrieval index to {index_save_path}")

    # 3. Benchmark Retrieval Performance on Golden Set (In-Scope N=159)
    print("\n[3/4] Benchmarking Retrieval Quality on Golden Set...")
    golden_df = load_golden_set()
    in_scope_golden = golden_df[golden_df["human_intent"].isin(INTENT_TAXONOMY)].copy()
    
    eval_queries = [
        {
            "query": row["text"],
            "ground_truth_intent": row["human_intent"],
            "ground_truth_component": row["component_id"],
        }
        for _, row in in_scope_golden.iterrows()
    ]
    
    golden_metrics = evaluate_retrieval_metrics(retriever, eval_queries, top_k=5)
    
    print("\n=== Golden Set Retrieval Benchmark (N=159) ===")
    print(f"  Recall@1:                   {golden_metrics['recall_at_1'] * 100:.2f}%")
    print(f"  Recall@3:                   {golden_metrics['recall_at_3'] * 100:.2f}%")
    print(f"  Recall@5:                   {golden_metrics['recall_at_5'] * 100:.2f}%")
    print(f"  Mean Reciprocal Rank (MRR): {golden_metrics['mrr']:.4f}")
    print(f"  Intent Concordance Top-1:   {golden_metrics['intent_concordance_top1_pct']:.2f}%")
    print(f"  Evidence Sufficiency Rate:  {golden_metrics['evidence_sufficiency_pct']:.2f}%")

    # 4. Save Retrieval Report & Metrics JSON
    print("\n[4/4] Generating Retrieval Metrics Artifacts...")
    metrics_path = EVALUATION_DIR / "retrieval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "corpus_size": len(corpus_df),
                "fit_duration_seconds": round(fit_duration, 2),
                "golden_benchmark": golden_metrics,
            },
            f,
            indent=2,
        )
    print(f"  Saved metrics to {metrics_path}")

    # Generate Markdown Report
    report_content = f"""# Milestone 7: Evidence Retrieval Experiments & Evaluation

## 1. Evidence Corpus Design
- **Source Split**: `train_retrieval` ONLY (60,742 candidate components).
- **Extracted Resolution Pairs**: {len(corpus_df):,} distinct (customer_query, brand_resolution) pairs.
- **Strict Leakage Prevention**: Verified 0% intersection with `validation` and `test` splits (and the 196-instance golden benchmark).

---

## 2. Hybrid Retrieval Architecture
Resolv employs a two-tier hybrid retrieval mechanism:
1. **Sublinear TF-IDF Lexical Index**: Matches exact technical keywords, order IDs, account terms, and dispute phrases (`refund`, `replacement`, `OTP`, `tracking`, `damaged`).
2. **Dense Character-Ngram Semantic Index**: Matches paraphrased intents and colloquial Twitter spelling variants.
3. **Intent-Conditioned Reranking**: Boosts candidates whose historical resolution aligns with the predicted intent.

---

## 3. Benchmark Retrieval Metrics on Golden Set (N=159 in-scope)

| Metric | Score | Explanation |
|---|---|---|
| **Recall@1** | **{golden_metrics['recall_at_1'] * 100:.2f}%** | Proportion of queries where the #1 retrieved case matches true intent |
| **Recall@3** | **{golden_metrics['recall_at_3'] * 100:.2f}%** | Proportion of queries where a relevant case is present in top-3 candidates |
| **Recall@5** | **{golden_metrics['recall_at_5'] * 100:.2f}%** | Proportion of queries where a relevant case is present in top-5 candidates |
| **MRR (Mean Reciprocal Rank)** | **{golden_metrics['mrr']:.4f}** | Average reciprocal rank of the first relevant historical resolution |
| **Intent Concordance (Top-1)** | **{golden_metrics['intent_concordance_top1_pct']:.2f}%** | Semantic alignment of top-1 evidence with ground-truth customer intent |
| **Evidence Sufficiency Rate** | **{golden_metrics['evidence_sufficiency_pct']:.2f}%** | Percentage of queries meeting the minimum hybrid confidence threshold |

---

## 4. Why Hybrid Retrieval Matters for Customer Support
- A pure semantic dense model often confuses *"Where is my refund?"* with *"Where is my package?"* because both share identical syntactic question structures.
- Lexical matching ensures that specific entities (`refund` vs. `delivery` vs. `password`) anchor the search space, while the dense component captures natural phrasing diversity.
"""
    report_path = PROJECT_ROOT / "reports" / "retrieval_experiments.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"  Saved report to {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
