# Milestone 7: Evidence Retrieval Experiments & Evaluation

## 1. Evidence Corpus Design
- **Source Split**: `train_retrieval` ONLY (60,742 candidate components).
- **Extracted Resolution Pairs**: 59,951 distinct (customer_query, brand_resolution) pairs.
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
| **Recall@1** | **64.97%** | Proportion of queries where the #1 retrieved case matches true intent |
| **Recall@3** | **87.90%** | Proportion of queries where a relevant case is present in top-3 candidates |
| **Recall@5** | **91.72%** | Proportion of queries where a relevant case is present in top-5 candidates |
| **MRR (Mean Reciprocal Rank)** | **0.7567** | Average reciprocal rank of the first relevant historical resolution |
| **Intent Concordance (Top-1)** | **64.97%** | Semantic alignment of top-1 evidence with ground-truth customer intent |
| **Evidence Sufficiency Rate** | **88.54%** | Percentage of queries meeting the minimum hybrid confidence threshold |

---

## 4. Why Hybrid Retrieval Matters for Customer Support
- A pure semantic dense model often confuses *"Where is my refund?"* with *"Where is my package?"* because both share identical syntactic question structures.
- Lexical matching ensures that specific entities (`refund` vs. `delivery` vs. `password`) anchor the search space, while the dense component captures natural phrasing diversity.
