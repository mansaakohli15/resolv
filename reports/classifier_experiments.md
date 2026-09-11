# Milestone 6: Intent Classifier Experiments & Baseline Analysis

## 1. Overview & Setup
This milestone benchmarks the intent classification capability of Resolv on both the component-held-out validation split and the independent 196-instance golden evaluation benchmark.

- **Training Corpus**: 19,979 conversation starting messages drawn strictly from the `train_retrieval` split (zero cross-split contamination).
- **Validation Split**: 2,497 held-out conversation components.
- **Golden Evaluation Benchmark**: 196 hand-annotated examples (159 in-scope across 8 intents, 37 out-of-scope/ambiguous/non-English).

---

## 2. Models Evaluated

1. **Baseline 1 — Majority Class Classifier**:
   - Trivial heuristic predicting the dominant training category (`Delivery Issue`).
   - Represents the empirical floor for precision, recall, and Macro-F1.

2. **Baseline 2 — Standard TF-IDF + Logistic Regression**:
   - Word unigrams (max 5,000 features), default L2 regularization ($C=1.0$), unweighted class distribution.

3. **Main Model — Resolv Calibrated Feature Union**:
   - **Feature Representation**: Union of Sublinear Word n-grams (1–3, 15,000 features) and Character n-grams (3–5, 25,000 features).
   - **Class Weighting**: Balanced class penalty to counteract dominant delivery complaints and protect rare intents (e.g., `Address / Delivery Redirect`, `Unauthorized / Non-Prime Charge`).
   - **Probability Calibration**: 5-fold cross-validated Sigmoid calibration (Platt scaling) producing trustworthy probability distributions for downstream escalation gating.

---

## 3. Benchmark Results

### A. Performance on Validation Split (N=2,497)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **Baseline 1 (Majority Class)** | 70.80% | 0.1036 | 0.5870 | 0.0885 | 0.1250 |
| **Baseline 2 (Simple TF-IDF)** | 96.08% | 0.8270 | 0.9581 | 0.8522 | 0.8078 |
| **Main Resolv Model** | **98.12%** | **0.9253** | **0.9810** | **0.9189** | **0.9324** |

### B. Performance on Golden Benchmark (In-Scope N=159)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|---|
| **Baseline 1 (Majority Class)** | 22.29% | 0.0456 | 0.0813 | 0.0279 | 0.1250 |
| **Baseline 2 (Simple TF-IDF)** | 71.34% | 0.7109 | 0.7152 | 0.7381 | 0.7134 |
| **Main Resolv Model** | **79.62%** | **0.7960** | **0.7959** | **0.8357** | **0.7962** |

---

## 4. Main Model Per-Intent Breakdown (Golden Benchmark)

| Intent | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Account Access / Security** | 93.75% | 93.75% | **0.9375** | 16 |
| **Refund Request** | 88.89% | 84.21% | **0.8649** | 19 |
| **Return / Wrong or Damaged Item** | 100.00% | 76.19% | **0.8649** | 21 |
| **Order Cancellation Request** | 85.71% | 80.00% | **0.8276** | 15 |
| **Delivery Issue** | 74.42% | 91.43% | **0.8205** | 35 |
| **Address / Delivery Redirect Issue** | 91.67% | 73.33% | **0.8148** | 15 |
| **Unauthorized / Non-Prime Charge** | 54.17% | 86.67% | **0.6667** | 15 |
| **Prime Membership Billing/Cancellation** | 71.43% | 47.62% | **0.5714** | 21 |

---

## 5. Performance by Difficulty Tier

| Difficulty Tier | Golden Examples | Accuracy | Macro F1 |
|---|---|---|---|
| **Easy** | 85 | **85.88%** | **0.8684** |
| **Medium** | 61 | **78.69%** | **0.7815** |
| **Hard** | 11 | **36.36%** | **0.2476** |

---

## 6. Key Findings & Error Patterns

1. **Why Word+Char Feature Union Outperforms Simple TF-IDF**:
   - Captures Twitter-specific noise, slang, typos (e.g., `passwrd`, `log-in`, `cancell`, `cant log in`), and token variations that word unigrams completely miss.
   - Boosts in-scope golden accuracy by **+8.28%** and Macro F1 by **+0.0851** over Baseline 2.

2. **The Charge vs. Prime Ambiguity Boundary**:
   - The primary error cluster occurs between `Unauthorized / Non-Prime Charge` and `Prime Membership Billing/Cancellation`. When a customer tweets *"Why was I charged $99 without authorization?"*, the model sometimes misses that the $99 fee corresponds to an annual Prime renewal unless "Prime" is explicitly stated.
   - This directly validates the design decision in our escalation policy: financial charge disputes must have high confidence thresholds and route to human review when ambiguous.

3. **Difficulty Degradation**:
   - The system maintains high accuracy on Easy (85.88%) and Medium (78.69%) queries, but drops on Hard (36.36%) queries which involve multi-turn background context or sarcastic/elliptical phrasing.
   - This drop on Hard cases underscores why Resolv needs an escalation engine rather than attempting to blindly auto-reply to all tweets.
