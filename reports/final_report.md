# Resolv: Evidence-Grounded Customer Support Automation
### Technical Report — Hiver SDE Intern Take-Home Assignment
**Author**: Mansaa Kohli | **Brand**: AmazonHelp | **Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)

---

## 1. Problem Framing: What "Good" Means for AmazonHelp & What We Chose NOT to Build

Automating customer support on Twitter is not a generic text generation problem. Twitter support messages are noisy, unstructured, emotionally charged, and frequently lack critical account context.

For **AmazonHelp**, a "good" automated support system must satisfy three non-negotiable criteria:
1. **Strict Factual Grounding**: The system must never hallucinate refund approvals, fake delivery promises, non-existent policies, or claim actions have been completed.
2. **Transparent Operational Gating**: When a customer's issue requires identity verification (e.g. password resets, unrecognized bank charges) or when historical precedent is ambiguous, the system must **escalate to a human agent with an explicit reason** rather than guessing.
3. **Evidence-Backed Justification**: Every automated suggestion must cite historical resolution cases that justify *why* that response is appropriate.

### What We Chose NOT to Build:
- **No Unconstrained Chatbot**: We rejected unconstrained end-to-end LLM chat (`tweet -> LLM -> reply`). Unconstrained LLMs hallucinate policies and cannot be calibrated for enterprise compliance.
- **No Disposable Frontend UI**: We prioritized rigorous evaluation harnesses, leakage-safe splits, and statistical judge validation over flashy dashboards.
- **No Complex Cloud Dependencies**: Resolv runs locally with 100% deterministic reproducibility in $<90$ seconds.

---

## 2. Dataset Profiling, Taxonomy & Leakage-Safe Splitting

### A. Brand Selection (AmazonHelp)
Profiling 10 candidate brands in the 2.8M-tweet TWCS dataset established **AmazonHelp** as the optimal candidate:
- **82,246 unique conversation components**
- **371,417 tweets** (196,476 customer turns, 169,209 brand replies)
- **94.2% brand response rate** across multi-turn threads.

### B. Final 8-Intent Taxonomy (Derived from Real Conversations)
1. `Delivery Issue` (delay, non-receipt, tracking)
2. `Address / Delivery Redirect Issue` (change delivery destination)
3. `Refund Request` (refund status, dispute)
4. `Return / Wrong or Damaged Item` (replacement, damaged item, defective product)
5. `Prime Membership Billing/Cancellation` (Prime subscription fees, auto-renewal)
6. `Order Cancellation Request` (cancel order before/after dispatch)
7. `Account Access / Security` (password lockout, hacked account)
8. `Unauthorized / Non-Prime Charge` (unrecognized card charge)
*Out-of-scope categories*: `Other / Out of Scope`, `Unsupported Language`, `Ambiguous`.

### C. Component-Level Splitting & Zero-Leakage Guarantee
Splits were constructed at the **conversation component level** (80% train_retrieval / 10% validation / 10% test). All retrieval evidence is extracted **STRICTLY from train_retrieval** (59,951 pairs), with **0% overlap** with validation or the golden test set.

### D. Golden Evaluation Set (196 Hand-Annotated Instances)
Sampled from the test split + non-English pool, stratified across all 8 intents and 3 difficulty tiers (105 Easy, 74 Medium, 17 Hard). Every row was independently annotated with ground-truth intent, difficulty, and annotator notes.

---

## 3. System Architecture & Baseline Comparisons

Resolv implements a modular, 5-stage inference pipeline:

$$\text{Customer Tweet} \longrightarrow \text{Intent Classifier} \longrightarrow \text{Hybrid Retrieval} \longrightarrow \text{Risk Escalation Engine} \longrightarrow \text{Grounded Reply}$$

```
                ┌── Intent Classifier (Word+Char Calibrated Logistic Model)
                │
Customer Tweet ─┼── Hybrid Retrieval (BM25 Lexical + Dense Char-Ngram Search)
                │
                ├── Risk Escalation Policy (6 Safety Signals -> AUTO_HANDLE / ESCALATE)
                │
                └── Grounded Reply Generator (Evidence-Anchored Synthesis)
```

### Baseline Models vs. Main Classifier:
- **Baseline 1 (Majority Class)**: Predicts dominant class (`Delivery Issue`).
- **Baseline 2 (Simple TF-IDF + LogReg)**: Unigram TF-IDF baseline without class weighting.
- **Main Model (Resolv Calibrated Feature Union)**: Sublinear Word (1–3 n-grams) + Char (3–5 n-grams) + Class-Weighted Calibrated Logistic Regression.

---

## 4. Empirical Benchmark Results

### A. Intent Classification Benchmark (Golden Set In-Scope N=159)

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Prec | Macro Recall |
|---|---|---|---|---|---|
| **Baseline 1 (Majority Class)** | 22.29% | 0.0456 | 0.0813 | 0.0279 | 0.1250 |
| **Baseline 2 (Simple TF-IDF)** | 71.34% | 0.7109 | 0.7152 | 0.7381 | 0.7134 |
| **Main Resolv Model** | **79.62%** | **0.7960** | **0.7959** | **0.8357** | **0.7962** |

*Difficulty Breakdown*: **Easy (85.88% Acc)** | **Medium (78.69% Acc)** | **Hard (36.36% Acc)**.

### B. Evidence Retrieval Benchmark (Corpus N=59,951 train-only pairs)

| Metric | Score | Operational Significance |
|---|---|---|
| **Recall@1** | **64.97%** | #1 retrieved case matches ground-truth intent |
| **Recall@3** | **87.90%** | Relevant resolution present in top-3 candidates |
| **Recall@5** | **91.72%** | Relevant resolution present in top-5 candidates |
| **MRR** | **0.7567** | High reciprocal ranking density of relevant evidence |
| **Sufficiency Rate** | **88.54%** | Queries meeting minimum evidence confidence threshold |

### C. Escalation & Auto-Handling Gating (Full Golden Set N=196)
- **Auto-Handle Rate**: **57.14%** (112/196 queries safely automated).
- **Escalation Rate**: **42.86%** (84/196 queries routed to human specialists).
- **Escalation Safety Recall**: **74.03%** (Catches 74%+ of out-of-scope, security, and high-difficulty issues).
- **Factual Safety Pass Rate**: **100.00%** (Zero hallucinated refunds, dates, or completed actions).

---

## 5. LLM-as-a-Judge Rubric & Human Agreement Validation

Automated replies were evaluated using a strict **6-dimension rubric** (scored 1 to 5, total out of 30 points):
- **Resolution Relevance**: **4.31 / 5.0**
- **Evidence Groundedness**: **3.00 / 5.0**
- **Factual Safety**: **5.00 / 5.0** (100% safe)
- **Actionability**: **3.41 / 5.0**
- **Tone & Empathy**: **4.71 / 5.0**
- **Conciseness**: **4.92 / 5.0**
- **Mean Total Score**: **25.36 / 30.0 (84.5%)**

### Empirical Human vs. LLM Judge Validation ($N=40$ Stratified Instances)
We validated the automated judge against independent double-blind human ratings:
- **Agreement within $\pm 1$ point**: **35.00%**
- **Agreement within $\pm 2$ points**: **57.50%**
- **Mean Absolute Error (MAE)**: **2.35 points / 30**
- **Pearson Correlation ($r$)**: **0.4027**
- **Spearman Correlation ($\rho$)**: **0.3811**

*Insight*: The automated judge over-rewards polite brand redirection phrases, while human reviewers penalize generic redirects when the customer asked a specific question. Documenting this discrepancy honestly is vital for production transparency.

---

## 6. Real Failure Analysis: Top 5 Failure Modes

1. **Dominant Delivery Token Gravity (`g0015`)**: Address update request was misclassified as `Delivery Issue` due to presence of `"delivery"` and `"order"` keywords, sending an unhelpful delivery tracking link.
2. **Missing-Contents & Empty Box Phrasing (`g0035`)**: Customer received an empty box without using the word `"damaged"`. Misclassified as delivery delay, but safely escalated due to low retrieval score ($0.27 < 0.28$).
3. **Latent Prime Fee vs. Card Charge Confusion (`g0112`)**: Customer complained about an unprompted $99 deduction without mentioning the word "Prime". Both intents correctly triggered mandatory human escalation.
4. **Non-Romance Script Misses (`g0043`)**: Japanese customer tweet bypassed regex split heuristics; caught downstream by Unicode character detection in the escalation policy.
5. **Sarcastic Narrative Venting (`g0044`)**: Sarcastic complaint about trailer hitch compatibility lacked formal support verbs; low retrieval score prevented sending irrelevant templates.

---

## 7. Mandatory Section: What is Misleading About My Headline Number?

1. **The Validation Split Illusion (98.12% vs. 79.62%)**: Citing the 98.12% validation accuracy would be deceptive because validation labels were derived from regex heuristics. True accuracy on messy human-annotated data is 79.62%.
2. **The Dominant Class Bias**: In-scope accuracy (79.62%) hides significant variance across intents: high-volume intents (`Account Access`: 0.9375 F1) perform well, while rare, ambiguous intents (`Prime Billing`: 0.5714 F1) fail 1 in 2 times.
3. **The Danger of the 57.14% Auto-Handle Rate**: Single-turn success does not guarantee multi-turn resolution. If a customer is misdirected on turn 1, customer frustration escalates rapidly.
4. **Judge Calibration Limits**: The 84.5% judge quality score reflects prompt compliance and politeness, but exhibits only moderate correlation ($r=0.4027$) with human expert judgment.

---

## 8. What We Would Do With One More Week
1. **Multi-Turn Conversational Discourse**: Track context across turns 2–5 where customers provide revised details.
2. **Fine-Tuned Bi-Encoder Dense Retrieval**: Fine-tune a lightweight transformer (`BGE-small` / `MiniLM`) using Multiple Negatives Ranking Loss on the 59k training pairs.
3. **Hard-Negative Active Learning**: Expand the golden benchmark to 500 instances with double-annotated edge cases.
4. **Tool-Use Execution Sandbox**: Equip generation with structured function calling (order status, return label generation) in a mock API sandbox.

---

## 9. 15-Point Decision Log Summary

1. **AmazonHelp Selected**: Highest multi-turn volume (82k components) and 94.2% agent response rate.
2. **Component-Level Split**: Guaranteed 0% cross-turn contamination between training and test sets.
3. **8 Data-Derived Intents**: Formulated from empirical cluster inspection rather than arbitrary generic taxonomies.
4. **Merged Delivery/Tracking**: Unified identical resolution workflows into a robust single intent.
5. **Pruned Sparse Intents**: Removed Kindle device issues (<0.4% volume) to maintain statistical reliability.
6. **Isolated Non-English Pool**: Preserved multilingual queries in a separate out-of-scope evaluation pool.
7. **Stratified Golden Set (196 instances)**: Oversampled rare intents to prevent delivery dominance in testing.
8. **Train-Only Evidence Corpus (59,951 pairs)**: Strict leakage prevention from validation/test/golden sets.
9. **Hybrid TF-IDF + Dense Search**: Combined exact keyword matching with dense char-ngram morphology.
10. **Intent Concordance Reranking**: Boosted evidence matching the predicted intent for resolution coherence.
11. **Multi-Signal Escalation Engine**: Evaluated 6 orthogonal risk signals rather than a naive confidence score.
12. **Mandatory Security Escalation**: Guaranteed human verification for account access and card charges.
13. **6-Dimension Rubric**: Separated factual safety and actionability from surface-level tone.
14. **Empirical Judge Validation**: Formally measured human vs. LLM judge agreement ($r=0.4027$, MAE=2.35).
15. **CLI & Reproducibility Over UI**: Prioritized ML evaluation rigor and $<90$s reproducible benchmarks.
