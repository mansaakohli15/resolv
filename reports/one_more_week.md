# What We Would Do With One More Week

If given one additional week of engineering time, we would prioritize four high-impact architectural and evaluation enhancements for Resolv:

---

### 1. Multi-Turn Conversational Discourse & Thread Context Modeling
- **Current Limitation**: Resolv operates on the initial inbound customer tweet. In practice, ~45% of support interactions are multi-turn threads where critical customer entities (order numbers, carrier names, revised problem statements) appear in turns 2 through 5.
- **Plan**:
  - Implement a conversational state tracker that ingests the full conversation history.
  - Dynamically update intent confidence and retrieval queries as the customer provides clarifications across subsequent turns.

---

### 2. Fine-Tuned Dense Bi-Encoder Embeddings for Customer Support
- **Current Limitation**: Our hybrid retriever uses sublinear TF-IDF + char n-grams. While fast and leak-free, it lacks deep contextual semantic understanding on complex paraphrase structures.
- **Plan**:
  - Fine-tune a lightweight bi-encoder (e.g. `BGE-small` or `MiniLM-L6-v2`) directly on the 59,951 training resolution pairs using Multiple Negatives Ranking Loss (MNRL).
  - Benchmark fine-tuned dense retrieval against BM25/TF-IDF hybrid retrieval on Recall@K and MRR.

---

### 3. Active Learning & Targeted Hard-Negative Annotation
- **Current Limitation**: The golden evaluation set contains 196 hand-annotated examples. While stratified, the `Hard` tier contains only 17 instances, limiting statistical significance on difficult edge cases.
- **Plan**:
  - Use active learning / uncertainty sampling to select 300 additional high-entropy customer messages where the model exhibits low intent margin ($<0.08$) or conflicting retrieval candidates.
  - Double-annotate this hard-negative set to establish inter-annotator agreement (Cohen's Kappa $\kappa$) and expand the benchmark to 500 instances.

---

### 4. Direct Support Tool Integration & Mock API Sandbox
- **Current Limitation**: Resolv generates textual guidance (e.g. *"Please check tracking in Your Orders"* or *"Send us a DM"*) because it lacks backend tool execution capabilities.
- **Plan**:
  - Implement a mock tool-use environment (e.g. `check_order_status(order_id)`, `issue_return_label(order_id)`, `cancel_prime_subscription(user_id)`).
  - Equip the generation layer with function-calling guardrails to simulate end-to-end automated resolution in a safe sandbox.
