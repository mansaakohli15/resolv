# Resolv Engineering Decision Log (15 Non-Obvious Decisions)

This document records the 15 primary architectural, modeling, evaluation, and safety decisions made during the design and implementation of **Resolv**.

---

### Decision 1: Selecting AmazonHelp Over Other Candidate Brands
- **Why**: AmazonHelp provided the highest total volume of multi-turn conversations (82,246 components, 371,417 tweets), a 94.2% agent response rate, and structured resolution patterns covering e-commerce transactions, deliveries, and subscriptions.
- **Alternative Considered**: `AppleSupport` (high volume, but dominated by generic OS troubleshooting links) or `Uber_Support` (shorter, less diverse resolution threads).
- **Trade-off**: AmazonHelp Twitter support heavily relies on redirecting users to DMs or self-service pages due to privacy, limiting the depth of in-tweet resolution details.

---

### Decision 2: Conversation Component-Level Splitting (Zero Tweet-Level Leakage)
- **Why**: Tweets within the same conversation thread share common entities, order IDs, timestamps, and customer phrasing. Splitting at the tweet level causes severe data contamination where training tweets predict validation turns.
- **Alternative Considered**: Random tweet-level 80/10/10 split.
- **Trade-off**: Required implementing an explicit connected-component graph reconstruction across `in_response_to_tweet_id` relationships.

---

### Decision 3: Defining an 8-Intent Taxonomy Rather Than Adopting Banking77
- **Why**: The take-home guidelines explicitly asked for *"intents that you define from the data"*. Real Twitter e-commerce support centers on 8 core customer tasks (delivery, returns, refunds, order cancellations, address changes, Prime billing, security, and unauthorized charges).
- **Alternative Considered**: Using polyAI Banking77 (77 banking-specific intents irrelevant to Amazon e-commerce).
- **Trade-off**: Required manual data discovery, clustering, topic inspection, and regex bootstrapping rather than downloading off-the-shelf labeled datasets.

---

### Decision 4: Merging Delivery Delay and Tracking Inquiries into a Single Intent
- **Why**: Empirical cluster analysis in Milestone 3 revealed that *"Where is my package?"* (tracking) and *"My package is 2 days late"* (delay) elicit identical resolution actions from AmazonHelp (check tracking URL, wait 24h, or DM order ID).
- **Alternative Considered**: Maintaining separate `Delivery Delay` and `Tracking Status` categories.
- **Trade-off**: Reduces taxonomy granularity in exchange for higher semantic consistency and cleaner decision boundaries.

---

### Decision 5: Removing Device / App Issues and Prime How-To from v1 Scope
- **Why**: Initial topic modeling suggested `Device / App Issues` (Kindle/FireTV) and `Prime Info / How-To`. Manual verification showed genuine Kindle technical issues represented <0.4% of total volume, while Prime how-to queries were heavily contaminated by delivery complaints mentioning Prime.
- **Alternative Considered**: Retaining 10–12 intents with extreme class sparsity (<50 training instances).
- **Trade-off**: Explicitly narrowing v1 scope to the 8 dominant e-commerce intents and categorizing rare device glitches as `Other / Out of Scope`.

---

### Decision 6: Setting Non-English Messages as Explicitly Out-of-Scope (Tagged, Not Deleted)
- **Why**: AmazonHelp operates international handles with Spanish, Portuguese, French, German, and Japanese queries. Training an English classifier on multilingual data without dedicated translation creates noisy representations.
- **Alternative Considered**: Deleting all non-English tweets or attempting multilingual zero-shot embeddings.
- **Trade-off**: Non-English components were isolated into an `out_of_scope_non_english` pool and used in the golden evaluation set to test the escalation engine's language-routing capability.

---

### Decision 7: Stratified & Oversampled Golden Evaluation Set (196 Instances)
- **Why**: If sampled purely randomly, a 196-example golden set would contain ~140 Delivery Issues and <2 Address/Security cases, rendering rare-intent evaluation statistically meaningless.
- **Alternative Considered**: Uniform random sampling from the test split.
- **Trade-off**: The golden set distribution does not mirror the raw prior distribution, but it provides statistically significant test power across all 8 intents and out-of-scope edge cases.

---

### Decision 8: Isolating the Evidence Corpus Strictly to `train_retrieval` Split
- **Why**: Allowing the retrieval index to search across the entire dataset causes catastrophic evaluation leakage: the retriever finds the exact customer tweet in the golden benchmark and returns 100% artificial accuracy.
- **Alternative Considered**: Indexing all 82,246 conversation components.
- **Trade-off**: Reduces retrieval corpus size from 82,246 to 59,951 components, but guarantees 100% leak-free evaluation integrity.

---

### Decision 9: Hybrid Retrieval (Lexical TF-IDF + Dense Char-Ngrams) Over Pure Dense Vectors
- **Why**: Customer support messages hinge on exact technical entities (`"refund"`, `"OTP"`, `"17-digit order number"`, `"damaged"`) that pure semantic dense embeddings frequently smooth over, while dense char n-grams handle typos and colloquial phrasing.
- **Alternative Considered**: Pure FAISS dense embedding search or pure BM25.
- **Trade-off**: Requires maintaining two feature matrices and tuning hybrid linear combination weights (0.45 lexical + 0.55 semantic).

---

### Decision 10: Intent-Concordance Bonus During Evidence Reranking
- **Why**: When a customer asks about a refund, retrieving a historically similar phrasing about a return replacement produces conflicting guidance. Boosting candidates that match the predicted intent ensures coherent resolution evidence.
- **Alternative Considered**: Pure similarity ranking agnostic of predicted intent.
- **Trade-off**: If the upstream classifier misclassifies the intent, the retrieval reranker inherits part of the error; mitigated by checking raw semantic similarity thresholds.

---

### Decision 11: Multi-Signal Transparent Escalation Policy vs. Naive Probability Threshold
- **Why**: A naive rule like `if confidence < 0.50: escalate` fails to catch high-confidence misclassifications, ignores evidence quality, and misses sensitive account security risks.
- **Alternative Considered**: Single threshold on classifier softmax output.
- **Trade-off**: Requires evaluating 6 distinct risk signals (language, human request, high-risk intent, confidence, margin, evidence sufficiency), producing explainable structured reason codes.

---

### Decision 12: Mandatory Escalation for Account Access and Unauthorized Charges
- **Why**: Issues involving locked passwords, hacked accounts, duplicate credit card deductions, or bank chargebacks require secure identity verification that an automated Twitter bot cannot and should not perform.
- **Alternative Considered**: Attempting to auto-handle all intents.
- **Trade-off**: Lower headline auto-handle rate for security intents, but 100% compliance with real-world enterprise compliance and data protection standards.

---

### Decision 13: 6-Dimension LLM Judge Rubric Scored Out of 30
- **Why**: Simple 1–10 quality ratings from LLMs are notoriously noisy and reward verbose politeness over factual correctness. A 6-dimension rubric (Relevance, Groundedness, Factual Safety, Actionability, Tone, Conciseness) forces fine-grained decomposition.
- **Alternative Considered**: Single-prompt holistic score (e.g. "Rate this reply 1-10").
- **Trade-off**: Requires structured parsing across 6 dimensions, but isolates factual safety from tone.

---

### Decision 14: Empirical Human vs. LLM Judge Validation ($N=40$)
- **Why**: The Hiver assignment explicitly mandates providing *"evidence of how well your judge agrees with a human"*. We conducted an independent double-blind evaluation on 40 stratified golden instances.
- **Alternative Considered**: Claiming the LLM judge is accurate without empirical validation.
- **Trade-off**: Uncovered a modest Pearson correlation ($r = 0.4027$) and MAE of 2.35 points, honestly documenting judge limitations rather than fabricating perfect agreement.

---

### Decision 15: CLI & Modular Architecture Over Disposable Frontend UI
- **Why**: The assignment tests ML evaluation rigor, leakage prevention, failure analysis, and system reproducibility under 15 minutes. Building a React/Streamlit dashboard consumes time without adding ML engineering value.
- **Alternative Considered**: Building a full-stack React / Flask customer support web interface.
- **Trade-off**: The interface is a clean Python API/CLI, but the entire benchmark reproduces in <90 seconds on standard CPU hardware.
