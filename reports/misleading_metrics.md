# Mandatory Report Section: What is Misleading About My Headline Number?

In customer support machine learning systems, high headline metrics frequently disguise catastrophic real-world operational failure modes. Below is an honest breakdown of why our headline metrics (e.g., **98.12% Validation Accuracy**, **79.62% Golden Intent Accuracy**, and **57.14% Auto-Handle Rate**) can create a false sense of safety if taken at face value.

---

### 1. The Validation Split Illusion (98.12% vs. 79.62%)
- On the component-held-out validation split, our model achieves an apparently stellar **98.12% Accuracy** and **0.9253 Macro F1**.
- However, on the independent 196-row hand-annotated golden set, accuracy drops to **79.62%** and Macro F1 drops to **0.7960**.
- **Why this happens**: The validation split labels were derived using weak heuristic regex rules from the discovery phase. The classifier learned to predict the heuristic's labeling patterns with high fidelity. But when tested on genuine, messy human language where real humans labeled difficult edge cases, the true performance is ~18.5% lower. Citing only the validation split would be deeply misleading.

---

### 2. The Dominant Class Gravity & Macro vs. Weighted F1
- In the raw training data, `Delivery Issue` represents **>70%** of all in-scope customer interactions.
- A trivial baseline that does nothing except predict `Delivery Issue` achieves **70.80% Accuracy** on the validation set, despite having a **Macro F1 of only 0.1036**.
- While our balanced feature-union model achieves **0.7960 Macro F1**, performance is uneven across intents:
  - High-volume intents (`Account Access / Security`: 0.9375 F1, `Refund Request`: 0.8649 F1) perform well.
  - Rare, high-ambiguity intents (`Prime Membership Billing`: 0.5714 F1, `Unauthorized Charge`: 0.6667 F1) suffer significantly.
- An aggregate accuracy of ~80% conceals that 1 out of every 2 Prime billing disputes is misclassified.

---

### 3. The Danger of the "57.14% Auto-Handle Rate"
- It is tempting to report to management: *"The system can safely automate 57.14% of incoming customer volume."*
- This is misleading for three reasons:
  1. **Golden Set Stratification Skew**: Our golden evaluation set deliberately oversampled rare intents and out-of-scope/unsupported language queries (37/196 = 18.9% out-of-scope). In a production stream, the true proportion of easy delivery delays is even higher, which might artificially inflate the raw auto-handle rate while masking precision drops.
  2. **False Auto-Handle Risk**: As shown in Failure Mode 1 (`g0015`), when a customer asked to change their address, the model auto-handled the query with a delivery delay tracking link. An incorrect auto-handle is substantially worse than an escalation: it wastes customer time and erodes brand trust.
  3. **Multi-Turn Context Blindness**: Resolv operates on the first inbound customer tweet. Real customer support conversations frequently span multiple turns where the initial tweet is merely an opening greeting (*"Hi, is anyone there?"*). Treating single-turn success as full conversational automation is a classic production fallacy.

---

### 4. LLM-as-a-Judge Agreement Discrepancies ($r = 0.4027$)
- Our automated LLM judge awarded auto-handled replies an average score of **25.36 / 30.0 (84.5%)**, with a **100% Factual Safety Pass Rate**.
- However, our empirical validation against 40 human-reviewed cases revealed:
  - Exact Agreement: **5.00%**
  - Agreement within $\pm 1$ point: **35.00%**
  - Agreement within $\pm 2$ points: **57.50%**
  - Pearson Correlation: **$r = 0.4027$**
  - Mean Absolute Error: **2.35 points / 30**
- The automated judge systematically scores polite, well-formatted brand redirects (*"Please send us a DM with your order ID"*) with maximum points (5/5), whereas human annotators penalize generic redirects when the customer asked a concrete question that could have been answered directly. Claiming an LLM judge is an infallible proxy for human satisfaction is inaccurate.

---

### 5. Summary Table: Headline Number vs. Reality

| Headline Metric | Surface Interpretation | Unvarnished Reality |
|---|---|---|
| **98.12% Validation Accuracy** | "Near-perfect customer intent detection" | Evaluated against weak heuristic labels; drops to 79.62% on human benchmark |
| **0.7960 Golden Macro F1** | "Robust performance across all intents" | Hides 0.5714 F1 on Prime Billing and frequent charge disputes |
| **57.14% Auto-Handle Rate** | "Over half of support volume automated" | Fails to capture multi-turn conversational complexity and subtle misdirection errors |
| **84.5% LLM Judge Score** | "Replies are 85% optimal" | Judge over-rewards generic brand politeness ($r=0.4027$ with humans) |
| **100% Factual Safety** | "Completely hallucination-free" | True for grounded templates, but reply utility is limited by historical redirect brevity |
