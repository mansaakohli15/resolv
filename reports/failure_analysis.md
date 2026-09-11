# Real Failure Analysis: Top 5 Failure Modes in Resolv

This report presents a forensic failure analysis based on actual errors observed during the evaluation of the 196-instance golden evaluation benchmark. Rather than hypothesizing theoretical failure modes, each category below cites real customer queries, model predictions, observed system behaviors, root causes, and concrete engineering remediations.

---

### Failure Mode 1: Lexical Gravity / Token Over-Attraction (`Delivery Issue` Bias)

- **Golden ID**: `g0015` (Difficulty: Easy)
- **Customer Message**:
  > *"@AmazonHelp Hi, need to change the delivery address for one of my orders. Kindly help."*
- **Ground Truth Intent**: `Address / Delivery Redirect Issue`
- **Model Prediction**: `Delivery Issue` (Confidence: 0.6901)
- **System Decision**: `AUTO_HANDLE`
- **Generated Reply**:
  > *"We're sorry to hear about the delivery delay. Please check your tracking link in 'Your Orders', or send us a DM with your order ID so we can investigate."*
- **Why It Failed**:
  The message contains the tokens `"delivery"` and `"orders"`, which appear tens of thousands of times in the dominant `Delivery Issue` training class. Even though the word `"address"` is present, the classifier assigned higher posterior probability to delivery delay. Because confidence was 0.69 and evidence similarity was 0.61, the escalation gate allowed an auto-reply that gave advice for a delivery delay rather than an address update.
- **Impact**: Customer received an unhelpful response telling them to check tracking instead of explaining how to update an address before dispatch.
- **Remediation Hypothesis**: Implement a priority rule or hierarchical intent classifier where specific intent-modifying entities (`"change address"`, `"redirect"`, `"wrong address"`) take strict precedence over generic delivery tokens.

---

### Failure Mode 2: Missing-Contents & Empty-Box Discrepancy

- **Golden ID**: `g0035` (Difficulty: Medium)
- **Customer Message**:
  > *"Hi @AmazonHelp order number is 171-4216090-2817924 ordered wallet yesterday but there is no wallet in it . Got empty box . See to it asap . This is a fraud happening..."*
- **Ground Truth Intent**: `Return / Wrong or Damaged Item`
- **Model Prediction**: `Delivery Issue` (Confidence: 0.9647)
- **System Decision**: `ESCALATE` (Safely Prevented Auto-Reply)
- **Escalation Reason**: *"Top historical evidence relevance score (0.27) is below minimum evidence threshold (0.28)."*
- **Why It Failed**:
  The customer describes an empty box without using standard keywords like `"return"`, `"replace"`, or `"damaged"`. The bag-of-words and char n-gram models associated `"ordered"`, `"yesterday"`, and `"box"` with delivery transit issues.
- **Impact**: Intent was misclassified, but the multi-signal escalation engine successfully prevented customer harm: because the retrieved evidence was weak ($0.27 < 0.28$), the system suppressed the reply and routed the ticket to a human agent.
- **Remediation Hypothesis**: Expand intent discovery with specialized sub-clusters for theft/empty box reports and train sentence-level semantic representations that capture the negative state *"no [item] in [package]"*.

---

### Failure Mode 3: Latent Prime Subscription vs. Unauthorized Charge Ambiguity

- **Golden ID**: `g0112` (Difficulty: Medium)
- **Customer Message**:
  > *"Why was my credit card charged $99 out of nowhere this morning when I haven't ordered anything in months?"*
- **Ground Truth Intent**: `Prime Membership Billing/Cancellation`
- **Model Prediction**: `Unauthorized / Non-Prime Charge` (Confidence: 0.7420)
- **System Decision**: `ESCALATE`
- **Escalation Reason**: *"Intent 'Unauthorized / Non-Prime Charge' involves account security or financial transactions requiring identity verification."*
- **Why It Failed**:
  The customer does not mention the word `"Prime"`. To a human expert, an unexpected annual fee of exactly $99/$119 on Amazon is almost certainly an automatic Prime renewal. However, strictly based on text features, the query matches unprompted bank deductions (`"charged out of nowhere"`, `"haven't ordered anything"`).
- **Impact**: Both intents are categorized as `HIGH_RISK_INTENTS` in our policy, so the escalation decision was 100% correct (human escalation), but intent classification accuracy suffered.
- **Remediation Hypothesis**: Incorporate numeric entity extraction for known subscription amounts ($12.99, $99.00, $119.00, $139.00) to conditionally elevate the prior probability of `Prime Membership Billing/Cancellation`.

---

### Failure Mode 4: Non-Romance Script & Multilingual Out-of-Scope Detection

- **Golden ID**: `g0043` (Difficulty: Medium)
- **Customer Message**:
  > *"Amazonで頼んだ荷物がまだ届かないんだけど、どうなってるの？確認してください。"* (Japanese: "The package I ordered from Amazon hasn't arrived yet, what's going on? Please check.")
- **Ground Truth Intent**: `Unsupported Language`
- **Model Prediction**: `Delivery Issue` (Confidence: 0.5810)
- **System Decision**: `ESCALATE`
- **Escalation Reason**: *"Unsupported language detected; requires specialized language support agent."*
- **Why It Failed**:
  The split-time heuristic only searched for common Romance language words (`hola`, `merci`, `pedido`). Non-Latin scripts (CJK, Arabic, Cyrillic) bypassed the split heuristic and entered the English modeling pool.
- **Impact**: While the base classifier attempted to force-fit the query into `Delivery Issue`, our character-level Unicode policy detected CJK characters (`[\u3040-\u30ff]`) and forced immediate escalation.
- **Remediation Hypothesis**: Replace regex-based language heuristics with a fast C-based language detection library (e.g., `pycld2` or `fasttext-langdetect`) upstream of all intent classification.

---

### Failure Mode 5: Sarcastic & Elliptical Product Complaints

- **Golden ID**: `g0044` (Difficulty: Medium)
- **Customer Message**:
  > *"Note to self: don't trust Your Garage. Bf bought a trailer hitch for my van and they said it would fit it. NOPE. Not even close."*
- **Ground Truth Intent**: `Return / Wrong or Damaged Item`
- **Model Prediction**: `Delivery Issue` (Confidence: 0.9864)
- **System Decision**: `ESCALATE`
- **Escalation Reason**: *"Top historical evidence relevance score (0.25) is below minimum evidence threshold (0.28)."*
- **Why It Failed**:
  The tweet is written as a sarcastic narrative / social commentary (`"Note to self"`, `"NOPE. Not even close"`) rather than a direct support request. The classifier failed to identify this as a compatibility / wrong-item issue and defaulted to the dominant training prior.
- **Impact**: Sarcastic tweets with low explicit intent markers can confuse linear classifiers. However, because historical resolution retrieval found no similar conversational pairs ($0.25 < 0.28$), the system avoided sending a misplaced delivery delay template.
- **Remediation Hypothesis**: Integrate sentiment and conversational discourse modeling to differentiate pure social venting from actionable customer service requests.
