# Intent Discovery — AmazonHelp (Milestone 3)

## Pipeline / reproducibility
```
python scripts/build_brand_dataset.py       # -> data/processed/amazonhelp_conversations.pkl
python scripts/discover_topics.py           # -> evaluation/topic_terms.csv, evaluation/topic_clusters.csv
python scripts/inspect_topic_examples.py > reports/topic_examples_raw.txt
```

## Step 1 — AmazonHelp conversation dataset
82,246 conversations (371,417 rows: 202,176 customer + 169,241 AmazonHelp), built from the
82,556 components where AmazonHelp replied, after dropping 310 cross-brand components (R3).
See docstring in `scripts/build_brand_dataset.py` for the full, numbered filtering rules —
nothing was dropped without a documented reason, and every row keeps its original `tweet_id`
and `_component` for traceability.

## Step 2 — Topic discovery: clustering did not work, frequency analysis did
- First-message word count: median 19 words, 10th pct 4, 95th pct 41 — mostly short but not
  trivially short; enough content to work with.
- **MiniBatchKMeans (TF-IDF, k=14) on the 82,246 first-customer-messages collapsed into two
  mega-clusters (54% and 46% of the data) plus a long tail of singleton clusters** — not usable
  topical structure (`evaluation/topic_clusters.csv`). This mirrors the inconclusive clustering
  signal seen in Milestone 2's brand diversity check. I'm not building the taxonomy from these
  clusters — reporting the attempt and the negative result rather than hiding it.
- Fell back to **keyword groups grounded in the actual top unigram/bigram frequency table**
  (`evaluation/topic_terms.csv`) and manually read real examples per group
  (`reports/topic_examples_raw.txt`, 11 groups × 6 sampled conversations each = 66 read).

## Step 3 — What the real examples show
- **The single biggest, clearest pattern:** a huge share of first messages are "marked/expected
  delivered but not received" or "delivery attempt claimed but never happened" — not vague
  shipping questions, specific non-delivery complaints. `delivery_timing` keyword group alone
  matched 22.74% of first messages.
- **AmazonHelp's replies are overwhelmingly redirects, not resolutions.** The large majority of
  sampled replies across every group are variants of "sorry to hear that, please contact us
  here: [link]" or "check your tracking here: [link]" — the actual resolution happens off-
  platform, outside the dataset. This is a real constraint on what "grounded reply generation"
  can mean for Resolv: for most historical cases, the "resolution" we can retrieve is a routing
  instruction, not a specific fix. Worth flagging now rather than discovering it during reply
  evaluation later.
- **Messages frequently mix issues in one tweet** (e.g. component 334544: no delivery AND a
  delayed refund in one message; component 1048441: cancellation AND missing refund together).
  Intent classification will need a policy for multi-intent messages (primary intent vs. multi-
  label) — not decided here.
- **The same underlying issue gets caught by multiple keyword groups** (a "still haven't
  received it" complaint matches `delivery_timing`, `order_tracking`, and sometimes
  `customer_service_complaint` simultaneously) — confirms these are not going to be cleanly
  mutually exclusive without an explicit precedence rule.
- **Real, measured multilingual content**, not just a `device_app_issue` artifact — Portuguese,
  French, and Spanish customer messages and AmazonHelp replies appeared in the samples (e.g.
  components 1142092, 2761885, 1067296). A rough heuristic marker earlier estimated ~6.8% of
  first messages as non-English; this is an approximation, not a validated language-ID count,
  but the phenomenon itself is confirmed by direct reading, not just the heuristic.
- `customer_service_complaint` messages (510 exact matches, 0.62%) mostly restate an
  underlying issue (bad delivery, bad support experience) with generic anger and no new
  actionable detail — this looks more like a **tone/escalation signal riding on top of another
  intent** than a standalone resolvable intent.

## Step 4 — Candidate intent taxonomy (NOT final — for review)

| # | Intent | Definition | Approx. examples | Confusable with | Reliably distinguishable? |
|---|---|---|---|---|---|
| 1 | **Delivery Delay / Non-Delivery** | Package marked or expected delivered but customer hasn't received it, or a delivery attempt is disputed | ~18,700 (keyword-matched, likely includes some overlap with #2) | Order Tracking, General Complaint | Yes — strong keyword/phrase signal ("haven't received", "marked delivered") |
| 2 | **Order Tracking / Status Inquiry** | Neutral request for current tracking/carrier status, not yet an escalated complaint | ~6,800 | Delivery Delay | Partially — the complaint-vs-neutral-inquiry line is fuzzy and needs a labeling rule, not just keywords |
| 3 | **Refund Request** | Explicit ask for money back (undelivered item, duplicate charge, unwanted purchase) | ~3,960 | Billing Dispute, Return | Yes, when "refund" is stated explicitly |
| 4 | **Return / Wrong or Damaged Item** | Item received is wrong, damaged, or defective; wants replacement/return | ~1,170 (narrow match — likely undercounted) | Refund Request | Yes, but often co-occurs with #3 in the same message |
| 5 | **Account Access / Security** | Login failure, password reset, or unauthorized account/email changes | ~3,110 | Billing Dispute (when compromise causes a charge) | Mostly yes |
| 6 | **Unauthorized/Unexpected Charge** | Charged for something not ordered or authorized, double-charged, or payment deducted despite a failed transaction | ~1,750 (includes some Prime-billing overlap, not cleanly separated yet) | Prime Billing, Account Access | Needs a precedence rule vs. #7 |
| 7 | **Prime Membership — Billing/Cancellation** | Unwanted Prime charge, wants to cancel/understand Prime billing | Sub-portion of ~2,640 `prime_membership` matches — not separately counted yet | Unauthorized Charge, Prime Info | No — currently tangled with #8 in the same keyword group, needs manual re-split |
| 8 | **Prime/Product Info or How-To Question** | Non-complaint informational question (does Prime Video cost extra, can Alexa do X) | Sub-portion of `prime_membership` + `device_app_issue` — not separately counted yet | Prime Billing, Device Issue | No — same issue as #7 |
| 9 | **Order Cancellation Request** | Explicit request to cancel an order (not Prime) before/after placing it | ~238 exact-phrase matches (likely undercounts broader "cancel" phrasing) | Refund Request | Yes when phrased explicitly, but volume needs re-measuring with a broader pattern |
| 10 | **Device / App Technical Issue** | App or device (Kindle, Alexa, Fire TV) not working or malfunctioning, distinct from a billing/how-to question | Sub-portion of ~2,380 `device_app_issue` matches | Prime Info | Partially — same keyword group mixes technical faults with capability questions |
| 11 | **Address / Delivery Redirect Issue** | Package sent to wrong/inaccessible address, needs redirect or pickup change | Not yet separately measured — seen in manual reading (e.g. component 1217073), no dedicated keyword group run yet | Delivery Delay | Unknown — needs its own measurement pass |

**Not proposed as a standalone intent:** a generic "Customer Service Complaint" / vague
dissatisfaction category. The 510 exact matches read as anger layered on top of one of the
intents above rather than a distinct actionable category — treating it as a tone/risk signal
(useful for the later escalation component) rather than an intent seems better supported by
what was actually read, but this is a judgment call worth you weighing in on.

**Not proposed yet, flagged as an open scope question:** multilingual (non-English) messages
are real and material (~6.8%, approximate) but cut across every intent above rather than being
one themselves — needs a decision (exclude from v1 scope vs. handle multilingually) before
taxonomy finalization, not made here.

## Main ambiguities / problems discovered
1. Intents #1/#2 and #6/#7/#8 are not cleanly separated by the keyword groups used — they need
   either a manual re-split with tighter patterns or to be merged; presenting them separately
   here so you can see the raw evidence rather than pre-deciding a merge.
2. Multi-intent messages are common (single tweet stating two problems) — taxonomy needs a
   stated policy (primary-intent-only vs. multi-label) before classifier work starts.
3. A large share of "resolutions" in the historical data are redirects to another channel, not
   actual fixes — this bounds what evidence-grounded reply generation can honestly claim later.
4. Approximate counts for intents #7, #8, #10, #11 are sub-portions of broader keyword groups
   or not yet separately measured — marked as such above, not presented as final numbers.
5. Order Cancellation (#9)'s exact-phrase count (238) is almost certainly an undercount given
   how narrow the regex was — needs a broader pass before trusting the volume for golden-set
   sampling feasibility.

## Recommendation for next milestone (superseded — see Re-pass below)
Before finalizing the taxonomy: a short, targeted re-pass that (a) splits the tangled Prime/
billing/info group with tighter rules, (b) re-measures Order Cancellation and Address/Redirect
with broader patterns, and (c) makes an explicit call on the vague-complaint and multilingual
questions above. That re-pass is smaller than a full Milestone 4 and would let the taxonomy
lock in on solid rather than approximate counts before any labeling work begins.

---

# Re-pass: targeted follow-up analysis

Script: `scripts/repass_intent_analysis.py` (reuses `data/processed/amazonhelp_conversations.pkl`
only, no raw CSV reload). Full output: `reports/repass_raw.txt`. All patterns shown inline in
the script for auditability. Findings below cite exact measured counts from that run.

### Q1 — Delivery Delay vs. Order Tracking: not genuinely separable → merge
Tight patterns: `delay` matched 1,764, `tracking` matched 265, overlap only 17. Low overlap
looks like separation at first glance, but reading the "tracking-only" sample shows most of
them are *not* neutral status checks — they're delivery complaints phrased as a question
("where is my order!! Is it coming from Mars?", "why did I get a notice that delivery
failed"). The low overlap reflects **different phrasings of the same situation**, not two
different customer needs — the requested agent action (check tracking, apologize, escalate to
carrier) is the same either way. **Recommendation: merge into one intent, "Delivery Issue
(Delay/Non-Delivery/Tracking)."**

### Q2 — Prime / billing / info / cancellation: one merge, one removal
- `prime+charge`: 348, `prime+cancel`: 354, overlap: 33 (~9–10%) — low overlap, but both are
  thematically tight ("stop/dispute the Prime charge") and each is individually thin. **Merge
  into one intent, "Prime Membership Billing/Cancellation" (~669 combined, minus overlap).**
- Residual "prime mention, no charge/cancel language": 8,489 of 9,139 total Prime mentions
  (92.9%) — originally hypothesized as a "Prime info/how-to" intent. Reading the sample shows
  this is **wrong**: the residual is dominated by delivery complaints that happen to mention
  Prime ("prime order was unable to delivered", "paying for prime and items don't ship") —
  Prime is incidental context on a Delivery Issue, not its own topic. **Recommendation: do NOT
  create a "Prime Info/How-To" intent — not supported by the data.**
- Order Cancellation (broad, excluding Prime mentions): re-measured at 492 (vs. the earlier
  undercounted 238 from the narrow first-pass phrase match). Real examples confirm a clear,
  distinct action (cancel an order, sometimes disputing who should do it). **Kept as its own
  intent** with the corrected count.

### Q3 — Refund vs. Return: sufficiently distinct → keep separate
`refund` matched 2,316, `return/damaged/wrong-item` matched 3,437, overlap only 502 (21.7% of
refund matches). Reading confirms a real difference: refund-only messages ask for money back
(often with no physical item involved — non-delivery, duplicate charge); return-only messages
are about a physical item being wrong/damaged and want it exchanged, not necessarily refunded.
**Recommendation: keep Refund Request and Return/Wrong-or-Damaged-Item separate** — low overlap
plus a real difference in the requested action, not just two words for the same thing.

### Q4 — Address / Delivery Redirect: real but thin
Broader pattern matched 155 (0.19%) — all six sampled examples are clean, specifically about a
wrong/misdelivered address. This is genuinely distinguishable (cause = location, not timing),
but the volume is thin for a top-level intent. **Recommendation: keep it separate but flagged
as a deliberately low-frequency class** — the assignment explicitly wants "uncommon issues" and
rare-class handling represented in the golden set and macro-F1 evaluation, and this is the
cleanest naturally-occurring rare intent found so far. This is a judgment call, not a strong
data mandate either way.

### Q5 — Multilingual content: exclude from v1, escalate instead of classify
Heuristic non-English marker matched 6,318 of 82,246 (7.68%) — consistent with the earlier
~6.8% estimate. Given the project's priority is evaluation reliability over coverage,
**recommendation: treat non-English messages as an explicit out-of-scope/escalation case in
v1** — if a message is detected as non-English, escalate with reason "unsupported language,"
rather than building multilingual intent/retrieval/generation. This keeps the v1 golden set and
metrics honestly scoped to what the system actually handles, and the escalation reason is
transparent and consistent with the project's interpretability goal.

### Q6 — Multi-intent messages: rare enough that single-primary-intent is the right policy
Across 7 core intent patterns, only 877 of 82,246 messages (1.07%) matched 2+ patterns
simultaneously — lower than the qualitative impression from the original milestone's manual
reading (which cherry-picked examples, not a random/quantified sample). **Recommendation:
assign ONE primary intent per message**, chosen as the customer's most concrete, actionable
request (e.g., a message describing a delayed delivery that ends with "just refund me" gets
labeled Refund Request, not Delivery Issue). This is far easier to evaluate reliably — standard
single-label metrics (accuracy, macro/weighted F1, confusion matrix) apply directly, versus
multi-label evaluation requiring its own methodology for a phenomenon affecting ~1% of data.
Multi-intent cases can be flagged as a labeling-difficulty note in the golden set (not a
separate policy) if the annotator finds the primary-intent call genuinely ambiguous.

### Q7 — Generic complaints: inconclusive quantitatively, still recommended as a risk signal
810 generic-complaint matches; only 15.4% also matched one of the 7 core patterns — lower
co-occurrence than expected. But this test only checked against 7 of the ~9 candidate intents
(it excludes Account Access and Device/App Issue, for example), and manual reading of the
"generic-only" residual shows several do reference a real issue our patterns simply didn't
catch (e.g., "orders have started arriving late!" — a genuine delivery complaint that the
strict delay-regex missed because it doesn't contain "haven't received" or "marked delivered").
**The quantitative check is inconclusive on its own strict terms, but doesn't contradict the
qualitative reading.** Recommendation stands: keep generic complaints out of the intent
taxonomy, treat as a tone/escalation risk signal — but flagging this as a softer call than the
other re-pass findings, since the numeric test here was limited in coverage rather than a clean
confirmation.

## FINAL recommended taxonomy — 9 intents (RECOMMENDATION, pending your review)

| # | Intent | Definition | Approx. examples | 2 real examples | Closest confusable | Exact boundary |
|---|---|---|---|---|---|---|
| 1 | **Delivery Issue (Delay/Non-Delivery/Tracking)** | Package is late, marked delivered but not received, or customer is asking about delivery/tracking status | 1,764+ (strict phrasing) up to ~18,700 (loose "delivery" mention) — range reflects pattern strictness, not two different estimates of the same fixed thing; true count needs a trained classifier, not a fixed regex | "still waiting for a parcel to arrive, what time do you deliver till?" / "I've not received the order yet, nor am I able to track down the same" | Address/Redirect | Delivery Issue = timing/receipt/status problem; escalates to Address/Redirect only if customer explicitly names a wrong/incorrect address as the cause |
| 2 | **Address / Delivery Redirect Issue** | Package sent to, or delivered at, a wrong/incorrect address | 155 (thin — deliberately kept as the clearest naturally-occurring rare intent) | "the delivery driver has delivered to wrong address" / "trying AGAIN to deliver a package to the wrong address" | Delivery Issue | Requires explicit wrong-address language, not just non-receipt |
| 3 | **Refund Request** | Customer asks for money back | 2,316 | "no refund of 8k till now not received" / "took 9 minutes for a rep to tell me i'm not receiving a refund... even though i haven't received an item" | Return/Wrong-or-Damaged Item | Refund = wants money; no physical item exchange implied |
| 4 | **Return / Wrong or Damaged Item** | Item received is wrong, damaged, or defective; wants replacement/return | 3,437 | "the replacement of Ajanta Clock in broken condition" / "trying to return an order since one month" | Refund Request | Return = wants a physical fix (replace/send back); only 21.7% of these also ask for money back explicitly |
| 5 | **Prime Membership Billing/Cancellation** | Unwanted/unexpected Prime charge, or wants to cancel Prime | ~669 (348 charge + 354 cancel − 33 overlap) | "how does one get charged for a prime subscription if their card isn't even linked" / "I will cancel prime. this is the third late shipment THIS MONTH" | Refund Request | Must specifically concern the Prime subscription charge/cancellation itself, not a delivery problem on a Prime order |
| 6 | **Order Cancellation Request** | Wants to cancel a specific order (not the Prime subscription) | 492 (excludes Prime mentions) | "i haven't ordered anything this year, But u canceled my order" / "Why was my thanksgiving grocery order canceled?" | Refund Request | Cancellation = stopping/reversing an order before resolution; becomes Refund Request once the ask shifts to money back |
| 7 | **Account Access / Security** | Login failure, password reset, or unauthorized account/email changes | ~3,110 (carried over from initial pass, not re-measured this round — flag before golden-set sampling) | (see `reports/topic_examples_raw.txt`, `account_login` group) | Unauthorized Charge (when compromise causes billing impact) | Not re-verified with tightened patterns in this re-pass |
| 8 | **Device / App Technical Issue** | App or device (Kindle, Alexa, Fire TV) malfunctioning | ~2,380 (carried over, not re-split this round into fault-vs-how-to) | (see `reports/topic_examples_raw.txt`, `device_app_issue` group) | Prime Membership Billing/Cancellation (when device relates to a paid feature) | Not re-verified this round — still tangles technical faults with capability questions per the original Step 3 finding |
| 9 | **Unauthorized/Unexpected Charge (non-Prime)** | Charged for something not ordered or authorized, double-charged, outside the Prime-specific case | Not separately measured this round — the original combined `payment_charge` group was ~1,750 (includes Prime-related); non-Prime portion unknown | (see `reports/topic_examples_raw.txt`, `payment_charge` group) | Prime Membership Billing/Cancellation, Account Access | Needs its own targeted count before golden-set sampling — flagged, not fabricated |

**Merged:** Order Tracking → into Delivery Issue; Prime Cancellation → into Prime Membership
Billing (now one intent).
**Removed:** "Prime/Product Info or How-To" — not supported by the data; the residual bucket
that motivated it turned out to be Delivery Issue with incidental Prime mentions.
**Kept separate, with reasons:** Refund vs. Return (low overlap, different requested actions);
Address/Redirect (thin but real and deliberately useful as a rare class); Order Cancellation vs.
Refund (different action/stage).
**Not part of the taxonomy:** generic complaints (treated as an escalation/risk signal) and
non-English messages (treated as an explicit out-of-scope escalation case in v1, not a
classified intent).

**This is a recommendation pending your review** — intents #7–#9 in particular still need a
proper re-measurement pass (same rigor as #1–#6 above) before the golden set is sampled from
them.

---

# Final verification: Account Access, Device/App, Unauthorized Charge

Script: `scripts/verify_remaining_intents.py` (reuses the cached processed pickle only). Full
output: `reports/verify_remaining_raw.txt`.

### 1. Account Access / Security — **KEEP**
687 matches (0.84%). Read 10 real examples — all genuine (no obvious false positives):
login failures, locked accounts, "hacked account" reports, unauthorized password-change
notifications. AmazonHelp's replies are consistently distinct in kind (spam-folder checks,
routing to an "Account Specialist Team", sign-in troubleshooting) — a recognizably different
response pattern than delivery/billing replies. Overlap with refund pattern: 8/687 (1.2%).
Overlap with the Unauthorized-Charge pattern (checked from the charge side): 1/203 — negligible.
**Definition:** customer cannot access their account (login/password failure) or reports it was
accessed/changed without their authorization.
**Boundary vs. Unauthorized/Non-Prime Charge:** Account Access concerns control of the account
itself; Unauthorized Charge concerns a specific wrong/unexpected transaction. If a message
states both, classify by which the customer explicitly asks to be fixed first/primarily.

### 2. Device / App Technical Issue — **REMOVE**
Only 60 matches (0.07%) under a tightened pattern — an order of magnitude smaller than the
~2,380 carried-over estimate from the original loose keyword group. That original estimate was
inflated by non-technical Kindle/Alexa mentions (capability questions, gift-card balance, etc.),
confirming the original Step 3 concern. The 10 read examples are genuine technical-fault
reports with a distinctive troubleshooting-style reply pattern ("try restarting", "reinstalling
the app") — qualitatively real, but **60 examples across the entire 82,246-conversation dataset
is too thin to reliably train, retrieve evidence for, or evaluate as its own class** — thinner
than Address/Redirect (155), which was already the borderline case. **Not recommended as a
standalone intent for v1** — insufficient volume, not a quality judgment about whether the
pattern is real.

### 3. Unauthorized / Non-Prime Charge — **KEEP**
203 matches (0.25%, Prime-mentioning messages already excluded since those belong to Prime
Billing/Cancellation). 10 read examples are genuine: double charges, charged for an
undelivered/uncertain order, discrepancy between invoice and charge amount — no obvious false
positives. Overlap with Refund pattern: 17/203 (8.4%) — low, and the framing differs (customer
disputes the transaction itself vs. asking for money back for an unrelated reason). Overlap
with Account Access: 1/203 — negligible.
**Definition:** customer disputes a specific charge/transaction as wrong, duplicate, or
unauthorized — a billing-accuracy problem, not the general "give me money back" case.
**Boundary vs. Refund Request:** Unauthorized Charge = the core complaint is that the
transaction itself is wrong; Refund Request = the core complaint is not getting what was paid
for (non-delivery, cancellation, unwanted item) and money back is the requested fix. If both
stated, classify by which framing is the customer's primary complaint.

## FINAL locked-candidate taxonomy — 8 intents (still pending your sign-off)

| # | Intent | Status |
|---|---|---|
| 1 | Delivery Issue (Delay/Non-Delivery/Tracking) | merged (Tracking → Delivery) |
| 2 | Address / Delivery Redirect Issue | kept — deliberate rare class |
| 3 | Refund Request | kept |
| 4 | Return / Wrong or Damaged Item | kept |
| 5 | Prime Membership Billing/Cancellation | merged (Cancel → Billing) |
| 6 | Order Cancellation Request | kept, re-measured (492) |
| 7 | Account Access / Security | kept, verified this pass |
| 8 | Unauthorized / Non-Prime Charge | kept, verified this pass |

**Removed entirely:** "Prime/Product Info or How-To" (not supported by data), "Device/App
Technical Issue" (real pattern, insufficient volume — 60 examples).
**Not part of the taxonomy, handled elsewhere:** generic complaints (escalation/risk signal),
non-English messages (out-of-scope escalation case in v1), multi-intent messages (single
primary-intent labeling policy).

8 intents is on the low end of the original 8–12 target range, consistent with "a smaller
clean taxonomy is preferable to several overlapping labels." **Recommendation pending your
review before locking.**
