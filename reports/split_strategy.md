# Split Strategy — Resolv (Milestone 4)

Script: `scripts/split_dataset.py` (reuses `data/processed/amazonhelp_conversations.pkl` only).
Outputs: `evaluation/split_assignment.pkl`, `evaluation/split_stats.csv`.

## Label disclaimer
No human intent labels exist yet. `heuristic_intent` in the split assignment is produced by the
same regex patterns from intent discovery, used **only** to stratify the split and sanity-check
rare-class representation — it is explicitly NOT ground truth. 67% of test-split components
(5,095/7,593) fall into `unmatched_other` — the strict patterns don't cover most phrasings, so
the true label distribution will only be known once the golden set is human-labeled. This is a
planning tool, not a labeling methodology.

## Split unit and leakage safeguards
- Split at the **conversation-component level** (`_component`), never the tweet level.
- Assertions in the script confirm every component maps to exactly one split.
- Retrieval/evidence corpus = the `train_retrieval` split only. Since validation, test, and any
  future golden-set component IDs never appear in `train_retrieval`, a test/golden example's own
  conversation can never appear among its own retrieved evidence, by construction.
- Non-English components (heuristic marker, ~6,318) are set aside as `out_of_scope_non_english`
  — not deleted, but excluded from train/val/test/retrieval for v1, per the approved taxonomy
  decision to escalate rather than classify them.

## Proportions and split sizes
80% train+retrieval / 10% validation / 10% test, stratified by `heuristic_intent` at the
component level (via `sklearn.train_test_split(..., stratify=...)`), applied only within the
75,928 English-heuristic in-scope components.

| Split | Components | Customer messages (all turns) |
|---|---|---|
| train_retrieval | 60,742 | 148,110 |
| validation | 7,593 | 18,138 |
| test | 7,593 | 18,515 |
| out_of_scope_non_english | 6,318 | 17,413 |

## Heuristic intent distribution (component-level, first message; NOT ground truth)
Full table in `evaluation/split_stats.csv`. Key rows:

| heuristic_intent | train | val | test |
|---|---|---|---|
| delivery_issue | 14,149 | 1,768 | 1,769 |
| unmatched_other | 40,763 | 5,096 | 5,095 |
| refund_request | 1,336 | 167 | 167 |
| return_wrong_damaged | 2,717 | 340 | 340 |
| account_access_security | 542 | 68 | 67 |
| prime_billing_cancellation | 425 | 53 | 53 |
| order_cancellation | 410 | 51 | 52 |
| **address_redirect** | **114** | **15** | **14** |
| **unauthorized_nonprime_charge** | **286** | **35** | **36** |

## Rare-class check
Address (114/15/14) and Unauthorized Charge (286/35/36) are thin everywhere, as expected from
their dataset-wide totals (155 and 203, measured in earlier milestones). Stratifying the split
prevented them from being accidentally under- or zero-represented in val/test, but:
- **14 and 36 heuristic-tagged examples in test is not enough to compute a stable per-intent F1**
  on its own. It's enough to guarantee both classes appear at all, and enough to hand-pick a
  meaningful (not statistically precise) slice for the golden set.
- **For classifier training**, 114/286 raw examples is thin for a supervised class, and the real
  count is unknown until human labeling happens (the 67% `unmatched_other` bucket may contain
  more true instances that the strict regex missed). This is a genuine open risk for Milestone 5,
  not solved here — likely needs class-weighting, oversampling, or accepting weaker per-intent
  performance on these two classes, decided against real numbers once labeled.

## Golden-set strategy (not created yet)
- Golden set (150–250 examples) will be sampled **from the `test` split only** — never from
  train_retrieval or validation — so it's automatically insulated from the retrieval corpus and
  from anything the classifier trains on.
- Sampling must be **stratified/oversampled by heuristic tag, not uniform** — a uniform random
  draw of ~200 from 7,593 test components would yield well under 1 expected Address example by
  chance. The plan: take most or all of the rare-tagged pools (14 address_redirect, 36
  unauthorized_nonprime_charge) plus a deliberately chosen mix from the larger buckets
  (delivery_issue, refund_request, return_wrong_damaged, unmatched_other, etc.), consistent with
  the assignment's "common and uncommon issues" sampling requirement. Exact target counts per
  bucket are a labeling-phase decision, not made here.
- A small number of `out_of_scope_non_english` and `unmatched_other` examples should likely be
  deliberately included to test that the escalation logic (unsupported language, low
  classifier confidence) behaves correctly — a decision for the labeling phase, flagged here as
  a recommendation, not implemented.

## Issues to resolve before classifier training
1. No human labels exist yet — heuristic tags are for split planning only and must not leak into
   training as if they were ground truth.
2. Address and Unauthorized Charge are thin in every split; a plan for handling rare classes
   (class weights, oversampling, or accepting the limitation honestly in the report) is needed
   before Milestone 5, once real counts are known.
3. `unmatched_other` is the majority class in this heuristic view (67%) — the real intent
   taxonomy's coverage of the data is unknown until labeling; it's possible some of these are
   genuinely out-of-taxonomy and will need a fallback/low-confidence handling path.
