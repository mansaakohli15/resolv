# Resolv — Evidence-Grounded Customer Support Automation (Hiver SDE Intern take-home)

Resolv is being built on the *Customer Support on Twitter* dataset (Kaggle,
`thoughtvector/customer-support-on-twitter`), with **AmazonHelp** as the selected brand.

## Status: Milestones 1–5B complete. Nothing past this has been built.

This project currently covers, and ONLY covers:

1. **Dataset inspection** — schema, row counts, null/duplicate rates, `inbound` field,
   `response_tweet_id`/`in_response_to_tweet_id` link structure (`scripts/inspect_dataset.py`).
2. **Conversation reconstruction** — thread reconstruction via union-find over
   `in_response_to_tweet_id`, with documented leakage/noise findings.
3. **AmazonHelp brand selection** — measured profiling of 10 candidate brands and a documented
   rationale for selecting AmazonHelp (`scripts/profile_brands.py`,
   `evaluation/brand_profile.csv`, `reports/brand_selection.md`).
4. **Intent discovery and taxonomy refinement** — frequency/TF-IDF analysis, manual reading of
   real examples, and two rounds of targeted re-verification, converging on a final **8-intent
   taxonomy** (`scripts/discover_topics.py`, `scripts/inspect_topic_examples.py`,
   `scripts/repass_intent_analysis.py`, `scripts/verify_remaining_intents.py`,
   `reports/intent_discovery.md`).
5. **Leakage-safe, component-level splitting** — 80/10/10 train-retrieval/validation/test split
   at the conversation-component level, stratified by a heuristic intent tag used only for
   split planning, with non-English conversations set aside as out-of-scope
   (`scripts/split_dataset.py`, `evaluation/split_assignment.pkl`,
   `evaluation/split_stats.csv`, `reports/split_strategy.md`).
6. **Golden evaluation set sampling** — 196 examples sampled from the test split (+ a small
   non-English quota from the out-of-scope pool), stratified/oversampled toward rare intents,
   with `human_intent`/`difficulty` deliberately left blank at sampling time
   (`scripts/build_golden_set.py`, `evaluation/golden_set.csv`,
   `reports/golden_set_sampling.md`).
7. **Golden-set annotation** — every one of the 196 examples independently read and labeled
   against the final 8-intent taxonomy (plus Other/Out of Scope, Unsupported Language,
   Ambiguous), with a full disagreement/quality-control writeup
   (`scripts/annotate_golden_set.py`, `reports/golden_set_annotation.md`).

## Final 8-intent taxonomy (locked)

1. Delivery Issue (Delay/Non-Delivery/Tracking)
2. Address / Delivery Redirect Issue
3. Refund Request
4. Return / Wrong or Damaged Item
5. Prime Membership Billing/Cancellation
6. Order Cancellation Request
7. Account Access / Security
8. Unauthorized / Non-Prime Charge

Plus three non-intent labels used in the golden set: `Other / Out of Scope`,
`Unsupported Language`, `Ambiguous`.

## NOT YET IMPLEMENTED

The following are explicitly **not built yet** — no code, no results, nothing to evaluate:

- Intent classifier (baselines or final model)
- Evidence retrieval (lexical/semantic/hybrid)
- Grounded reply generation
- Escalation/confidence model
- Baseline comparisons (majority-class, TF-IDF+LogReg, or any other)
- Final evaluation harness / headline metrics / failure analysis

Any numbers you see in `evaluation/` and `reports/` are about the **data and the golden set
only** (brand profiling, topic discovery, split statistics, annotation distribution) — none of
them are model performance numbers, because no model has been trained or run.

## Project layout

```
scripts/      one script per milestone step (see docstring at the top of each for what it does
               and reads/writes)
data/
  processed/  amazonhelp_conversations.pkl — the filtered, chronologically-ordered AmazonHelp
              conversation dataset (82,246 conversations). This is a derived artifact, not the
              raw dataset.
              first_msg_clusters.pkl — per-conversation cluster assignments from the (largely
              inconclusive, see reports/intent_discovery.md) topic-clustering attempt.
  raw/        INTENTIONALLY EMPTY in this download — see "Reproducing from scratch" below.
evaluation/   CSV/pickle artifacts: brand profiling, topic terms/clusters, split assignment and
              stats, the golden evaluation set (evaluation/golden_set.csv, fully annotated).
reports/      Markdown/text reports documenting every milestone's findings and decisions.
```

## Reproducing from scratch

The raw dataset (`twcs.csv`, ~2.8M rows, and the cached full-dataframe pickle derived from it)
is **not included** in this download — it's large (~550MB as cached artifacts) and not needed to
continue past Milestone 5B, since `data/processed/amazonhelp_conversations.pkl` already contains
everything downstream work depends on.

To re-run Milestones 1–2 (dataset inspection, brand profiling) from scratch:
1. Download the Kaggle dataset (`thoughtvector/customer-support-on-twitter`) and place
   `twcs.csv` at `data/raw/twcs/twcs.csv`.
2. Run `python scripts/inspect_dataset.py` — this rebuilds `data/raw/twcs_cache.pkl` and is a
   prerequisite for `scripts/profile_brands.py` and `scripts/build_brand_dataset.py`
   (both also expect `data/raw/twcs_components.pkl`, produced by the ad-hoc union-find
   reconstruction documented in `reports/brand_selection.md` — not yet its own script; flagged
   here rather than silently assumed).

To continue from Milestone 6 onward, `data/processed/amazonhelp_conversations.pkl` and
everything in `evaluation/` is already sufficient — no raw data needed.

## Setup

```
pip install -r requirements.txt
```

No API keys or credentials are used or required by anything in this project yet.
