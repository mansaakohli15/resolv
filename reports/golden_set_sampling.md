# Golden Evaluation Set — Sampling Report (Milestone 5)

Script: `scripts/build_golden_set.py`. Reuses `data/processed/amazonhelp_conversations.pkl`
and `evaluation/split_assignment.pkl` only — no raw CSV reload, no re-splitting.
Output: `evaluation/golden_set.csv`.

## Total size
**196 examples** (target ~200; landed slightly under because `address_redirect` only had 14
components available in the entire test split — took all of them rather than padding with a
weaker match).

## Sampling method
- One example per conversation component, the first customer message (consistent with every
  prior milestone's unit of analysis).
- 8 intents + `unmatched_other`: sampled **only from the `test` split** (Milestone 4), with a
  fixed quota per heuristic bucket, deliberately **not** proportional to raw pool size —
  oversampling the rare classes.
- Non-English quota (12): sampled from the `out_of_scope_non_english` split, not `test`. This is
  a documented, deliberate exception to "test only" — that pool was never part of
  `train_retrieval` or `validation` either (see `split_strategy.md`), so it carries no leakage
  risk; it's simply a different held-out pool than `test`, and it's the only place non-English
  examples exist since Milestone 4 routed them there.
- Within each non-rare, non-`unmatched_other` bucket, ~30% of the quota is deliberately drawn
  from components whose first message matched **2+ intent patterns simultaneously** (the same
  multi-hit signal used in Milestone 3's Q6 analysis) — these become the ambiguous/confusable
  examples, flagged in `annotator_notes`, not pre-labeled.
- `human_intent` and `difficulty` are left blank for every row. Per the milestone instructions,
  no heuristic or LLM labeling was performed — only selection.

## Distribution achieved

| heuristic_intent | count |
|---|---|
| delivery_issue | 30 |
| unmatched_other | 30 |
| unauthorized_nonprime_charge | 25 |
| refund_request | 20 |
| return_wrong_damaged | 20 |
| account_access_security | 15 |
| prime_billing_cancellation | 15 |
| order_cancellation | 15 |
| address_redirect | 14 (all available in test) |
| non_english_out_of_scope | 12 |

- Rare-class examples: **address_redirect (14) + unauthorized_nonprime_charge (25) = 39**, well
  above their ~7% combined share of the raw test-split distribution.
- `unmatched_other` examples: **30** — deliberately capped far below its 67% raw share of the
  test split, so the golden set isn't dominated by unlabeled/uncertain traffic, while still
  giving annotators real cases to classify as `Other / Out of Scope` or something the heuristics
  simply missed.
- Non-English examples: **12**, for testing the "unsupported language → escalate" path.
- Ambiguous/multi-pattern examples: **40** (flagged via `annotator_notes`, not a separate count
  field), spread across the buckets that had multi-hit candidates available.

## Why stratified (not uniform) sampling
A uniform random draw of ~200 from the 7,593-component test split would, in expectation, pull
roughly 0.4 `address_redirect` examples and ~1 `unauthorized_nonprime_charge` example — the rare
classes would very likely be entirely absent or represented by a single point, which the
assignment explicitly warned against ("carefully sampled rather than randomly chosen"; "common
and uncommon issues"). Stratifying by heuristic bucket guarantees every final intent is present
and gives the rare classes a real (if still modest) foothold for per-intent evaluation later.

## Leakage checks
- `component_id` uniqueness asserted (0 duplicates) — no component contributes more than one
  example.
- Asserted zero overlap between golden-set component IDs and `train_retrieval` component IDs.
- Asserted zero overlap between golden-set component IDs and `validation` component IDs.
- All 196 components therefore sit inside `test` or `out_of_scope_non_english` — both splits
  that were never used to build the retrieval corpus or train anything, so no golden example's
  own conversation can appear as its own evidence later.

## Sampling problems / classes that couldn't get full quota
- `address_redirect`: quota was 14, pool was exactly 14 — every available test-split example was
  taken. If more are needed later, the only source is re-pulling from `train_retrieval` or
  `validation`, which would require re-examining leakage implications (not recommended lightly).
- All other buckets met their quota from the test-split pool without exhausting it.

## Command to reproduce
```
python scripts/build_golden_set.py
```
