# Golden Set Human Annotation Report (Milestone 5B)

Script: `scripts/annotate_golden_set.py`. Every row was read independently against the message
text and the taxonomy rules — `heuristic_intent` was used only as the earlier sampling aid, not
copied. Updated in place: `evaluation/golden_set.csv`.

## Total examples annotated
196 (all rows in the golden set; every row has both `human_intent` and `difficulty` filled).

## Final human_intent distribution

| human_intent | count |
|---|---|
| Delivery Issue | 35 |
| Return / Wrong or Damaged Item | 21 |
| Prime Membership Billing/Cancellation | 21 |
| Refund Request | 19 |
| Unsupported Language | 18 |
| Other / Out of Scope | 16 |
| Account Access / Security | 16 |
| Address / Delivery Redirect Issue | 15 |
| Unauthorized / Non-Prime Charge | 15 |
| Order Cancellation Request | 15 |
| Ambiguous | 5 |

## Difficulty distribution
Easy: 105, Medium: 74, Hard: 17.

## Heuristic vs. human disagreements
166 of 196 rows had a heuristic tag that asserted a specific intent (the other 30 were
`unmatched_other`, which asserts nothing). Of those 166, **27 (16.3%) were corrected** on
independent reading — kept as the human label with a note, per the annotation rule. Breakdown:

- **10 of 27**: `unauthorized_nonprime_charge` → `Prime Membership Billing/Cancellation`, all
  because the message explicitly named a Prime charge — the stated rule ("Prime charges belong
  to the Prime intent") directly resolves these; the keyword-driven heuristic doesn't know that
  rule.
- **5 of 27**: `prime_billing_cancellation` → something else (Delivery Issue ×2, Order
  Cancellation, Refund, Other) — cases where "prime" appeared near "cancel"/context but there
  was no actual current Prime billing/cancellation action.
- **3 of 27**: `return_wrong_damaged` → `Other / Out of Scope` — positive or how-to messages
  that merely mentioned a return in passing.
- Remainder: individual boundary calls (delivery vs. address, refund vs. charge, order
  cancellation vs. its knock-on effects) — see the representative examples below.

## unmatched_other did not turn out to be one thing
The 30 `unmatched_other` rows resolved to: Other/Out of Scope (11), **Unsupported Language
(6)**, Return/Wrong-or-Damaged (4), Ambiguous (4), Delivery Issue (3), Address/Redirect (1),
Account Access/Security (1). The 6 language misses are notable: the split-time heuristic's
non-English marker list only covered Spanish/Portuguese/French terms and missed Japanese and
German entirely — those messages were sitting in `unmatched_other` rather than
`non_english_out_of_scope`, and would have been silently excluded from the "non-English" study
if not read individually here.

## Ambiguous / Other / Unsupported Language counts
- **Ambiguous: 5** (2.6%) — genuinely rare, consistent with Milestone 3's finding that most
  messy-looking messages do have a defensible primary intent once read carefully.
- **Other / Out of Scope: 16** (8.2%) — real and non-trivial: product feedback, feature
  requests, spam/marketing complaints, positive-only messages, website bugs unrelated to the 8
  support intents.
- **Unsupported Language: 18** (9.2%, including the 6 recovered from `unmatched_other`) —
  somewhat higher than the ~6.8–7.7% heuristic estimates from Milestones 3–4, because those
  estimates also had the same Romance-language-only blind spot.

## 8 representative difficult/disputed examples

1. **g0068** (`refund_request` → `Prime Membership Billing/Cancellation`): "received refund for
   last amazon prime but now see that the past two months have been charged." The word "refund"
   triggered the heuristic, but the live, unresolved issue is a recurring Prime charge.
2. **g0075** (Hard): "was meant to be getting a refund and Amazon took more money out of my
   account!" Two stacked issues (pending refund + a new charge) — classified as
   `Unauthorized / Non-Prime Charge` since the new deduction is the more concrete, current
   complaint.
3. **g0117** (Hard, heuristic false positive): "Amazon Prime Now just cancelled my order for no
   reason." "Prime Now" is a service *name* here, not the subscription — the cancel-near-prime
   regex matched by coincidence. Classified `Order Cancellation Request`.
4. **g0134** (Hard, heuristic false positive): "...after assuring have not cancelled the
   order." The word "cancelled" appears only to deny that a cancellation happened; the real
   issue is non-delivery. Classified `Delivery Issue`.
5. **g0157** (Hard, genuine multi-intent): a numbered two-part message — "1. My Order is going
   to the wrong address... 2. My white gloves order is apparently double cream." Two
   independent, equally concrete issues in one tweet. Classified `Address / Delivery Redirect
   Issue` (listed first, more systemic) per the single-primary-intent policy, but this is the
   clearest evidence in the set that the policy sometimes discards real information.
6. **g0171** (Hard, genuine taxonomy edge case): "got charged for a tv episode that is included
   with prime. How can I dispute this?" This is a charge for *content under a Prime benefit*,
   not the Prime *subscription* fee — the stated Prime-charge rule doesn't cleanly cover it.
   Classified `Unauthorized / Non-Prime Charge`, flagged as a rule gap (see below).
7. **g0102** (Hard, genuine toss-up): locked out of the account *and* being charged for Prime
   *and* unable to cancel it because of the lockout. Classified `Account Access / Security`
   since the lockout is what's blocking resolution of everything else, but `Prime Membership
   Billing/Cancellation` is equally defensible.
8. **g0058** (Medium, heuristic language-detection gap): German text asking how to remove an
   expired watchlist offer. The heuristic's non-English marker list doesn't cover German at
   all — classified `Unsupported Language` on independent reading.

## Is the 8-intent taxonomy holding up?
Yes, largely. Every one of the 8 intents received real, independently-confirmed examples in this
pass, including the two previously-thin classes (Address: 15, Unauthorized Charge: 15) — neither
turned out to be a fabricated or forced category once read directly. The explicit "Prime charges
go to Prime, not Unauthorized Charge" rule did real work and resolved the majority of
disagreements cleanly. `Other / Out of Scope`, `Unsupported Language`, and `Ambiguous` are
functioning as genuine, necessary fallbacks rather than dumping grounds — Ambiguous in
particular stayed rare (5/196), which is a good sign the taxonomy has real boundaries rather than
needing a catch-all.

## Taxonomy changes believed genuinely necessary
1. **Refine the Prime-charge rule** (not the taxonomy structure itself): it should explicitly
   distinguish "charged for the Prime *subscription*" (→ Prime Membership Billing/Cancellation)
   from "charged for content/a purchase that happens to relate to a Prime benefit" (→
   Unauthorized/Non-Prime Charge), per g0171. Currently the rule text doesn't draw this line and
   an annotator has to infer it.
2. **Write an explicit Order Cancellation Request boundary rule**: several judgment calls
   (g0117, g0131, g0135, g0158) turned on whether "the conversation is about a cancellation
   event" is enough, versus whether the customer's *current unresolved ask* is something else
   (a refund, an address fix, or nothing at all because it already happened). This was the
   single most frequent source of borderline calls after the Prime rule.
3. **No change recommended to the 8 intents themselves** — no evidence from this annotation
   pass supports adding, removing, or merging any of the 8. `Other / Out of Scope` should remain
   a permanent fallback, not be absorbed into the 8, since 8.2% of real traffic genuinely doesn't
   fit and forcing it in would corrupt the classes that do have clean boundaries.

These are recommendations for your review — no taxonomy or label was changed automatically past
what's documented here.
