"""
Milestone 5 -- sample (NOT label) the Resolv golden evaluation set.

Reuses:
  data/processed/amazonhelp_conversations.pkl
  evaluation/split_assignment.pkl   (from Milestone 4 -- split + heuristic_intent per component)

No human_intent or difficulty values are assigned here -- both are left blank for manual
annotation, per the milestone instructions. heuristic_intent, language_scope, and
annotator_notes are dataset-derived/heuristic fields, clearly distinct from the (blank)
human_intent column.

Sampling source:
  - The 8 final intents + unmatched_other: sampled ONLY from the `test` split (Milestone 4).
  - A small non-English quota: sampled from the `out_of_scope_non_english` split. This split was
    never part of train_retrieval or validation either (see split_strategy.md), so drawing from
    it does not reintroduce any leakage risk -- it's a separate, always-held-out pool, just not
    literally the `test` split. Documented explicitly, not silently done.

Run: python scripts/build_golden_set.py
Output: evaluation/golden_set.csv
"""
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42

INTENT_PATTERNS = [
    ("account_access_security", re.compile(
        r"(password|log ?in failed|can.?t log ?in|locked out|hack(ed)?|account (compromised|hacked|locked|breached)"
        r"|reset my password|unable to (log ?in|access my account)|someone (accessed|changed) my account)", re.I)),
    ("unauthorized_nonprime_charge", re.compile(
        r"(charged (me )?(for|twice|\$)|double charged|unauthorized charge|charged .{0,20}(didn.?t|never) (order|buy|authorize)"
        r"|wrongly charged|erroneous charge|deducted .{0,20}(twice|money|amount)|payment .{0,15}deducted)", re.I)),
    ("prime_billing_cancellation", re.compile(
        r"(prime.{0,25}(charg|billed|renew)|(?:charg|billed).{0,25}prime|cancel.{0,15}prime|prime.{0,15}cancel)", re.I)),
    ("order_cancellation", re.compile(
        r"(cancel(l?ing|led|ed)?\s+(my |the )?order|order\s+cancel(l?ation)?)", re.I)),
    ("return_wrong_damaged", re.compile(
        r"(\breturn(ed|ing)?\b|\bwrong item\b|\bdamaged\b|\bdefective\b|\breplacement\b)", re.I)),
    ("refund_request", re.compile(r"\brefund\b", re.I)),
    ("address_redirect", re.compile(
        r"(wrong address|change (my |the )?address|address is wrong|update (my |the )?address"
        r"|not at (this|that) address|left at (the )?wrong (address|place)|sent to (the )?wrong address"
        r"|deliver(ed)? to (the )?(wrong|incorrect) address)", re.I)),
    ("delivery_issue", re.compile(
        r"(haven.?t received|not received|never (arrived|delivered)|still (waiting|havent|haven.?t)"
        r"|marked (as )?delivered but|missed (the )?delivery|failed delivery|delivery (attempt )?(failed|missed)"
        r"|delivery (was )?(late|delayed)|deliver|delivered|delivery|arrive)", re.I)),
]

# quota per heuristic bucket; deliberately NOT proportional to raw pool size (oversampling rare classes)
QUOTAS = {
    "delivery_issue": 30,
    "unmatched_other": 30,
    "refund_request": 20,
    "return_wrong_damaged": 20,
    "account_access_security": 15,
    "prime_billing_cancellation": 15,
    "order_cancellation": 15,
    "address_redirect": 14,          # take (near) all available -- pool itself is only 14 in test
    "unauthorized_nonprime_charge": 25,
}
NON_ENGLISH_QUOTA = 12
MULTI_HIT_FRACTION = 0.3  # within each non-rare bucket, prefer this share from ambiguous (multi-pattern) examples


def count_hits(text):
    return sum(1 for _, rx in INTENT_PATTERNS if rx.search(str(text)))


def main():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    customer = df[df["inbound"] == True]
    first_msgs = customer.sort_values("turn_index").groupby("_component").first().reset_index()
    comp_sizes = df.groupby("_component").size()

    assignment = pd.read_pickle("evaluation/split_assignment.pkl")
    merged = first_msgs.merge(assignment, on="_component", suffixes=("", "_assign"))
    merged["n_pattern_hits"] = merged["text"].map(count_hits)
    merged["component_size"] = merged["_component"].map(comp_sizes)

    rows = []
    rng = np.random.RandomState(RANDOM_STATE)

    test_pool = merged[merged["split"] == "test"]

    for intent, quota in QUOTAS.items():
        bucket = test_pool[test_pool["heuristic_intent"] == intent]
        quota = min(quota, len(bucket))
        if intent == "unmatched_other":
            chosen = bucket.sample(n=quota, random_state=RANDOM_STATE)
            for _, r in chosen.iterrows():
                rows.append((r, "no heuristic pattern matched -- candidate Other/Out-of-Scope or missed phrasing"))
            continue

        multi = bucket[bucket["n_pattern_hits"] >= 2]
        single = bucket[bucket["n_pattern_hits"] < 2]
        n_multi = min(int(round(quota * MULTI_HIT_FRACTION)), len(multi))
        n_single = quota - n_multi
        n_single = min(n_single, len(single))
        # if single pool short, backfill from remaining multi
        shortfall = quota - n_multi - n_single
        chosen_multi = multi.sample(n=n_multi, random_state=RANDOM_STATE) if n_multi > 0 else multi.iloc[0:0]
        chosen_single = single.sample(n=n_single, random_state=RANDOM_STATE) if n_single > 0 else single.iloc[0:0]
        extra = pd.DataFrame()
        if shortfall > 0:
            remaining_multi = multi.drop(chosen_multi.index)
            extra = remaining_multi.sample(n=min(shortfall, len(remaining_multi)), random_state=RANDOM_STATE)
        for _, r in pd.concat([chosen_multi, extra]).iterrows():
            rows.append((r, f"heuristic multi-pattern match ({r['n_pattern_hits']} patterns) -- candidate ambiguous/confusable example"))
        for _, r in chosen_single.iterrows():
            rows.append((r, "heuristic single-pattern match"))

    non_en_pool = merged[merged["split"] == "out_of_scope_non_english"]
    non_en_quota = min(NON_ENGLISH_QUOTA, len(non_en_pool))
    non_en_chosen = non_en_pool.sample(n=non_en_quota, random_state=RANDOM_STATE)
    for _, r in non_en_chosen.iterrows():
        rows.append((r, "sampled from out_of_scope_non_english pool (heuristic non-English marker matched)"))

    golden = pd.DataFrame([{
        "golden_id": f"g{idx:04d}",
        "source_tweet_id": r["tweet_id"],
        "component_id": r["_component"],
        "text": r["text"],
        "heuristic_intent": r["heuristic_intent"],
        "human_intent": "",         # left blank for manual annotation
        "difficulty": "",           # left blank for manual annotation
        "language_scope": "out_of_scope_non_english_heuristic" if r["split"] == "out_of_scope_non_english" else "in_scope_english_heuristic",
        "annotator_notes": note,
    } for idx, (r, note) in enumerate(rows, start=1)])

    # leakage checks
    assert golden["component_id"].nunique() == len(golden), "duplicate component in golden set!"
    assert not set(golden["component_id"]) & set(assignment.loc[assignment["split"] == "train_retrieval", "_component"]), \
        "golden set overlaps train_retrieval!"
    assert not set(golden["component_id"]) & set(assignment.loc[assignment["split"] == "validation", "_component"]), \
        "golden set overlaps validation!"

    golden.to_csv("evaluation/golden_set.csv", index=False)

    print(f"Golden set size: {len(golden)}")
    print("\nBy heuristic_intent:")
    print(golden["heuristic_intent"].value_counts().to_string())
    print("\nBy language_scope:")
    print(golden["language_scope"].value_counts().to_string())
    print(f"\nAmbiguous (multi-pattern) notes: {golden['annotator_notes'].str.contains('ambiguous').sum()}")
    print("\nSaved evaluation/golden_set.csv")


if __name__ == "__main__":
    main()
