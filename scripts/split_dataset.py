"""
Milestone 4 — leakage-safe split for Resolv.

Reuses data/processed/amazonhelp_conversations.pkl only (no raw CSV reload).

IMPORTANT LABEL DISCLAIMER:
No human intent labels exist yet. The "heuristic_intent" column below is produced by the same
regex patterns used during intent discovery (Milestones 3), applied here ONLY to (a) check that
rare classes get enough representation in val/test, and (b) stratify the split so that doesn't
happen by chance. It is NOT ground truth and must not be treated as such downstream -- real
labels come from human labeling of the golden set later. This is flagged explicitly rather than
silently used as if it were reliable.

Unit of split = conversation component (_component). No component appears in more than one
split. Non-English components (heuristic marker) are set aside from the intent-modeling pools
entirely -- they're out of scope for v1 classification per the approved taxonomy decision, but
are NOT deleted, only tagged, in case they're needed later for escalation-behavior testing.

Output: evaluation/split_assignment.pkl  (columns: _component, split, heuristic_intent, is_non_english_heuristic)
        evaluation/split_stats.csv

Run: python scripts/split_dataset.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

NON_EN_RX = re.compile(
    r"\b(que|para|con|env[ií]o|pedido|gracias|hola|est[aá]|n[ãa]o|voc[eê]|obrigad|bonjour|merci|colis|livraison)\b",
    re.I,
)

# Priority order = most specific/actionable first, matching the single-primary-intent policy
# from Milestone 3. A component gets the FIRST pattern (in this order) that matches its first
# customer message; if none match, it's tagged "unmatched_other" (includes generic complaints,
# removed candidates like Prime info/how-to and device issues, and anything else uncovered).
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


def tag_intent(text):
    for name, rx in INTENT_PATTERNS:
        if rx.search(str(text)):
            return name
    return "unmatched_other"


def main():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    customer = df[df["inbound"] == True]
    first_msgs = customer.sort_values("turn_index").groupby("_component").first().reset_index()

    first_msgs["is_non_english_heuristic"] = first_msgs["text"].str.contains(NON_EN_RX, na=False)
    first_msgs["heuristic_intent"] = first_msgs["text"].map(tag_intent)
    # non-English components are out of scope for intent modeling; tag them distinctly
    first_msgs.loc[first_msgs["is_non_english_heuristic"], "heuristic_intent"] = "non_english_out_of_scope"

    in_scope = first_msgs[~first_msgs["is_non_english_heuristic"]].copy()
    out_of_scope = first_msgs[first_msgs["is_non_english_heuristic"]].copy()
    print(f"Total components: {len(first_msgs)}")
    print(f"In-scope (English-heuristic): {len(in_scope)}")
    print(f"Out-of-scope (non-English-heuristic): {len(out_of_scope)}")

    # component-level stratified split: 80% train(+retrieval corpus) / 10% val / 10% test
    train_ids, temp_ids = train_test_split(
        in_scope["_component"], test_size=0.20, random_state=RANDOM_STATE,
        stratify=in_scope["heuristic_intent"],
    )
    temp_df = in_scope[in_scope["_component"].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_df["_component"], test_size=0.50, random_state=RANDOM_STATE,
        stratify=temp_df["heuristic_intent"],
    )

    split_map = {}
    for cid in train_ids:
        split_map[cid] = "train_retrieval"
    for cid in val_ids:
        split_map[cid] = "validation"
    for cid in test_ids:
        split_map[cid] = "test"
    for cid in out_of_scope["_component"]:
        split_map[cid] = "out_of_scope_non_english"

    assignment = first_msgs[["_component", "heuristic_intent", "is_non_english_heuristic"]].copy()
    assignment["split"] = assignment["_component"].map(split_map)

    assert assignment["_component"].nunique() == len(assignment), "component appears more than once!"
    assert assignment.groupby("_component")["split"].nunique().max() == 1, "component split leakage detected!"

    # customer-message counts per split (all turns, not just first message)
    comp_to_split = assignment.set_index("_component")["split"]
    df["split"] = df["_component"].map(comp_to_split)
    msg_counts = df[df["inbound"] == True].groupby("split").size()

    print("\n=== Components per split ===")
    print(assignment["split"].value_counts())
    print("\n=== Customer messages per split (all turns) ===")
    print(msg_counts)

    print("\n=== Heuristic intent distribution per split (component-level, first message) ===")
    dist = assignment.groupby(["split", "heuristic_intent"]).size().unstack(fill_value=0)
    print(dist.to_string())

    # rare class check
    print("\n=== Rare class check (address_redirect, unauthorized_nonprime_charge) ===")
    for cls in ["address_redirect", "unauthorized_nonprime_charge"]:
        row = dist[cls] if cls in dist.columns else None
        if row is not None:
            print(f"{cls}: train={row.get('train_retrieval',0)}, val={row.get('validation',0)}, test={row.get('test',0)}")

    assignment.to_pickle("evaluation/split_assignment.pkl")
    dist.to_csv("evaluation/split_stats.csv")
    print("\nSaved evaluation/split_assignment.pkl, evaluation/split_stats.csv")


if __name__ == "__main__":
    main()
