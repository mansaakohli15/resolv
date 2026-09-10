"""
Milestone 3, Step 3 — pull real example conversations for manual inspection, grouped
by keyword patterns derived directly from evaluation/topic_terms.csv (the frequency
table from Step 2). Not LLM-generated, not fabricated — every keyword group is backed
by an observed high-frequency term/bigram.

NOTE: MiniBatchKMeans clustering in Step 2 did not produce usable topical clusters
(two mega-clusters covering ~100% of data + a long tail of singleton clusters — see
evaluation/topic_clusters.csv). It is not used here. This script uses simple keyword
matching on the frequency-table terms instead, which is a cruder but more honestly
grounded method given that finding.

Run: python scripts/inspect_topic_examples.py > reports/topic_examples_raw.txt
"""
import re
import pandas as pd

KEYWORD_GROUPS = {
    "delivery_timing": r"\b(deliver|delivered|delivery|arrive|arrived|waiting|late)\b",
    "order_tracking": r"\b(track|tracking|parcel|package|where.?s my order|wheres my order)\b",
    "refund_return": r"\b(refund|return|money back|reimburse)\b",
    "wrong_item": r"\b(wrong item|wrong order|damaged|broken|missing item|not what i ordered)\b",
    "account_login": r"\b(account|login|log in|password|locked out|sign in)\b",
    "prime_membership": r"\b(prime membership|amazon prime|cancel.{0,10}prime|renew)\b",
    "payment_charge": r"\b(charged|charge|payment|card declined|double charged|billed)\b",
    "cancel_order": r"\b(cancel my order|cancel order|cancel it)\b",
    "device_app_issue": r"\b(kindle|alexa|firestick|fire tv|app (crash|not working|won.?t))\b",
    "customer_service_complaint": r"\b(worst customer service|unhelpful|rude|terrible service|awful service)\b",
    "email_contact": r"\b(email|emailed|no response|no reply)\b",
}


def main():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    customer = df[df["inbound"] == True]
    first_msgs = customer.sort_values("turn_index").groupby("_component").first().reset_index()

    # first AmazonHelp reply per component (lowest turn_index among inbound=False rows)
    brand_replies = df[df["inbound"] == False].sort_values("turn_index").groupby("_component").first()

    for name, pattern in KEYWORD_GROUPS.items():
        rx = re.compile(pattern, re.I)
        matches = first_msgs[first_msgs["text"].str.contains(rx, na=False)]
        print(f"\n\n===== GROUP: {name}  (matches: {len(matches)}, {100*len(matches)/len(first_msgs):.2f}% of first messages) =====")
        sample = matches.sample(n=min(6, len(matches)), random_state=7)
        for _, row in sample.iterrows():
            comp = row["_component"]
            print(f"\n--- component {comp} ---")
            print(f"CUSTOMER: {row['text']}")
            if comp in brand_replies.index:
                print(f"AMAZONHELP: {brand_replies.loc[comp, 'text']}")
            else:
                print("AMAZONHELP: [no reply found -- should not happen given filtering, flag if seen]")


if __name__ == "__main__":
    main()
