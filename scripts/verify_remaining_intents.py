"""
Final targeted verification pass for: Account Access/Security, Device/App Technical Issue,
Unauthorized/Non-Prime Charge. Reuses data/processed/amazonhelp_conversations.pkl only.

Run: python scripts/verify_remaining_intents.py > reports/verify_remaining_raw.txt
"""
import re
import pandas as pd

pd.set_option("display.max_colwidth", 220)


def load():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    customer = df[df["inbound"] == True]
    first = customer.sort_values("turn_index").groupby("_component").first().reset_index()
    brand_replies = df[df["inbound"] == False].sort_values("turn_index").groupby("_component").first()
    return first, brand_replies


def show(name, mask, first, brand_replies, n=10):
    matched = first[mask]
    print(f"\n{name}: {matched.shape[0]} ({100*matched.shape[0]/len(first):.2f}% of {len(first)})")
    sample = matched.sample(n=min(n, len(matched)), random_state=11)
    for _, r in sample.iterrows():
        comp = r["_component"]
        print(f"\n  [{comp}] CUSTOMER: {r['text']}")
        if comp in brand_replies.index:
            print(f"       AMAZONHELP: {brand_replies.loc[comp, 'text']}")
        else:
            print("       AMAZONHELP: [no reply on record]")
    return matched


def main():
    first, brand_replies = load()
    text = first["text"].fillna("")

    prime_rx = re.compile(r"\bprime\b", re.I)
    refund_rx = re.compile(r"\brefund\b", re.I)

    print("=" * 15, "1. Account Access / Security", "=" * 15)
    account_rx = re.compile(
        r"(password|log ?in failed|can.?t log ?in|locked out|hack(ed)?|account (compromised|hacked|locked|breached)"
        r"|reset my password|unable to (log ?in|access my account)|someone (accessed|changed) my account)",
        re.I,
    )
    account_mask = text.str.contains(account_rx, na=False)
    show("Account Access/Security matches", account_mask, first, brand_replies)
    print(f"\noverlap with refund pattern: {(account_mask & text.str.contains(refund_rx, na=False)).sum()}")
    print(f"overlap with 'prime': {(account_mask & text.str.contains(prime_rx, na=False)).sum()}")

    print("\n" + "=" * 15 + " 2. Device / App Technical Issue " + "=" * 15)
    device_rx = re.compile(
        r"(app (keeps? crashing|not working|won.?t (open|load|work)|is broken|glitch(ing)?)"
        r"|kindle (won.?t|not working|broken|frozen|stopped working)"
        r"|alexa (won.?t|not working|broken|stopped working|isn.?t responding)"
        r"|fire ?(tv|stick) (won.?t|not working|broken|frozen)"
        r"|website (is )?(down|not working|broken)|search (function\w*|bar)? (is )?broken)",
        re.I,
    )
    device_mask = text.str.contains(device_rx, na=False)
    show("Device/App Technical Issue matches", device_mask, first, brand_replies)
    print(f"\noverlap with prime pattern: {(device_mask & text.str.contains(prime_rx, na=False)).sum()}")

    print("\n" + "=" * 15 + " 3. Unauthorized / Non-Prime Charge " + "=" * 15)
    charge_rx = re.compile(
        r"(charged (me )?(for|twice|\$)|double charged|unauthorized charge|charged .{0,20} (didn.?t|never) (order|buy|authorize)"
        r"|wrongly charged|erroneous charge|deducted .{0,20}(twice|money|amount)|payment .{0,15}deducted)",
        re.I,
    )
    charge_mask = text.str.contains(charge_rx, na=False)
    charge_nonprime_mask = charge_mask & ~text.str.contains(prime_rx, na=False)
    print(f"charge pattern (all): {charge_mask.sum()}")
    print(f"charge pattern, prime-mentioning (excluded, already covered by Prime Billing intent): {(charge_mask & text.str.contains(prime_rx, na=False)).sum()}")
    show("Unauthorized/Non-Prime Charge matches (prime mentions excluded)", charge_nonprime_mask, first, brand_replies)
    print(f"\noverlap with refund pattern: {(charge_nonprime_mask & text.str.contains(refund_rx, na=False)).sum()}")
    print(f"overlap with account pattern: {(charge_nonprime_mask & account_mask).sum()}")


if __name__ == "__main__":
    main()
