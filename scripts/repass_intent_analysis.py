"""
Milestone 3 targeted re-pass. Reuses data/processed/amazonhelp_conversations.pkl only
(no raw CSV reparse). Answers the 7 specific re-pass questions with tighter, documented
regex patterns. All patterns are shown inline so counts are reproducible/auditable.

Run: python scripts/repass_intent_analysis.py > reports/repass_raw.txt
"""
import re
import pandas as pd

pd.set_option("display.max_colwidth", 200)


def load_first_msgs():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    customer = df[df["inbound"] == True]
    first = customer.sort_values("turn_index").groupby("_component").first().reset_index()
    return first


def show(name, mask, first, n=4):
    matched = first[mask]
    print(f"\n{name}: {matched.shape[0]} ({100*matched.shape[0]/len(first):.2f}% of {len(first)})")
    for _, r in matched.sample(n=min(n, len(matched)), random_state=3).iterrows():
        print(f"   [{r['_component']}] {r['text']}")
    return matched


def main():
    first = load_first_msgs()
    text = first["text"].fillna("")

    print("=" * 20, "Q1: Delivery Delay vs Order Tracking", "=" * 20)
    delay_rx = re.compile(
        r"(haven.?t received|not received|never (arrived|delivered)|still (waiting|havent|haven.?t)"
        r"|marked (as )?delivered but|missed (the )?delivery|failed delivery|delivery (attempt )?(failed|missed)"
        r"|delivery (was )?(late|delayed))",
        re.I,
    )
    tracking_rx = re.compile(
        r"(track(ing)? my (order|package|parcel)|where.?s my order|where is my order|tracking number"
        r"|order status|what.?s the status)",
        re.I,
    )
    delay_mask = text.str.contains(delay_rx, na=False)
    tracking_mask = text.str.contains(tracking_rx, na=False)
    both = delay_mask & tracking_mask
    delay_only = delay_mask & ~tracking_mask
    tracking_only = tracking_mask & ~delay_mask
    print(f"delay-pattern matches: {delay_mask.sum()}")
    print(f"tracking-pattern matches: {tracking_mask.sum()}")
    print(f"overlap (both patterns): {both.sum()}")
    show("Delay-only examples", delay_only, first)
    show("Tracking-only examples", tracking_only, first)

    print("\n" + "=" * 20 + " Q2: Prime / billing / info / cancellation " + "=" * 20)
    prime_any_rx = re.compile(r"\bprime\b", re.I)
    prime_charge_rx = re.compile(r"prime.{0,25}(charg|billed|renew)|(?:charg|billed).{0,25}prime", re.I)
    prime_cancel_rx = re.compile(r"cancel.{0,15}prime|prime.{0,15}cancel", re.I)
    order_cancel_rx = re.compile(r"cancel(l?ing|led|ed)?\s+(my |the )?order|order\s+cancel(l?ation)?", re.I)

    prime_mask = text.str.contains(prime_any_rx, na=False)
    prime_charge_mask = text.str.contains(prime_charge_rx, na=False)
    prime_cancel_mask = text.str.contains(prime_cancel_rx, na=False)
    order_cancel_mask = text.str.contains(order_cancel_rx, na=False)
    prime_info_mask = prime_mask & ~prime_charge_mask & ~prime_cancel_mask

    print(f"any 'prime' mention: {prime_mask.sum()}")
    print(f"prime+charge/billed/renew pattern: {prime_charge_mask.sum()}")
    print(f"prime+cancel pattern: {prime_cancel_mask.sum()}")
    print(f"overlap charge&cancel: {(prime_charge_mask & prime_cancel_mask).sum()}")
    print(f"prime mention, no charge/cancel pattern (residual 'info/other'): {prime_info_mask.sum()}")
    print(f"order-cancel pattern (broad): {order_cancel_mask.sum()}")
    print(f"order-cancel that also mentions prime: {(order_cancel_mask & prime_mask).sum()}")

    show("Prime charge examples", prime_charge_mask, first)
    show("Prime cancel examples", prime_cancel_mask, first)
    show("Prime residual ('info/other') examples", prime_info_mask, first)
    show("Order cancel (broad) examples", order_cancel_mask & ~prime_mask, first)

    print("\n" + "=" * 20 + " Q3: Refund vs Return " + "=" * 20)
    refund_rx = re.compile(r"\brefund\b", re.I)
    return_rx = re.compile(r"\breturn(ed|ing)?\b|\bwrong item\b|\bdamaged\b|\bdefective\b|\breplacement\b", re.I)
    refund_mask = text.str.contains(refund_rx, na=False)
    return_mask = text.str.contains(return_rx, na=False)
    print(f"refund pattern: {refund_mask.sum()}")
    print(f"return/damaged/wrong-item pattern: {return_mask.sum()}")
    print(f"overlap: {(refund_mask & return_mask).sum()} "
          f"({100*(refund_mask & return_mask).sum()/max(1,refund_mask.sum()):.1f}% of refund matches also match return)")
    show("Refund-only examples", refund_mask & ~return_mask, first)
    show("Return-only examples", return_mask & ~refund_mask, first)
    show("Both refund+return examples", refund_mask & return_mask, first)

    print("\n" + "=" * 20 + " Q4: Address / delivery redirect " + "=" * 20)
    address_rx = re.compile(
        r"(wrong address|change (my |the )?address|address is wrong|update (my |the )?address"
        r"|not at (this|that) address|left at (the )?wrong (address|place)|sent to (the )?wrong address"
        r"|deliver(ed)? to (the )?(wrong|incorrect) address)",
        re.I,
    )
    address_mask = text.str.contains(address_rx, na=False)
    show("Address examples", address_mask, first, n=6)

    print("\n" + "=" * 20 + " Q5: Multilingual content (heuristic) " + "=" * 20)
    es_pt_fr_rx = re.compile(
        r"\b(que|para|con|env[ií]o|pedido|gracias|hola|est[aá]|n[ãa]o|voc[eê]|obrigad|bonjour|merci|colis|livraison)\b",
        re.I,
    )
    non_en_mask = text.str.contains(es_pt_fr_rx, na=False)
    print(f"non-English marker matches: {non_en_mask.sum()} ({100*non_en_mask.sum()/len(first):.2f}%) -- heuristic, approximate")

    print("\n" + "=" * 20 + " Q6: Multi-intent overlap across core groups " + "=" * 20)
    core_patterns = {
        "delivery_delay": delay_rx,
        "order_tracking": tracking_rx,
        "refund": refund_rx,
        "return": return_rx,
        "prime_charge": prime_charge_rx,
        "order_cancel": order_cancel_rx,
        "address": address_rx,
    }
    hit_counts = pd.DataFrame({k: text.str.contains(v, na=False) for k, v in core_patterns.items()}).sum(axis=1)
    multi_hit = (hit_counts >= 2).sum()
    print(f"messages matching >=2 core patterns simultaneously: {multi_hit} ({100*multi_hit/len(first):.2f}%)")
    multi_examples = first[hit_counts >= 2]
    for _, r in multi_examples.sample(n=min(5, len(multi_examples)), random_state=3).iterrows():
        print(f"   [{r['_component']}] {r['text']}")

    print("\n" + "=" * 20 + " Q7: Generic complaints ride on other intents? " + "=" * 20)
    generic_rx = re.compile(r"(worst customer service|unhelpful|terrible service|awful service|rude|pathetic service)", re.I)
    generic_mask = text.str.contains(generic_rx, na=False)
    core_mask = pd.DataFrame({k: text.str.contains(v, na=False) for k, v in core_patterns.items()}).any(axis=1)
    generic_and_core = generic_mask & core_mask
    print(f"generic-complaint matches: {generic_mask.sum()}")
    print(f"of those, also matching a core intent pattern: {generic_and_core.sum()} "
          f"({100*generic_and_core.sum()/max(1,generic_mask.sum()):.1f}%)")
    show("Generic-complaint-only (no other core pattern) examples", generic_mask & ~core_mask, first)


if __name__ == "__main__":
    main()
