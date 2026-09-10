"""
Milestone 3, Step 1 — build the AmazonHelp conversation dataset.

Reuses Milestone 1 cached artifacts (no CSV reparse):
  data/raw/twcs_cache.pkl
  data/raw/twcs_components.pkl

Filtering rules (documented, applied in order, each drop counted):
  R1. Component must contain >=1 reply from AmazonHelp (inbound=False, author_id=="AmazonHelp").
      -> defines the candidate AmazonHelp component set.
  R2. Component must contain >=1 customer (inbound=True) tweet.
      -> drops brand-only components (e.g. AmazonHelp replying to itself/another brand only).
  R3. Component must NOT touch any OTHER brand account (any other inbound=False author_id
      besides AmazonHelp). Cross-brand threads are dropped entirely, not trimmed, because we
      can't reliably tell if AmazonHelp's reply was actually responding to the same issue.
  R4. Rows with empty/whitespace-only text are dropped (individual rows, not whole conversations).
  R5. created_at must parse to a valid timestamp; unparseable rows are dropped (individual rows).

Nothing else is discarded. No dedup of repeated customer phrasing is applied here — that is a
downstream modeling decision, not a data-quality one, so raw near-duplicates are preserved and
left for later milestones to decide how to treat.

Output: data/processed/amazonhelp_conversations.pkl
  One row per tweet, columns:
    tweet_id, _component, author_id, inbound, created_at (parsed), text,
    turn_index (0-based chronological position within its conversation)

Run: python scripts/build_brand_dataset.py
"""
import pandas as pd

BRAND = "AmazonHelp"
OUT_PATH = "data/processed/amazonhelp_conversations.pkl"


def main():
    print("Loading cached artifacts ...")
    df = pd.read_pickle("data/raw/twcs_cache.pkl")
    comp = pd.read_pickle("data/raw/twcs_components.pkl")
    df = df.merge(comp, on="tweet_id")

    n_start = len(df)
    n_components_start = df["_component"].nunique()

    # R1: components where AmazonHelp replied at least once
    amazon_reply_components = set(
        df.loc[(df["inbound"] == False) & (df["author_id"] == BRAND), "_component"]
    )
    df1 = df[df["_component"].isin(amazon_reply_components)]
    print(f"R1 (AmazonHelp reply present): {len(amazon_reply_components)} components, {len(df1)} rows")

    # R2: components with >=1 customer tweet
    has_customer = set(df1.loc[df1["inbound"] == True, "_component"])
    df2 = df1[df1["_component"].isin(has_customer)]
    dropped_r2 = len(amazon_reply_components) - len(has_customer)
    print(f"R2 (has customer tweet): dropped {dropped_r2} components -> {df2['_component'].nunique()} remain")

    # R3: drop components touching any OTHER brand account
    brand_authors_per_component = (
        df2[df2["inbound"] == False].groupby("_component")["author_id"].nunique()
    )
    clean_components = set(brand_authors_per_component[brand_authors_per_component == 1].index)
    dropped_r3 = df2["_component"].nunique() - len(clean_components)
    df3 = df2[df2["_component"].isin(clean_components)]
    print(f"R3 (single-brand only): dropped {dropped_r3} cross-brand components -> {df3['_component'].nunique()} remain")

    # R4: empty/whitespace text rows
    non_empty_mask = df3["text"].astype(str).str.strip() != ""
    dropped_r4 = (~non_empty_mask).sum()
    df4 = df3[non_empty_mask]
    print(f"R4 (non-empty text): dropped {dropped_r4} rows")

    # R5: parseable timestamp
    parsed_ts = pd.to_datetime(df4["created_at"], errors="coerce", format="%a %b %d %H:%M:%S %z %Y")
    dropped_r5 = parsed_ts.isna().sum()
    df5 = df4.copy()
    df5["created_at"] = parsed_ts
    df5 = df5[parsed_ts.notna()]
    print(f"R5 (parseable created_at): dropped {dropped_r5} rows")

    # chronological ordering within each conversation
    df5 = df5.sort_values(["_component", "created_at", "tweet_id"])
    df5["turn_index"] = df5.groupby("_component").cumcount()

    final = df5[["tweet_id", "_component", "author_id", "inbound", "created_at", "text", "turn_index"]].reset_index(drop=True)

    print("\n=== SUMMARY ===")
    print(f"Rows: {n_start} -> {len(final)}")
    print(f"Components: {n_components_start} -> {final['_component'].nunique()}")
    print(f"Customer rows: {(final['inbound']==True).sum()}, AmazonHelp reply rows: {(final['inbound']==False).sum()}")

    import os
    os.makedirs("data/processed", exist_ok=True)
    final.to_pickle(OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
