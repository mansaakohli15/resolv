"""
Milestone 1 — dataset reconnaissance for Resolv.
Read-only inspection of the Customer Support on Twitter dataset.
Does NOT clean, filter, or transform the data — only measures it.

Run: python scripts/inspect_dataset.py
"""
import pandas as pd

PATH = "data/raw/twcs/twcs.csv"

print("Loading twcs.csv ...")
df = pd.read_csv(PATH, dtype={"tweet_id": "int64", "author_id": "str"})

print("\n=== SHAPE ===")
print(df.shape)

print("\n=== DTYPES (as loaded) ===")
print(df.dtypes)

print("\n=== NULL COUNTS ===")
print(df.isnull().sum())

print("\n=== DUPLICATE tweet_id COUNT ===")
print(df["tweet_id"].duplicated().sum())

print("\n=== FULLY DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== inbound VALUE COUNTS ===")
print(df["inbound"].value_counts(dropna=False))

print("\n=== EMPTY/WHITESPACE-ONLY text ROWS ===")
print((df["text"].astype(str).str.strip() == "").sum())

print("\n=== response_tweet_id / in_response_to_tweet_id NULL RATES ===")
print("response_tweet_id null:", df["response_tweet_id"].isnull().sum())
print("in_response_to_tweet_id null:", df["in_response_to_tweet_id"].isnull().sum())

# author_id for inbound=False rows -> candidate brand accounts
brand_rows = df[df["inbound"] == False]
print("\n=== TOP 25 author_id AMONG inbound=False (candidate brands) ===")
print(brand_rows["author_id"].value_counts().head(25))

print("\n=== inbound=True author_id sample (should look like numeric customer IDs) ===")
print(df[df["inbound"] == True]["author_id"].head(10).tolist())

# root tweets = inbound customer tweets that are NOT a reply to anything (in_response_to_tweet_id null)
roots = df[(df["inbound"] == True) & (df["in_response_to_tweet_id"].isnull())]
print("\n=== ROOT (conversation-starting) inbound tweets ===")
print(len(roots))

df.to_pickle("data/raw/twcs_cache.pkl")
print("\nCached full dataframe to data/raw/twcs_cache.pkl for faster reuse in later milestones.")
