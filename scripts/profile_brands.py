"""
Milestone 2 — brand profiling for Resolv brand selection.
Reuses the cached artifacts from Milestone 1 (scripts/inspect_dataset.py):
  data/raw/twcs_cache.pkl        -> full tweet table
  data/raw/twcs_components.pkl   -> tweet_id -> conversation component id
Does NOT reload/re-parse the raw 2.8M-row CSV.

Run: python scripts/profile_brands.py
Outputs:
  evaluation/brand_profile.csv
"""
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
CLUSTER_SAMPLE_MAX = 3000   # per-brand cap for the diversity clustering step, for speed
N_CLUSTERS = 8

# Top 10 candidate brands by conversation-component count, from Milestone 1 recon.
CANDIDATE_BRANDS = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "AmericanAir",
    "Delta", "comcastcares", "TMobileHelp", "SouthwestAir", "Ask_Spectrum",
]

MENTION_RE = re.compile(r"@\S+")
URL_RE = re.compile(r"https?://\S+")


def normalize_text(t):
    t = MENTION_RE.sub("", str(t))
    t = URL_RE.sub("", t)
    return " ".join(t.lower().split())


def main():
    print("Loading cached artifacts ...")
    df = pd.read_pickle("data/raw/twcs_cache.pkl")
    comp = pd.read_pickle("data/raw/twcs_components.pkl")
    df = df.merge(comp, on="tweet_id")

    comp_sizes = df.groupby("_component").size()

    rows = []
    for brand in CANDIDATE_BRANDS:
        brand_reply_mask = (df["inbound"] == False) & (df["author_id"] == brand)
        components_touching = set(df.loc[brand_reply_mask, "_component"])
        n_components = len(components_touching)

        df_brand = df[df["_component"].isin(components_touching)]
        customer_tweets = df_brand[df_brand["inbound"] == True]
        this_brand_replies = df_brand[(df_brand["inbound"] == False) & (df_brand["author_id"] == brand)]

        # usable = component has >=1 customer tweet AND >=1 reply from THIS brand
        has_customer = customer_tweets.groupby("_component").size()
        has_this_brand_reply = this_brand_replies.groupby("_component").size()
        usable_components = set(has_customer.index) & set(has_this_brand_reply.index)
        n_usable = len(usable_components)
        pct_with_brand_response = round(100 * n_usable / n_components, 2) if n_components else 0.0

        sizes = comp_sizes.reindex(list(components_touching))
        median_len = float(sizes.median())
        mean_len = float(sizes.mean())
        pct_multi_turn = round(100 * (sizes > 2).sum() / n_components, 2) if n_components else 0.0

        # duplicate/noise rate among customer tweets addressed to this brand (normalized text)
        norm_texts = customer_tweets["text"].map(normalize_text)
        n_customer_msgs = len(norm_texts)
        n_distinct_customer_msgs = norm_texts.nunique()
        dup_rate = round(100 * (1 - n_distinct_customer_msgs / n_customer_msgs), 2) if n_customer_msgs else 0.0

        # rough diversity: TF-IDF + KMeans on a capped sample of distinct customer messages
        distinct_texts = norm_texts.drop_duplicates()
        distinct_texts = distinct_texts[distinct_texts.str.len() > 0]
        sample = distinct_texts.sample(
            n=min(CLUSTER_SAMPLE_MAX, len(distinct_texts)), random_state=RANDOM_STATE
        )
        diversity_note = ""
        silhouette = np.nan
        cluster_balance = np.nan
        if len(sample) >= N_CLUSTERS * 5:
            vec = TfidfVectorizer(max_features=3000, min_df=3, stop_words="english")
            X = vec.fit_transform(sample)
            if X.shape[1] >= N_CLUSTERS:
                km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=5)
                labels = km.fit_predict(X)
                try:
                    silhouette = round(float(silhouette_score(X, labels, sample_size=min(1500, X.shape[0]),
                                                               random_state=RANDOM_STATE)), 4)
                except Exception as e:
                    diversity_note = f"silhouette failed: {e}"
                counts = pd.Series(labels).value_counts(normalize=True)
                # 1.0 = perfectly balanced clusters (max diversity signal), lower = one dominant topic
                cluster_balance = round(float(1 - counts.max()), 4)
            else:
                diversity_note = "too few TF-IDF features after min_df filter"
        else:
            diversity_note = "sample too small for reliable clustering"

        rows.append({
            "brand": brand,
            "n_conversation_components": n_components,
            "n_customer_tweets": len(customer_tweets),
            "n_brand_replies": len(this_brand_replies),
            "pct_conversations_with_brand_response": pct_with_brand_response,
            "median_conversation_length": median_len,
            "mean_conversation_length": round(mean_len, 3),
            "pct_multi_turn_conversations_gt2": pct_multi_turn,
            "n_usable_support_cases": n_usable,
            "n_customer_messages_total": n_customer_msgs,
            "n_distinct_customer_messages": n_distinct_customer_msgs,
            "duplicate_customer_msg_rate_pct": dup_rate,
            "diversity_kmeans_k": N_CLUSTERS if not diversity_note else None,
            "diversity_cluster_sample_size": len(sample),
            "diversity_silhouette_APPROX": silhouette,
            "diversity_cluster_balance_APPROX": cluster_balance,
            "diversity_note": diversity_note,
        })
        print(f"  done: {brand} ({n_components} components)")

    out = pd.DataFrame(rows).sort_values("n_conversation_components", ascending=False)
    out.to_csv("evaluation/brand_profile.csv", index=False)
    print("\nSaved evaluation/brand_profile.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
