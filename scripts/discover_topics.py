"""
Milestone 3, Step 2 — discover recurring topics in AmazonHelp customer messages.
No LLM used here — purely frequency/TF-IDF/clustering based discovery.

Input: data/processed/amazonhelp_conversations.pkl (from build_brand_dataset.py)
Outputs:
  evaluation/topic_terms.csv          top unigrams/bigrams overall
  evaluation/topic_clusters.csv       per-cluster size, top terms, sample messages
  evaluation/message_length_stats.txt

Run: python scripts/discover_topics.py
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans

RANDOM_STATE = 42
N_CLUSTERS = 14
MAX_FEATURES = 6000

MENTION_RE = re.compile(r"@\S+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def normalize(t):
    t = MENTION_RE.sub(" ", str(t))
    t = URL_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip().lower()
    return t


def main():
    df = pd.read_pickle("data/processed/amazonhelp_conversations.pkl")
    # first customer message per conversation = the initiating issue statement (most useful for intent discovery)
    customer = df[df["inbound"] == True].copy()
    first_msgs = customer.sort_values("turn_index").groupby("_component").first().reset_index()
    print(f"Customer rows total: {len(customer)}; first-message-per-conversation set: {len(first_msgs)}")

    first_msgs["norm_text"] = first_msgs["text"].map(normalize)
    first_msgs["word_count"] = first_msgs["norm_text"].str.split().map(len)

    # --- message length stats ---
    stats = first_msgs["word_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    with open("evaluation/message_length_stats.txt", "w") as f:
        f.write("Word-count stats for first customer message per AmazonHelp conversation:\n")
        f.write(stats.to_string())
    print("\n=== message length stats (words) ===")
    print(stats)

    # --- top unigrams/bigrams (frequency, not TF-IDF) ---
    cv = CountVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english", min_df=5)
    X_counts = cv.fit_transform(first_msgs["norm_text"])
    freqs = np.asarray(X_counts.sum(axis=0)).ravel()
    terms = cv.get_feature_names_out()
    top_terms = pd.DataFrame({"term": terms, "count": freqs}).sort_values("count", ascending=False)
    top_terms.to_csv("evaluation/topic_terms.csv", index=False)
    print("\n=== top 30 terms/bigrams ===")
    print(top_terms.head(30).to_string(index=False))

    # --- TF-IDF + MiniBatchKMeans clustering ---
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), stop_words="english", min_df=5)
    X = tfidf.fit_transform(first_msgs["norm_text"])
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10, batch_size=2048)
    labels = km.fit_predict(X)
    first_msgs["cluster"] = labels

    vocab = np.array(tfidf.get_feature_names_out())
    rows = []
    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]
        size = len(idx)
        centroid = km.cluster_centers_[c]
        top_idx = centroid.argsort()[::-1][:12]
        top_words = ", ".join(vocab[top_idx])

        # representative messages: closest to centroid among this cluster's points
        sub_X = X[idx]
        dists = np.asarray(sub_X.multiply(centroid).sum(axis=1)).ravel()  # cosine-ish similarity proxy
        order = np.argsort(dists)[::-1][:5]
        sample_msgs = first_msgs.iloc[idx[order]]["text"].tolist()

        rows.append({
            "cluster": c,
            "size": size,
            "pct_of_total": round(100 * size / len(first_msgs), 2),
            "top_terms": top_words,
            "sample_1": sample_msgs[0] if len(sample_msgs) > 0 else "",
            "sample_2": sample_msgs[1] if len(sample_msgs) > 1 else "",
            "sample_3": sample_msgs[2] if len(sample_msgs) > 2 else "",
            "sample_4": sample_msgs[3] if len(sample_msgs) > 3 else "",
            "sample_5": sample_msgs[4] if len(sample_msgs) > 4 else "",
        })

    clusters_df = pd.DataFrame(rows).sort_values("size", ascending=False)
    clusters_df.to_csv("evaluation/topic_clusters.csv", index=False)
    print("\n=== clusters (size, top terms) ===")
    print(clusters_df[["cluster", "size", "pct_of_total", "top_terms"]].to_string(index=False))

    first_msgs[["tweet_id", "_component", "cluster"]].to_pickle("data/processed/first_msg_clusters.pkl")
    print("\nSaved evaluation/topic_terms.csv, evaluation/topic_clusters.csv, data/processed/first_msg_clusters.pkl")


if __name__ == "__main__":
    main()
