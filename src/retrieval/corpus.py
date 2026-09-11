"""
Build and manage the Resolv Evidence Retrieval Corpus.
Constructed STRICTLY from the train_retrieval split (zero leakage from validation/test/golden).
Optimized using vectorized groupby operations.
"""
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import (
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    INTENT_TAXONOMY,
)
from src.data_utils import clean_text, HEURISTIC_TO_CANONICAL


def build_evidence_corpus(min_reply_len: int = 15) -> pd.DataFrame:
    """
    Extracts high-quality (customer_query, brand_reply) resolution pairs from train_retrieval split.
    
    Returns DataFrame with columns:
    - case_id: Unique identifier for the evidence instance
    - component_id: Conversation component id
    - customer_text: Original customer tweet
    - cleaned_customer_text: Normalized customer text for indexing
    - agent_reply: AmazonHelp's historical response
    - intent: Heuristic/canonical support intent of the issue
    """
    df = pd.read_pickle(PROCESSED_DATA_PATH)
    splits = pd.read_pickle(SPLIT_ASSIGNMENT_PATH)
    
    # Strictly filter to train_retrieval split
    train_components = set(splits.loc[splits["split"] == "train_retrieval", "_component"])
    train_df = df[df["_component"].isin(train_components)].copy()
    
    # Sort chronologically by component and turn index
    train_df = train_df.sort_values(["_component", "turn_index"])
    
    # Vectorized extraction of first customer tweet per component
    cust_df = (
        train_df[train_df["inbound"] == True]
        .groupby("_component", as_index=False)
        .first()
    )
    
    # Vectorized extraction of first AmazonHelp tweet per component
    agent_df = (
        train_df[(train_df["inbound"] == False) & (train_df["author_id"] == "AmazonHelp")]
        .groupby("_component", as_index=False)
        .first()
    )
    
    # Merge on _component
    merged = cust_df.merge(agent_df, on="_component", suffixes=("_cust", "_agent"))
    
    # Ensure brand response came after customer message
    merged = merged[merged["turn_index_agent"] > merged["turn_index_cust"]].copy()
    
    # Filter on length
    merged["cust_text"] = merged["text_cust"].astype(str).str.strip()
    merged["agent_text"] = merged["text_agent"].astype(str).str.strip()
    
    merged = merged[
        (merged["agent_text"].str.len() >= min_reply_len) & 
        (merged["cust_text"].str.len() >= 5)
    ].copy()
    
    # Map intent
    comp_to_intent = splits.set_index("_component")["heuristic_intent"].to_dict()
    merged["heuristic_intent"] = merged["_component"].map(comp_to_intent).fillna("unmatched_other")
    merged["intent"] = merged["heuristic_intent"].map(HEURISTIC_TO_CANONICAL).fillna("Other / Out of Scope")
    
    merged["cleaned_customer_text"] = merged["cust_text"].map(clean_text)
    merged["case_id"] = merged["_component"].apply(lambda cid: f"case_{cid}")
    merged["component_id"] = merged["_component"]
    merged["customer_text"] = merged["cust_text"]
    merged["agent_reply"] = merged["agent_text"]
    
    corpus_df = merged[[
        "case_id",
        "component_id",
        "customer_text",
        "cleaned_customer_text",
        "agent_reply",
        "intent",
    ]].reset_index(drop=True)
    
    return corpus_df
