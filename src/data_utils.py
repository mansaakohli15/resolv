"""
Data loading and preprocessing utilities for Resolv.
"""
import re
import pandas as pd
from typing import Tuple, Optional
from src.config import (
    PROCESSED_DATA_PATH,
    SPLIT_ASSIGNMENT_PATH,
    GOLDEN_SET_PATH,
    INTENT_TAXONOMY,
)

MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
NON_ALPHANUMERIC_RE = re.compile(r"[^a-zA-Z0-9\s$.,!?'-]")

# Mapping heuristic regex intent names in split_assignment to standard taxonomy names
HEURISTIC_TO_CANONICAL = {
    "delivery_issue": "Delivery Issue",
    "address_redirect": "Address / Delivery Redirect Issue",
    "refund_request": "Refund Request",
    "return_wrong_damaged": "Return / Wrong or Damaged Item",
    "prime_billing_cancellation": "Prime Membership Billing/Cancellation",
    "order_cancellation": "Order Cancellation Request",
    "account_access_security": "Account Access / Security",
    "unauthorized_nonprime_charge": "Unauthorized / Non-Prime Charge",
    "unmatched_other": "Other / Out of Scope",
    "non_english_out_of_scope": "Unsupported Language",
}

CANONICAL_TO_HEURISTIC = {v: k for k, v in HEURISTIC_TO_CANONICAL.items()}


def clean_text(text: str) -> str:
    """
    Normalizes tweet text for modeling:
    - Removes @handles and URLs
    - Normalizes excessive whitespace
    - Preserves key currency/punctuation symbols for context
    """
    if not isinstance(text, str):
        return ""
    t = MENTION_RE.sub("", text)
    t = URL_RE.sub("", t)
    t = NON_ALPHANUMERIC_RE.sub(" ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


def load_conversations() -> pd.DataFrame:
    """Loads the processed AmazonHelp conversation dataset."""
    return pd.read_pickle(PROCESSED_DATA_PATH)


def load_split_assignments() -> pd.DataFrame:
    """Loads the component-level split assignment dataframe."""
    return pd.read_pickle(SPLIT_ASSIGNMENT_PATH)


def load_golden_set() -> pd.DataFrame:
    """Loads the 196-row hand-annotated golden evaluation benchmark."""
    return pd.read_csv(GOLDEN_SET_PATH)


def get_training_data() -> pd.DataFrame:
    """
    Extracts customer starting messages from the train_retrieval split ONLY.
    Strictly guarantees zero leakage from validation, test, or golden sets.
    """
    convs = load_conversations()
    splits = load_split_assignments()
    
    # Filter only train_retrieval components
    train_components = set(splits.loc[splits["split"] == "train_retrieval", "_component"])
    
    # Get first customer message per component
    customer_msgs = convs[(convs["inbound"] == True) & (convs["_component"].isin(train_components))].copy()
    first_msgs = customer_msgs.sort_values("turn_index").groupby("_component").first().reset_index()
    
    # Merge with heuristic tag (which serves as weak training labels for the in-scope intents)
    merged = first_msgs.merge(splits[["_component", "heuristic_intent"]], on="_component")
    
    # Map heuristic tag to canonical intent
    merged["intent"] = merged["heuristic_intent"].map(HEURISTIC_TO_CANONICAL)
    merged["cleaned_text"] = merged["text"].map(clean_text)
    
    # Keep only in-scope taxonomy classes for classifier training
    train_df = merged[merged["intent"].isin(INTENT_TAXONOMY)].copy()
    return train_df


def get_validation_data() -> pd.DataFrame:
    """
    Extracts validation split instances for threshold tuning and intermediate validation.
    """
    convs = load_conversations()
    splits = load_split_assignments()
    
    val_components = set(splits.loc[splits["split"] == "validation", "_component"])
    customer_msgs = convs[(convs["inbound"] == True) & (convs["_component"].isin(val_components))].copy()
    first_msgs = customer_msgs.sort_values("turn_index").groupby("_component").first().reset_index()
    
    merged = first_msgs.merge(splits[["_component", "heuristic_intent"]], on="_component")
    merged["intent"] = merged["heuristic_intent"].map(HEURISTIC_TO_CANONICAL)
    merged["cleaned_text"] = merged["text"].map(clean_text)
    
    val_df = merged[merged["intent"].isin(INTENT_TAXONOMY)].copy()
    return val_df
