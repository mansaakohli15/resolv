"""
Resolv - Core Configuration & Taxonomy Constants
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and Artifact Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "amazonhelp_conversations.pkl"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GOLDEN_SET_PATH = EVALUATION_DIR / "golden_set.csv"
SPLIT_ASSIGNMENT_PATH = EVALUATION_DIR / "split_assignment.pkl"
SPLIT_STATS_PATH = EVALUATION_DIR / "split_stats.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 8 Final In-Scope Intents
INTENT_TAXONOMY = [
    "Delivery Issue",
    "Address / Delivery Redirect Issue",
    "Refund Request",
    "Return / Wrong or Damaged Item",
    "Prime Membership Billing/Cancellation",
    "Order Cancellation Request",
    "Account Access / Security",
    "Unauthorized / Non-Prime Charge",
]

# Non-intent classes in Golden Evaluation Set
OUT_OF_SCOPE_CLASSES = [
    "Other / Out of Scope",
    "Unsupported Language",
    "Ambiguous",
]

ALL_EVAL_CLASSES = INTENT_TAXONOMY + OUT_OF_SCOPE_CLASSES

# High-Risk Intents requiring mandatory human escalation
HIGH_RISK_INTENTS = [
    "Account Access / Security",
    "Unauthorized / Non-Prime Charge",
]

RANDOM_SEED = 42
