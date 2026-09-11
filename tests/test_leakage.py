"""
Unit test to strictly verify zero data leakage across splits, retrieval corpus, and golden benchmark.
"""
import unittest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import load_split_assignments, load_golden_set
from src.retrieval.corpus import build_evidence_corpus


class TestDataLeakage(unittest.TestCase):
    def test_zero_split_leakage(self):
        """Verify that no conversation component exists in more than one split."""
        splits = load_split_assignments()
        self.assertEqual(splits["_component"].nunique(), len(splits), "Duplicate components found in split assignment!")
        self.assertEqual(splits.groupby("_component")["split"].nunique().max(), 1, "Component assigned to multiple splits!")

    def test_golden_set_leakage(self):
        """Verify that golden evaluation set components never appear in train_retrieval."""
        splits = load_split_assignments()
        golden_df = load_golden_set()
        
        train_comps = set(splits.loc[splits["split"] == "train_retrieval", "_component"])
        val_comps = set(splits.loc[splits["split"] == "validation", "_component"])
        golden_comps = set(golden_df["component_id"])
        
        self.assertEqual(len(golden_comps & train_comps), 0, "Golden evaluation set leaked into train_retrieval!")
        self.assertEqual(len(golden_comps & val_comps), 0, "Golden evaluation set leaked into validation!")

    def test_retrieval_corpus_leakage(self):
        """Verify that the evidence retrieval index contains zero test or golden components."""
        splits = load_split_assignments()
        golden_df = load_golden_set()
        corpus_df = build_evidence_corpus()
        
        test_comps = set(splits.loc[splits["split"] == "test", "_component"])
        golden_comps = set(golden_df["component_id"])
        corpus_comps = set(corpus_df["component_id"])
        
        self.assertEqual(len(corpus_comps & test_comps), 0, "Test components leaked into evidence corpus!")
        self.assertEqual(len(corpus_comps & golden_comps), 0, "Golden components leaked into evidence corpus!")


if __name__ == "__main__":
    unittest.main()
