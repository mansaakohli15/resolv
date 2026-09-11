"""
Unit tests for Resolv Intent Classifier, Escalation Policy, and End-to-End Pipeline.
"""
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTENT_TAXONOMY, HIGH_RISK_INTENTS
from src.classifier.model import ResolvIntentClassifier
from src.escalation.policy import ResolvEscalationPolicy
from src.pipeline import ResolvPipeline


class TestResolvPipeline(unittest.TestCase):
    def test_classifier_output_schema(self):
        """Verify that classifier outputs correct intent labels and calibrated probabilities."""
        clf = ResolvIntentClassifier.load("data/processed/intent_classifier.pkl")
        preds = clf.predict_with_confidence(["Where is my late package?"])
        
        self.assertEqual(len(preds), 1)
        p = preds[0]
        self.assertIn(p["intent"], INTENT_TAXONOMY)
        self.assertTrue(0.0 <= p["confidence"] <= 1.0)
        self.assertTrue(0.0 <= p["margin"] <= 1.0)
        self.assertIn("all_probabilities", p)
        self.assertEqual(len(p["all_probabilities"]), len(INTENT_TAXONOMY))

    def test_escalation_security_intents(self):
        """Verify that high-risk account and security intents always trigger escalation."""
        policy = ResolvEscalationPolicy()
        
        for intent in HIGH_RISK_INTENTS:
            res = policy.evaluate(
                customer_message="I have an issue.",
                predicted_intent=intent,
                intent_confidence=0.99,
                intent_margin=0.50,
                evidence_cases=[{"score": 0.80, "case_id": "case_1"}],
            )
            self.assertEqual(res["decision"], "ESCALATE")
            self.assertIn("HIGH_RISK_INTENT", res["reason_codes"])

    def test_escalation_unsupported_language(self):
        """Verify that non-English messages trigger language escalation."""
        policy = ResolvEscalationPolicy()
        res = policy.evaluate(
            customer_message="Hola, necesito ayuda con mi pedido por favor.",
            predicted_intent="Delivery Issue",
            intent_confidence=0.95,
            intent_margin=0.40,
            evidence_cases=[{"score": 0.80, "case_id": "case_1"}],
        )
        self.assertEqual(res["decision"], "ESCALATE")
        self.assertIn("UNSUPPORTED_LANGUAGE", res["reason_codes"])

    def test_pipeline_end_to_end(self):
        """Verify end-to-end pipeline execution and JSON response structure."""
        pipeline = ResolvPipeline.load_from_checkpoints()
        res = pipeline.process_message("My order was supposed to arrive today but tracking is stuck.")
        
        self.assertIn(res["intent"], INTENT_TAXONOMY)
        self.assertIn(res["decision"], ["AUTO_HANDLE", "ESCALATE"])
        self.assertIsInstance(res["reason"], str)
        self.assertTrue(len(res["suggested_reply"]) > 10)
        self.assertTrue(len(res["evidence"]) > 0)
        self.assertIn("case_id", res["evidence"][0])
        self.assertIn("similarity_score", res["evidence"][0])


if __name__ == "__main__":
    unittest.main()
