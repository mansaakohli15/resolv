"""
Resolv: Unified End-to-End Inference Pipeline.
Links Preprocessing -> Classification -> Retrieval -> Escalation -> Grounded Reply Generation.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.config import INTENT_TAXONOMY, HIGH_RISK_INTENTS
from src.data_utils import clean_text
from src.classifier.model import ResolvIntentClassifier
from src.retrieval.hybrid_search import ResolvHybridRetriever
from src.generation.generator import ResolvReplyGenerator
from src.escalation.policy import ResolvEscalationPolicy


class ResolvPipeline:
    """
    Production end-to-end Resolv pipeline for evidence-grounded customer support.
    """
    def __init__(
        self,
        classifier: Optional[ResolvIntentClassifier] = None,
        retriever: Optional[ResolvHybridRetriever] = None,
        generator: Optional[ResolvReplyGenerator] = None,
        escalation_policy: Optional[ResolvEscalationPolicy] = None,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator or ResolvReplyGenerator()
        self.escalation_policy = escalation_policy or ResolvEscalationPolicy()

    @classmethod
    def load_from_checkpoints(
        cls,
        classifier_path: str = "data/processed/intent_classifier.pkl",
        retriever_path: str = "data/processed/retrieval_index.pkl",
    ) -> "ResolvPipeline":
        """Loads fitted components from disk."""
        clf = ResolvIntentClassifier.load(classifier_path)
        ret = ResolvHybridRetriever.load(retriever_path)
        gen = ResolvReplyGenerator()
        esc = ResolvEscalationPolicy()
        return cls(classifier=clf, retriever=ret, generator=gen, escalation_policy=esc)

    def process_message(
        self,
        customer_message: str,
        top_k_evidence: int = 3,
        is_known_non_english: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes complete inference flow for an incoming customer message.
        """
        cleaned_query = clean_text(customer_message)
        
        # 1. Intent Classification
        clf_res = self.classifier.predict_with_confidence([customer_message])[0]
        predicted_intent = clf_res["intent"]
        intent_confidence = clf_res["confidence"]
        intent_margin = clf_res["margin"]
        
        # 2. Historical Evidence Retrieval
        evidence_cases = self.retriever.retrieve(
            query=customer_message,
            predicted_intent=predicted_intent,
            top_k=top_k_evidence,
        )
        
        # 3. Multi-Signal Escalation Decision
        esc_decision = self.escalation_policy.evaluate(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            intent_margin=intent_margin,
            evidence_cases=evidence_cases,
            is_known_non_english=is_known_non_english,
        )
        
        # 4. Grounded Reply Generation
        gen_output = self.generator.generate_reply(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_cases=evidence_cases,
        )
        
        # Format final structured response
        return {
            "customer_message": customer_message,
            "intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "intent_margin": intent_margin,
            "decision": esc_decision["decision"],
            "reason": esc_decision["primary_reason"],
            "reason_codes": esc_decision["reason_codes"],
            "suggested_reply": gen_output["reply"],
            "is_grounded": gen_output.get("grounded", True),
            "evidence": [
                {
                    "case_id": c["case_id"],
                    "similarity_score": c["score"],
                    "intent": c["intent"],
                    "historical_customer_text": c["customer_text"],
                    "historical_agent_reply": c["agent_reply"],
                }
                for c in evidence_cases
            ],
        }
