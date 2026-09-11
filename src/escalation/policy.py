"""
Multi-Signal Risk-Aware Escalation Policy for Resolv.

Evaluates multiple risk and safety signals to decide:
- AUTO_HANDLE: Safe to provide automated grounded response.
- ESCALATE: Route to human support agent with explicit reason.
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from src.config import HIGH_RISK_INTENTS, INTENT_TAXONOMY

# Common non-English markers (Spanish, Portuguese, French, German, Japanese, etc.)
NON_EN_WORDS_RE = re.compile(
    r"\b(que|para|con|env[ií]o|pedido|gracias|hola|est[aá]|n[ãa]o|voc[eê]|obrigad|bonjour|merci|colis|livraison|bitte|danke|bestellung|nicht|por favor|ayuda)\b",
    re.I,
)

# Explicit customer requests for human assistance
HUMAN_AGENT_RE = re.compile(
    r"(speak to a human|talk to a human|real person|human agent|live agent|customer representative|speak to someone|talk to someone|representative|operator)",
    re.I,
)

# Non-ASCII / CJK character detection
NON_LATIN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]")


class ResolvEscalationPolicy:
    """
    Transparent, explainable multi-signal escalation engine.
    """
    def __init__(
        self,
        min_intent_confidence: float = 0.52,
        min_evidence_similarity: float = 0.28,
        min_margin: float = 0.08,
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_evidence_similarity = min_evidence_similarity
        self.min_margin = min_margin

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        intent_margin: float,
        evidence_cases: List[Dict[str, Any]],
        is_known_non_english: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates safety and risk signals to return AUTO_HANDLE or ESCALATE with reason.
        """
        reasons = []
        reason_codes = []

        # 1. Unsupported Language Check
        if is_known_non_english or NON_EN_WORDS_RE.search(customer_message) or NON_LATIN_RE.search(customer_message):
            reason_codes.append("UNSUPPORTED_LANGUAGE")
            reasons.append("Unsupported language detected; requires specialized language support agent.")

        # 2. Explicit Human Request Check
        if HUMAN_AGENT_RE.search(customer_message):
            reason_codes.append("EXPLICIT_HUMAN_REQUEST")
            reasons.append("Customer explicitly requested human agent assistance.")

        # 3. High-Risk Intent Check (Account Security, Unauthorized Charges)
        if predicted_intent in HIGH_RISK_INTENTS:
            reason_codes.append("HIGH_RISK_INTENT")
            reasons.append(
                f"Intent '{predicted_intent}' involves account security or financial transactions requiring identity verification."
            )

        # 4. Out of Scope or Ambiguous Intent Check
        if predicted_intent not in INTENT_TAXONOMY:
            reason_codes.append("OUT_OF_SCOPE_INTENT")
            reasons.append(f"Customer query falls outside the supported 8-intent customer support taxonomy.")

        # 5. Low Intent Confidence / Margin Check
        if intent_confidence < self.min_intent_confidence:
            reason_codes.append("LOW_INTENT_CONFIDENCE")
            reasons.append(
                f"Classifier confidence ({intent_confidence:.2f}) is below minimum safety threshold ({self.min_intent_confidence:.2f})."
            )
        elif intent_margin < self.min_margin:
            reason_codes.append("AMBIGUOUS_INTENT_MARGIN")
            reasons.append(
                f"Ambiguous intent margin ({intent_margin:.2f}) between top classes."
            )

        # 6. Evidence Sufficiency Check
        if not evidence_cases or len(evidence_cases) == 0:
            reason_codes.append("NO_HISTORICAL_EVIDENCE")
            reasons.append("No relevant historical resolution examples were found.")
        else:
            top_score = evidence_cases[0].get("score", 0.0)
            if top_score < self.min_evidence_similarity:
                reason_codes.append("LOW_RETRIEVAL_CONFIDENCE")
                reasons.append(
                    f"Top historical evidence relevance score ({top_score:.2f}) is below minimum evidence threshold ({self.min_evidence_similarity:.2f})."
                )

        # Decision synthesis
        if len(reason_codes) > 0:
            decision = "ESCALATE"
            primary_reason = reasons[0]
        else:
            decision = "AUTO_HANDLE"
            primary_reason = f"High intent confidence ({intent_confidence:.2f}) and verified grounded historical evidence ({evidence_cases[0].get('score', 0.0):.2f})."

        return {
            "decision": decision,
            "primary_reason": primary_reason,
            "all_reasons": reasons,
            "reason_codes": reason_codes,
            "is_escalated": decision == "ESCALATE",
        }
