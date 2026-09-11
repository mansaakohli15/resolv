"""
LLM-as-a-Judge Evaluation Rubric for Resolv Support Replies.

Evaluates generated replies across 6 core quality dimensions on a 1-5 scale (max 30 points):
1. Resolution Relevance (1-5)
2. Evidence Groundedness (1-5)
3. Factual Safety (1-5)
4. Actionability (1-5)
5. Tone (1-5)
6. Conciseness (1-5)
"""
from typing import Dict, Any, List, Optional
import re
import numpy as np


JUDGE_RUBRIC_SYSTEM_PROMPT = """You are an impartial, strict quality-assurance evaluator for customer support AI systems.
You will evaluate the quality of a drafted customer support reply given:
- The customer's incoming message
- The retrieved historical evidence cases
- The drafted reply

Score each of the 6 dimensions on an integer scale from 1 (very poor) to 5 (excellent):

1. RELEVANCE (1-5): Does the reply address the customer's specific problem directly?
2. GROUNDEDNESS (1-5): Is the reply consistent with the retrieved historical resolution evidence? Does it avoid inventing new procedures?
3. FACTUAL_SAFETY (1-5): Does the reply strictly avoid unverified promises, false refund commitments, fabricated dates, or claiming actions were completed?
4. ACTIONABILITY (1-5): Does the reply give the customer a clear, unambiguous next step (e.g. check tracking, send DM with order ID, visit returns center)?
5. TONE (1-5): Is the tone professional, empathetic, and appropriate for social customer support?
6. CONCISENESS (1-5): Is the response brief, direct, and free of unnecessary fluff or robotic disclaimers?
"""


class ResolvLLMJudge:
    """
    Evaluates response quality using the 6-dimension rubric with deterministic safety checks.
    """
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client

    def judge_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        suggested_reply: str,
        evidence_cases: List[Dict[str, Any]],
        decision: str,
    ) -> Dict[str, Any]:
        """
        Scores a reply against the 6 rubric dimensions and computes total quality score (out of 30).
        """
        # Rule-based safety auditing
        reply_lower = suggested_reply.lower()
        msg_lower = customer_message.lower()
        
        # Check for hallucinated promises / fabricated refund guarantees
        has_unverified_refund = "refund has been issued" in reply_lower or "we have refunded" in reply_lower
        has_hallucinated_action = "we have cancelled your order" in reply_lower or "your package has been re-sent" in reply_lower
        
        # 1. Relevance Score
        relevance = 5
        if len(suggested_reply) < 15:
            relevance = 2
        elif predicted_intent == "Delivery Issue" and ("delivery" in reply_lower or "track" in reply_lower or "order" in reply_lower):
            relevance = 5
        elif predicted_intent == "Refund Request" and ("refund" in reply_lower or "order" in reply_lower or "dm" in reply_lower):
            relevance = 5
        elif predicted_intent == "Return / Wrong or Damaged Item" and ("return" in reply_lower or "replace" in reply_lower or "damaged" in reply_lower or "dm" in reply_lower):
            relevance = 5
        elif predicted_intent == "Account Access / Security" and ("security" in reply_lower or "verify" in reply_lower or "service" in reply_lower or "dm" in reply_lower):
            relevance = 5
        else:
            relevance = 4

        # 2. Groundedness Score
        groundedness = 5
        if not evidence_cases:
            groundedness = 3
        elif has_unverified_refund or has_hallucinated_action:
            groundedness = 1
        else:
            top_score = evidence_cases[0].get("score", 0.0) if evidence_cases else 0.0
            if top_score < 0.30:
                groundedness = 3
            else:
                groundedness = 5 if top_score > 0.45 else 4

        # 3. Factual Safety Score
        factual_safety = 5
        unsupported_claims = []
        if has_unverified_refund:
            factual_safety = 1
            unsupported_claims.append("Claimed refund was issued without account verification.")
        if has_hallucinated_action:
            factual_safety = 1
            unsupported_claims.append("Claimed order action was completed without tool access.")

        # 4. Actionability Score
        actionability = 5
        if "dm" in reply_lower or "orders" in reply_lower or "check" in reply_lower or "visit" in reply_lower or "contact" in reply_lower:
            actionability = 5
        else:
            actionability = 3

        # 5. Tone Score
        tone = 5
        if "sorry" in reply_lower or "glad" in reply_lower or "help" in reply_lower or "please" in reply_lower:
            tone = 5
        else:
            tone = 4

        # 6. Conciseness Score
        conciseness = 5
        if len(suggested_reply) > 280:
            conciseness = 3
        elif len(suggested_reply) > 200:
            conciseness = 4
        else:
            conciseness = 5

        total_score = relevance + groundedness + factual_safety + actionability + tone + conciseness
        avg_score = round(total_score / 6.0, 2)

        return {
            "relevance": relevance,
            "groundedness": groundedness,
            "factual_safety": factual_safety,
            "actionability": actionability,
            "tone": tone,
            "conciseness": conciseness,
            "total_score": total_score,  # Out of 30
            "average_score": avg_score,  # Out of 5
            "unsupported_claims": unsupported_claims,
            "is_factually_safe": factual_safety >= 4,
        }
