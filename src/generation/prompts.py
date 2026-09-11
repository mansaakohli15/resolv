"""
Prompts and templates for Evidence-Grounded Customer Support Reply Generation.
"""

GROUNDED_SYSTEM_PROMPT = """You are Resolv, an evidence-grounded customer support reply assistant for Amazon customer Twitter support.

Your primary objective is to draft a helpful, professional, and concise reply that is STRICTLY GROUNDED in historical support resolutions.

CRITICAL GUARDRAILS:
1. STRICT FACTUAL GROUNDING: Rely ONLY on the provided historical evidence. Do not invent policies, dates, delivery guarantees, prices, or refunds.
2. ACTION ACCURACY: If historical resolutions typically ask the customer to check tracking, visit a self-service link, or send account details via DM, instruct them accordingly. Never promise that an action has already been taken.
3. CONCISION & TONE: Keep replies under 280 characters (standard Twitter format). Maintain an empathetic, professional, and brand-appropriate tone.
4. UNCERTAINTY: If the evidence does not provide a clear resolution pattern, advise contacting support or sending a direct message with order details.
5. NO HALLUCINATED URLs: Use standard Amazon self-service references or prompt for a DM. Do not create fake URLs.
"""

GROUNDED_REPLY_USER_TEMPLATE = """CUSTOMER MESSAGE:
"{customer_message}"

PREDICTED INTENT:
{predicted_intent} (Confidence: {intent_confidence:.2f})

RETRIEVED HISTORICAL EVIDENCE:
{evidence_text}

Draft a concise, evidence-grounded Twitter support reply following the guidelines.
"""


def format_evidence_for_prompt(evidence_cases: list) -> str:
    """Formats retrieved evidence cases into a readable string for prompting."""
    if not evidence_cases:
        return "No historical evidence available."
    
    formatted = []
    for idx, case in enumerate(evidence_cases, start=1):
        formatted.append(
            f"Case #{idx} (ID: {case.get('case_id')}, Similarity: {case.get('score', 0.0):.2f}, Intent: {case.get('intent')}):\n"
            f"  Customer asked: \"{case.get('customer_text', '')}\"\n"
            f"  Brand resolution: \"{case.get('agent_reply', '')}\""
        )
    return "\n\n".join(formatted)
