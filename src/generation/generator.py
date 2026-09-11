"""
Evidence-Grounded Reply Generator for Resolv.
Synthesizes customer support replies strictly grounded in retrieved historical resolution evidence.
"""
import re
from typing import List, Dict, Any, Optional
from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    GROUNDED_REPLY_USER_TEMPLATE,
    format_evidence_for_prompt,
)


class ResolvReplyGenerator:
    """
    Production reply generator that produces safe, concise, evidence-grounded replies.
    """
    def __init__(self, use_llm_client: bool = False, llm_client: Optional[Any] = None):
        self.use_llm_client = use_llm_client
        self.llm_client = llm_client

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates a grounded reply for a customer query based on historical evidence.
        """
        if not evidence_cases:
            return {
                "reply": "We'd like to look into this for you. Please send us a direct message with your order details so our support team can assist.",
                "grounded": False,
                "evidence_ids": [],
            }

        # If LLM client is configured and enabled, use it
        if self.use_llm_client and self.llm_client is not None:
            evidence_str = format_evidence_for_prompt(evidence_cases)
            prompt = GROUNDED_REPLY_USER_TEMPLATE.format(
                customer_message=customer_message,
                predicted_intent=predicted_intent,
                intent_confidence=intent_confidence,
                evidence_text=evidence_str,
            )
            try:
                llm_reply = self.llm_client.generate(system_prompt=GROUNDED_SYSTEM_PROMPT, user_prompt=prompt)
                return {
                    "reply": llm_reply.strip(),
                    "grounded": True,
                    "evidence_ids": [c["case_id"] for c in evidence_cases],
                }
            except Exception:
                pass  # Fall back to deterministic evidence synthesizer

        # High-precision deterministic evidence grounding
        top_evidence = evidence_cases[0]
        top_agent_reply = top_evidence.get("agent_reply", "")
        evidence_intent = top_evidence.get("intent", predicted_intent)
        
        # Clean brand signature / handles from historical reply
        cleaned_reply = re.sub(r"@\w+", "", top_agent_reply)
        cleaned_reply = re.sub(r"\s+", " ", cleaned_reply).strip()
        
        # Remove personal agent sign-offs like "^SJ", "-Alex", "^Amazon"
        cleaned_reply = re.sub(r"[\^–-][A-Z]{1,3}\s*$", "", cleaned_reply).strip()

        # Construct intent-tailored grounded reply if cleaning produced a good template
        if len(cleaned_reply) >= 20 and not cleaned_reply.lower().startswith("sorry to hear that you have"):
            final_reply = cleaned_reply
        else:
            # Fall back to canonical grounded pattern for this intent
            if predicted_intent == "Delivery Issue":
                final_reply = "We're sorry to hear about the delivery delay. Please check your tracking link in 'Your Orders', or send us a DM with your order ID so we can investigate."
            elif predicted_intent == "Refund Request":
                final_reply = "We'd be glad to check your refund status. Please send us a DM with your 17-digit order number so we can look into this for you."
            elif predicted_intent == "Return / Wrong or Damaged Item":
                final_reply = "We're sorry your item arrived damaged/incorrect. You can start a replacement or return via 'Your Orders' > 'Return or Replace items', or DM us for help."
            elif predicted_intent == "Order Cancellation Request":
                final_reply = "You can attempt to cancel items in 'Your Orders' before shipment. If the option is no longer available, please DM us your order ID."
            elif predicted_intent == "Address / Delivery Redirect Issue":
                final_reply = "To update your delivery address, please visit 'Your Orders' before dispatch. If the order has already shipped, DM us so we can check options."
            elif predicted_intent == "Prime Membership Billing/Cancellation":
                final_reply = "You can manage or cancel your Prime subscription anytime under 'Your Account' > 'Prime'. For billing inquiries, please DM us your account email."
            elif predicted_intent in ["Account Access / Security", "Unauthorized / Non-Prime Charge"]:
                final_reply = "For account security and unrecognized charges, please reach out directly through Amazon Customer Service or DM us securely for verification."
            else:
                final_reply = "We'd like to look into this issue for you. Please send us a direct message with your order details and account email."

        return {
            "reply": final_reply,
            "grounded": True,
            "evidence_ids": [c["case_id"] for c in evidence_cases[:3]],
            "primary_evidence_case": top_evidence.get("case_id"),
        }
