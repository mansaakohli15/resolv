"""
Milestone 5B -- apply human-determined annotations to evaluation/golden_set.csv.

Every (human_intent, difficulty, note) value below was determined by independently reading
the actual message text against the 8-intent taxonomy + Other/Out of Scope + Unsupported
Language + Ambiguous, per the annotation rules in the milestone brief -- NOT copied from
heuristic_intent. Disagreements with heuristic_intent are kept as the human label and noted.

Run: python scripts/annotate_golden_set.py
"""
import pandas as pd

# golden_id -> (human_intent, difficulty, annotator_notes)
ANNOTATIONS = {
"g0001": ("Delivery Issue", "Easy", ""),
"g0002": ("Delivery Issue", "Medium", "Complains delivery isn't meeting the Prime guarantee; no charge/cancel action requested, so classified as delivery, not Prime billing."),
"g0003": ("Delivery Issue", "Easy", ""),
"g0004": ("Delivery Issue", "Medium", "Prime mentioned as context for a delivery-speed complaint, not a billing/cancel request."),
"g0005": ("Delivery Issue", "Easy", ""),
"g0006": ("Delivery Issue", "Medium", "Disputes a failed-delivery claim; core issue is non-receipt."),
"g0007": ("Delivery Issue", "Medium", "Misdelivered to an unnamed neighbour -- no address-correction request stated, so kept as Delivery not Address."),
"g0008": ("Delivery Issue", "Medium", "Same as g0007: uncertain-location non-receipt, not an explicit wrong-address report."),
"g0009": ("Delivery Issue", "Easy", ""),
"g0010": ("Delivery Issue", "Medium", "Garbled phrasing but core issue is a disputed delivery attempt."),
"g0011": ("Delivery Issue", "Easy", ""),
"g0012": ("Delivery Issue", "Medium", "Prime mentioned as context; complaint is delivery speed."),
"g0013": ("Delivery Issue", "Easy", ""),
"g0014": ("Delivery Issue", "Easy", ""),
"g0015": ("Address / Delivery Redirect Issue", "Easy", "Heuristic tagged delivery_issue only because 'address' loosely matched the delivery regex; message explicitly requests an address change."),
"g0016": ("Delivery Issue", "Easy", ""),
"g0017": ("Delivery Issue", "Medium", ""),
"g0018": ("Delivery Issue", "Easy", ""),
"g0019": ("Delivery Issue", "Easy", ""),
"g0020": ("Delivery Issue", "Medium", "Prime fee context; complaint is delivery timing."),
"g0021": ("Delivery Issue", "Medium", "No address correction requested, just a non-receipt/false-delivery-claim dispute."),
"g0022": ("Delivery Issue", "Medium", "Mentions driver went to wrong house, but the stated concern is timely arrival, not an address fix request."),
"g0023": ("Delivery Issue", "Medium", "Food delivery apparently taken by the driver -- non-receipt issue."),
"g0024": ("Delivery Issue", "Medium", "Prime value complaint driven by delivery unreliability; no charge/cancel request."),
"g0025": ("Delivery Issue", "Medium", "Prime context; complaint is delivery timing."),
"g0026": ("Delivery Issue", "Easy", ""),
"g0027": ("Delivery Issue", "Medium", "Prime context; complaint is loss of a delivery option/speed."),
"g0028": ("Delivery Issue", "Easy", ""),
"g0029": ("Delivery Issue", "Easy", ""),
"g0030": ("Delivery Issue", "Easy", ""),

"g0031": ("Other / Out of Scope", "Medium", "General UX/process suggestion, not a specific support request."),
"g0032": ("Other / Out of Scope", "Medium", "Feature request (multiple payment methods); not a refund/charge dispute."),
"g0033": ("Delivery Issue", "Medium", "Heuristic missed this (uses 'ship' not 'deliver'); shipment never went out -- a genuine delivery delay."),
"g0034": ("Address / Delivery Redirect Issue", "Medium", "Heuristic missed 'pin code entered INCORRECT' as an address problem."),
"g0035": ("Return / Wrong or Damaged Item", "Medium", "Ordered item; box arrived empty -- treated as a wrong/missing-item case."),
"g0036": ("Other / Out of Scope", "Easy", "Positive thank-you message, no support request."),
"g0037": ("Ambiguous", "Hard", "Garbled/incomplete sentence; references an order and a decision to use the app or not, but no concrete issue is stated."),
"g0038": ("Return / Wrong or Damaged Item", "Medium", "Package arrived wet and packaging ruined -- treated as a damaged-item report despite the heuristic missing the word 'damaged'."),
"g0039": ("Other / Out of Scope", "Easy", "Product commentary, not a support request."),
"g0040": ("Ambiguous", "Hard", "References redoing something 3 times without stating what the underlying issue is; not enough standalone context."),
"g0041": ("Ambiguous", "Hard", "Payment/balance-credit issue tied to an unclear 'recharge and win' promotion; could plausibly be Refund, Unauthorized Charge, or Other depending on unstated context."),
"g0042": ("Other / Out of Scope", "Medium", "Checkout/quantity-limit question, not covered by the 8 intents."),
"g0043": ("Unsupported Language", "Medium", "Japanese text; heuristic's non-English marker list only covers Romance languages and missed this."),
"g0044": ("Return / Wrong or Damaged Item", "Medium", "Item (trailer hitch) doesn't fit as described -- treated as a wrong-item case, though no explicit return request is stated."),
"g0045": ("Other / Out of Scope", "Easy", "Positive feedback, no issue."),
"g0046": ("Other / Out of Scope", "Easy", "Generic complaint (misspelling), not an actionable support intent."),
"g0047": ("Other / Out of Scope", "Medium", "Broken help-center link; a website bug, not one of the 8 support intents."),
"g0048": ("Other / Out of Scope", "Medium", "Packaging-quality feedback, no damaged item stated."),
"g0049": ("Account Access / Security", "Medium", "Heuristic missed 'closed my account without any reason' as an account-access issue."),
"g0050": ("Other / Out of Scope", "Easy", "Packaging feedback."),
"g0051": ("Return / Wrong or Damaged Item", "Medium", "Product stopped working after a month; also mentions an unrelated broken review link, but the defect is the concrete issue."),
"g0052": ("Unsupported Language", "Easy", "Japanese; heuristic missed it (Romance-language markers only)."),
"g0053": ("Unsupported Language", "Easy", "Japanese; heuristic missed it."),
"g0054": ("Delivery Issue", "Medium", "Complaint about delivery handling/security practice (leaving packages exposed)."),
"g0055": ("Ambiguous", "Hard", "No context for what 'this' refers to; cannot determine a primary intent."),
"g0056": ("Other / Out of Scope", "Easy", "Marketing-email/spam complaint, not one of the 8 intents."),
"g0057": ("Unsupported Language", "Medium", "Japanese; heuristic missed it. Content is delivery/shipment-split related but language rule takes precedence per the taxonomy scope decision."),
"g0058": ("Unsupported Language", "Medium", "German; heuristic missed it."),
"g0059": ("Unsupported Language", "Medium", "Japanese; content is substantively a refund question but classified as Unsupported Language per the v1 scope decision."),
"g0060": ("Delivery Issue", "Medium", "Complaint about delivery placement practice."),

"g0061": ("Refund Request", "Easy", ""),
"g0062": ("Refund Request", "Medium", "Root cause is a delivery delay, but the concrete ask is a refund of the extra fee."),
"g0063": ("Refund Request", "Medium", "Delivery status mentioned, but explicit ask is a refund."),
"g0064": ("Refund Request", "Easy", ""),
"g0065": ("Refund Request", "Medium", "Dual-option request (redeliver or refund); refund is an explicit stated option."),
"g0066": ("Refund Request", "Easy", ""),
"g0067": ("Refund Request", "Easy", ""),
"g0068": ("Prime Membership Billing/Cancellation", "Medium", "Heuristic matched on 'refund', but the live issue is an ongoing/erroneous Prime charge after a prior refund."),
"g0069": ("Refund Request", "Medium", "Root cause is delivery mishandling; explicit stated ask is a refund."),
"g0070": ("Refund Request", "Medium", "Refund eligibility question tied to a gift card."),
"g0071": ("Refund Request", "Easy", ""),
"g0072": ("Refund Request", "Easy", ""),
"g0073": ("Refund Request", "Easy", ""),
"g0074": ("Refund Request", "Easy", ""),
"g0075": ("Unauthorized / Non-Prime Charge", "Hard", "Combines an expected-but-missing refund with a newly reported extra charge; the concrete new complaint (money taken out) is the charge dispute, chosen as primary over the pre-existing refund expectation."),
"g0076": ("Refund Request", "Easy", ""),
"g0077": ("Refund Request", "Medium", "Mentions 'Charge Dispute' process by name, but the live complaint is a refund not received, not disputing the charge itself."),
"g0078": ("Refund Request", "Easy", ""),
"g0079": ("Refund Request", "Medium", "No defect/wrong item stated; wants to send back an unsatisfactory device for money back -- classified by the stated goal (refund), not as a Return."),
"g0080": ("Return / Wrong or Damaged Item", "Medium", "Heuristic matched on 'refund', but the item is explicitly defective (won't charge/turn on) and exchange is co-requested -- Return is the better fit per the stated defect."),

"g0081": ("Return / Wrong or Damaged Item", "Medium", "Complaint about the cost/fairness of returning an item; topic is fundamentally a return."),
"g0082": ("Return / Wrong or Damaged Item", "Medium", "Damaged item already in the return process; refund status is secondary."),
"g0083": ("Return / Wrong or Damaged Item", "Medium", "Fraudulent/wrong item already returned; refund promise being broken is the newer complaint but originating issue is the item itself."),
"g0084": ("Ambiguous", "Hard", "Conflates a returned-shipping-fee question with a separate non-delivery complaint; unclear if they're the same order or which is primary."),
"g0085": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0086": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0087": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0088": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0089": ("Return / Wrong or Damaged Item", "Medium", "Replacement shipment poorly packaged/likely damaged."),
"g0090": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0091": ("Return / Wrong or Damaged Item", "Medium", "Damaged gift card -- slightly unusual 'item' but fits the damaged-item definition."),
"g0092": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0093": ("Other / Out of Scope", "Medium", "Positive anecdote about a return interaction; no active issue to resolve."),
"g0094": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0095": ("Other / Out of Scope", "Medium", "Concrete question asked is a how-to about the messaging feature, only tangentially tied to a defective-product context."),
"g0096": ("Other / Out of Scope", "Hard", "Vague dissatisfaction; 'may just return it' is contemplation, not a concrete return/cancellation request."),
"g0097": ("Return / Wrong or Damaged Item", "Easy", ""),
"g0098": ("Return / Wrong or Damaged Item", "Medium", "General how-to question about returning a gift, not tied to a specific defect."),
"g0099": ("Return / Wrong or Damaged Item", "Medium", "Return process/fee question; fee complaint is tied directly to the return, not a separate billing dispute."),
"g0100": ("Return / Wrong or Damaged Item", "Easy", ""),

"g0101": ("Account Access / Security", "Medium", "Hacked account is the root issue; refund-not-received is a secondary consequence."),
"g0102": ("Account Access / Security", "Hard", "Genuine toss-up between Account Access (blocking lockout) and Prime Billing (can't cancel, being charged); classified by the access problem since it's what's preventing resolution of the rest."),
"g0103": ("Account Access / Security", "Easy", ""),
"g0104": ("Account Access / Security", "Easy", ""),
"g0105": ("Account Access / Security", "Easy", ""),
"g0106": ("Account Access / Security", "Easy", ""),
"g0107": ("Account Access / Security", "Medium", ""),
"g0108": ("Account Access / Security", "Easy", ""),
"g0109": ("Account Access / Security", "Easy", ""),
"g0110": ("Account Access / Security", "Easy", ""),
"g0111": ("Account Access / Security", "Easy", ""),
"g0112": ("Account Access / Security", "Easy", ""),
"g0113": ("Account Access / Security", "Medium", "Login issue specific to the Kindle app, but fundamentally an authentication problem."),
"g0114": ("Account Access / Security", "Easy", ""),
"g0115": ("Account Access / Security", "Easy", ""),

"g0116": ("Delivery Issue", "Medium", "Heuristic false positive: 'reconsider my Prime renewal next year' is a future contingent statement, not a current charge/cancel action; core complaint is delivery speed."),
"g0117": ("Order Cancellation Request", "Hard", "Heuristic false positive: 'Prime Now' is a service name here, not a Prime-subscription cancellation -- 'cancel' matched near 'prime' by coincidence. Actual topic is Amazon cancelling the customer's order."),
"g0118": ("Prime Membership Billing/Cancellation", "Medium", "Root complaint is delivery, but an explicit 'cancel my prime membership' statement is present, distinguishing it from g0116/g0123."),
"g0119": ("Refund Request", "Hard", "Multiple sub-issues (order cancellation, pending refund, future Prime-cancellation musing); the concrete current ask is the refund for the cancelled order."),
"g0120": ("Other / Out of Scope", "Medium", "Shipping-fee-structure gripe with sarcastic tone; no explicit refund/cancellation request."),
"g0121": ("Prime Membership Billing/Cancellation", "Easy", ""),
"g0122": ("Prime Membership Billing/Cancellation", "Medium", "Self-inflicted (forgot to cancel), but topically a Prime billing complaint."),
"g0123": ("Delivery Issue", "Medium", "Same pattern as g0116: 'cancel my prime' is rhetorical; the real complaint is shipping delays."),
"g0124": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule, Prime charges belong to the Prime intent even though phrased as 'charged 3 times'."),
"g0125": ("Prime Membership Billing/Cancellation", "Easy", ""),
"g0126": ("Prime Membership Billing/Cancellation", "Easy", ""),
"g0127": ("Prime Membership Billing/Cancellation", "Easy", ""),
"g0128": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule, Prime charges go to the Prime intent even though 'didn't request or authorised' echoes Unauthorized-Charge language."),
"g0129": ("Prime Membership Billing/Cancellation", "Easy", ""),
"g0130": ("Prime Membership Billing/Cancellation", "Easy", ""),

"g0131": ("Order Cancellation Request", "Medium", "Order cancelled by Amazon; topic centers on that cancellation event."),
"g0132": ("Order Cancellation Request", "Medium", "Also touches payment, but the core dispute is the cancellation itself vs. a promised replacement."),
"g0133": ("Order Cancellation Request", "Easy", ""),
"g0134": ("Delivery Issue", "Hard", "Heuristic false positive: 'cancelled' appears only to confirm the order was NOT cancelled; actual issue is non-delivery."),
"g0135": ("Order Cancellation Request", "Medium", "Complaint about Amazon's order-cancellation practice after price changes."),
"g0136": ("Order Cancellation Request", "Easy", ""),
"g0137": ("Order Cancellation Request", "Easy", ""),
"g0138": ("Refund Request", "Hard", "Cancellation already happened; the unresolved, live complaint is the missing refund for it."),
"g0139": ("Order Cancellation Request", "Easy", ""),
"g0140": ("Order Cancellation Request", "Easy", ""),
"g0141": ("Order Cancellation Request", "Easy", ""),
"g0142": ("Order Cancellation Request", "Medium", "Ambiguous phrasing ('no order cancellation') but clearly cancellation-topic."),
"g0143": ("Order Cancellation Request", "Easy", ""),
"g0144": ("Order Cancellation Request", "Medium", "Unusual detail (courier agent threat) but core topic is an unwanted cancellation."),
"g0145": ("Order Cancellation Request", "Medium", "Immediate question is about escalation process, but the underlying topic is a cancelled order."),

"g0146": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0147": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0148": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0149": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0150": ("Address / Delivery Redirect Issue", "Medium", "Root cause is a wrong-address delivery; also touches a return-related complication (send back an item never received)."),
"g0151": ("Address / Delivery Redirect Issue", "Medium", ""),
"g0152": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0153": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0154": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0155": ("Address / Delivery Redirect Issue", "Easy", ""),
"g0156": ("Address / Delivery Redirect Issue", "Medium", "Framed as a UX complaint, but the explicit ask is how to change the address/instructions."),
"g0157": ("Address / Delivery Redirect Issue", "Hard", "Two separate numbered issues (wrong address; wrong item colour) bundled in one message -- genuine multi-intent case; address chosen as primary since listed first and is the more systemic issue."),
"g0158": ("Order Cancellation Request", "Hard", "Root cause is a wrong-address delivery, but the concrete event described is the customer cancelling their own order because of it."),
"g0159": ("Address / Delivery Redirect Issue", "Easy", ""),

"g0160": ("Prime Membership Billing/Cancellation", "Easy", "Entirely about a Prime charge/cancellation confusion; per the stated rule, Prime charges go to the Prime intent."),
"g0161": ("Prime Membership Billing/Cancellation", "Hard", "Garbled message with two concerns (payment-method security, double Prime charge); the explicit 'double charged...for one prime membership' clause makes Prime the defensible primary per the stated rule."),
"g0162": ("Prime Membership Billing/Cancellation", "Medium", "The $80 charge amount is unclear, but the complaint is explicitly tied to being charged for Prime despite a service failure -- Prime rule applies."),
"g0163": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0164": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0165": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0166": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0167": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0168": ("Unauthorized / Non-Prime Charge", "Easy", "No Prime mention -- true positive for this bucket."),
"g0169": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule."),
"g0170": ("Prime Membership Billing/Cancellation", "Easy", "Per the stated rule (typo 'prume' = prime)."),
"g0171": ("Unauthorized / Non-Prime Charge", "Hard", "Genuinely subtle: charge is for a piece of Prime-included content, not the Prime subscription fee itself -- classified as a content billing error, not Prime-subscription billing. Flagged as a taxonomy boundary worth refining."),
"g0172": ("Other / Out of Scope", "Medium", "Heuristic false positive: this is a pre-purchase timing question ('when will I be charged'), not a dispute of an actual charge."),
"g0173": ("Unauthorized / Non-Prime Charge", "Medium", "No Prime mention; ongoing charge for content that was previously free."),
"g0174": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0175": ("Unauthorized / Non-Prime Charge", "Medium", "'Membership' type is unspecified and not explicitly named Prime -- kept as non-Prime charge on that literal basis, flagged as an interpretive call."),
"g0176": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0177": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0178": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0179": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0180": ("Unauthorized / Non-Prime Charge", "Hard", "Truncated/garbled text; combines a billing dispute (seller repayment charge) with an account-login mention -- charge dispute chosen as the stated primary complaint."),
"g0181": ("Unauthorized / Non-Prime Charge", "Medium", "Subscription type unspecified/not named Prime."),
"g0182": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0183": ("Unauthorized / Non-Prime Charge", "Easy", ""),
"g0184": ("Unauthorized / Non-Prime Charge", "Medium", "Third-party merchant (redbus), likely via Amazon Pay -- kept as a charge dispute since that's the substance of the complaint to AmazonHelp."),

"g0185": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0186": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0187": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0188": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0189": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0190": ("Unsupported Language", "Easy", "Confirmed French."),
"g0191": ("Unsupported Language", "Easy", "Confirmed Portuguese."),
"g0192": ("Unsupported Language", "Easy", "Confirmed Portuguese."),
"g0193": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0194": ("Unsupported Language", "Easy", "Confirmed Spanish (reads as an unrelated marketing pitch, but language rule takes precedence)."),
"g0195": ("Unsupported Language", "Easy", "Confirmed Spanish."),
"g0196": ("Unsupported Language", "Easy", "Confirmed Spanish."),
}


def main():
    g = pd.read_csv("evaluation/golden_set.csv")
    assert len(ANNOTATIONS) == len(g), f"annotation count {len(ANNOTATIONS)} != golden set size {len(g)}"

    missing = set(g["golden_id"]) - set(ANNOTATIONS.keys())
    assert not missing, f"missing annotations for {missing}"

    g["human_intent"] = g["golden_id"].map(lambda gid: ANNOTATIONS[gid][0])
    g["difficulty"] = g["golden_id"].map(lambda gid: ANNOTATIONS[gid][1])
    existing_notes = g["annotator_notes"].fillna("")
    new_notes = g["golden_id"].map(lambda gid: ANNOTATIONS[gid][2])
    g["annotator_notes"] = [
        (sn + " | " + nn).strip(" |") if nn else sn
        for sn, nn in zip(existing_notes, new_notes)
    ]

    ALLOWED = {
        "Delivery Issue", "Address / Delivery Redirect Issue", "Refund Request",
        "Return / Wrong or Damaged Item", "Prime Membership Billing/Cancellation",
        "Order Cancellation Request", "Account Access / Security",
        "Unauthorized / Non-Prime Charge", "Other / Out of Scope",
        "Unsupported Language", "Ambiguous",
    }
    assert set(g["human_intent"]) <= ALLOWED, set(g["human_intent"]) - ALLOWED
    assert set(g["difficulty"]) <= {"Easy", "Medium", "Hard"}
    assert g["human_intent"].notna().all() and (g["human_intent"] != "").all()
    assert g["difficulty"].notna().all() and (g["difficulty"] != "").all()

    g.to_csv("evaluation/golden_set.csv", index=False)

    print(f"Annotated {len(g)} rows")
    print("\nFinal human_intent distribution:")
    print(g["human_intent"].value_counts().to_string())
    print("\nDifficulty distribution:")
    print(g["difficulty"].value_counts().to_string())
    disagreements = g[g["heuristic_intent"] != g["human_intent"]]
    # heuristic_intent uses snake_case tags; normalize for a fair comparison isn't needed here,
    # we just report raw string mismatches since the two label sets differ by design (see report)
    print(f"\nRows where human_intent differs from heuristic_intent tag: (see report for the real count using label mapping)")


if __name__ == "__main__":
    main()
