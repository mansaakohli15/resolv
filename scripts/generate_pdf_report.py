"""
Script to generate a professional PDF submission report for Hiver SDE Intern Take-Home Assignment.
Uses reportlab to create a clean, beautifully formatted multi-page document.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)

PDF_OUTPUT_PATH = PROJECT_ROOT / "reports" / "Hiver_SDE_Intern_Report_Mansaa_Kohli.pdf"


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")    # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    c_accent = colors.HexColor("#C53030")     # Crimson Accent
    c_dark = colors.HexColor("#2D3748")       # Charcoal Body Text
    c_light_bg = colors.HexColor("#F7FAFC")   # Light Table BG
    c_border = colors.HexColor("#E2E8F0")     # Border Grey

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark,
        spaceAfter=6,
    )

    body_bold = ParagraphStyle(
        "Body_Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=c_dark,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold",
        textColor=c_primary,
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=body_style,
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2C5282"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Hiver SDE Intern Assignment — Technical Report", title_style))
    story.append(Paragraph("<b>Resolv: Evidence-Grounded Customer Support Automation</b>", subtitle_style))
    
    meta_table_data = [
        [
            Paragraph("<b>Candidate:</b> Mansaa Kohli", meta_style),
            Paragraph("<b>Selected Brand:</b> AmazonHelp", meta_style),
            Paragraph("<b>Submission:</b> anurag@hiverhq.com", meta_style),
        ],
        [
            Paragraph("<b>Repo:</b> github.com/mansaakohli15/resolv", meta_style),
            Paragraph("<b>Dataset:</b> Kaggle TWCS (3M Tweets)", meta_style),
            Paragraph("<b>Golden Set:</b> 196 Hand-Annotated Instances", meta_style),
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[180, 170, 180])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light_bg),
        ('BOX', (0,0), (-1,-1), 0.5, c_border),
        ('INNERGRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceBefore=4, spaceAfter=8))

    # Section 1
    story.append(Paragraph("1. Problem Framing: What \"Good\" Means for AmazonHelp & What We Chose NOT to Build", h1_style))
    story.append(Paragraph(
        "Automating customer support on Twitter is not an unconstrained text generation problem. Tweets are noisy, emotionally charged, and frequently lack critical account context. For <b>AmazonHelp</b>, an effective automated support agent must satisfy three strict operational criteria:",
        body_style
    ))
    story.append(Paragraph("• <b>Strict Factual Grounding:</b> The system must never hallucinate refund confirmations, fake delivery guarantees, or claim actions were taken.", bullet_style))
    story.append(Paragraph("• <b>Transparent Operational Gating:</b> When an inquiry requires authentication (password resets, card charges) or evidence is weak, the system must <b>escalate to a human specialist with an explicit reason</b>.", bullet_style))
    story.append(Paragraph("• <b>Evidence-Backed Justification:</b> Every automated reply must cite historical brand resolution cases explaining <i>why</i> that action is recommended.", bullet_style))
    story.append(Paragraph(
        "<b>What We Chose NOT to Build:</b> We rejected unconstrained LLM chatbots (which hallucinate policies and cannot be calibrated for enterprise compliance), disposable frontend dashboards (focusing our engineering time on leakage prevention and statistical validation), and large cloud dependencies (the full pipeline reproduces locally in &lt;90 seconds).",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Dataset Profiling, 8-Intent Taxonomy & Leakage-Safe Splitting", h1_style))
    story.append(Paragraph(
        "<b>Brand Selection:</b> Profiling 10 candidate brands in the 2.8M-tweet TWCS dataset established <b>AmazonHelp</b> as the optimal candidate (82,246 conversation components, 371,417 tweets, and a 94.2% agent response rate).",
        body_style
    ))
    story.append(Paragraph(
        "<b>8-Intent Taxonomy (Data-Derived):</b> Discovered through semantic clustering and manual validation: (1) <i>Delivery Issue</i>, (2) <i>Address / Delivery Redirect Issue</i>, (3) <i>Refund Request</i>, (4) <i>Return / Wrong or Damaged Item</i>, (5) <i>Prime Membership Billing/Cancellation</i>, (6) <i>Order Cancellation Request</i>, (7) <i>Account Access / Security</i> [High Risk], and (8) <i>Unauthorized / Non-Prime Charge</i> [High Risk]. Out-of-taxonomy categories include <i>Other / Out of Scope</i>, <i>Unsupported Language</i>, and <i>Ambiguous</i>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Component-Level Split & Leakage Prevention:</b> Partitioned at the <b>conversation component level</b> (80% train_retrieval / 10% validation / 10% test). The retrieval index was populated <b>strictly from train_retrieval</b> (59,951 pairs), with <b>0% leakage</b> into validation or the 196-instance golden benchmark.",
        body_style
    ))

    # Section 3: Baseline Comparisons
    story.append(Paragraph("3. Empirical Results vs. Baselines (Golden Benchmark N=159 In-Scope)", h1_style))
    
    clf_data = [
        [Paragraph("Model Architecture", table_header), Paragraph("In-Scope Acc", table_header), Paragraph("Macro F1", table_header), Paragraph("Weighted F1", table_header), Paragraph("Val Acc", table_header), Paragraph("Key Characteristic", table_header)],
        [Paragraph("<b>Baseline 1 (Majority Class)</b>", table_cell), Paragraph("22.29%", table_cell), Paragraph("0.0456", table_cell), Paragraph("0.0813", table_cell), Paragraph("70.80%", table_cell), Paragraph("Always predicts Delivery Issue (empirical floor)", table_cell)],
        [Paragraph("<b>Baseline 2 (Simple TF-IDF)</b>", table_cell), Paragraph("71.34%", table_cell), Paragraph("0.7109", table_cell), Paragraph("0.7152", table_cell), Paragraph("96.08%", table_cell), Paragraph("Unigram TF-IDF + Logistic Regression", table_cell)],
        [Paragraph("<b>Main Resolv Model</b>", table_cell_bold), Paragraph("<b>79.62%</b>", table_cell_bold), Paragraph("<b>0.7960</b>", table_cell_bold), Paragraph("<b>0.7959</b>", table_cell_bold), Paragraph("<b>98.12%</b>", table_cell_bold), Paragraph("<b>Word+Char Feature Union + Calibrated LogReg</b>", table_cell_bold)],
    ]
    t_clf = Table(clf_data, colWidths=[120, 65, 60, 65, 55, 165])
    t_clf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_clf)
    story.append(Spacer(1, 4))
    story.append(Paragraph("<i>Difficulty Breakdown (Main Model):</i> <b>Easy (N=85):</b> 85.88% Acc, 0.8684 Macro F1 | <b>Medium (N=61):</b> 78.69% Acc, 0.7815 Macro F1 | <b>Hard (N=11):</b> 36.36% Acc, 0.2476 Macro F1.", callout_style))

    # Section 4: Retrieval & Escalation
    story.append(Paragraph("4. Evidence Retrieval Quality & Multi-Signal Risk Escalation", h1_style))
    
    ret_esc_data = [
        [Paragraph("Retrieval Metric (Corpus N=59,951)", table_header), Paragraph("Score", table_header), Paragraph("Escalation & Safety Metric (Golden N=196)", table_header), Paragraph("Score", table_header)],
        [Paragraph("<b>Recall@1 (Top-1 Match)</b>", table_cell), Paragraph("<b>64.97%</b>", table_cell), Paragraph("<b>Auto-Handle Rate</b>", table_cell), Paragraph("<b>57.14%</b> (112/196)", table_cell)],
        [Paragraph("<b>Recall@3 (Top-3 Match)</b>", table_cell), Paragraph("<b>87.90%</b>", table_cell), Paragraph("<b>Escalation Rate</b>", table_cell), Paragraph("<b>42.86%</b> (84/196)", table_cell)],
        [Paragraph("<b>Recall@5 (Top-5 Match)</b>", table_cell), Paragraph("<b>91.72%</b>", table_cell), Paragraph("<b>Escalation Safety Recall</b>", table_cell), Paragraph("<b>74.03%</b> (Catches risks)", table_cell)],
        [Paragraph("<b>Mean Reciprocal Rank (MRR)</b>", table_cell), Paragraph("<b>0.7567</b>", table_cell), Paragraph("<b>Factual Safety Pass Rate</b>", table_cell), Paragraph("<b>100.00%</b> (0 hallucinations)", table_cell)],
    ]
    t_ret_esc = Table(ret_esc_data, colWidths=[155, 110, 165, 100])
    t_ret_esc.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_ret_esc)

    # Section 5: LLM Judge & Validation
    story.append(Paragraph("5. LLM-as-a-Judge Rubric & Double-Blind Human Agreement Validation", h1_style))
    story.append(Paragraph(
        "<b>6-Dimension Rubric (Max 30 Points):</b> Evaluated across Relevance (4.31/5), Groundedness (3.00/5), Factual Safety (5.00/5), Actionability (3.41/5), Tone (4.71/5), and Conciseness (4.92/5) — achieving an overall mean score of <b>25.36 / 30.0 (84.5%)</b>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Empirical Validation on N=40 Stratified Instances:</b> Agreement within &plusmn;1 point: <b>35.00%</b> | Agreement within &plusmn;2 points: <b>57.50%</b> | <b>MAE: 2.35 points / 30</b> | <b>Pearson r: 0.4027</b> | <b>Spearman &rho;: 0.3811</b>. <i>Insight:</i> Automated judges over-reward generic brand politeness (*'Please DM us'*), while human reviewers penalize non-specific redirects when direct answers were possible.",
        body_style
    ))

    # Section 6: Failure Analysis
    story.append(Paragraph("6. Real Failure Analysis: Top 5 Failure Modes Observed", h1_style))
    story.append(Paragraph("1. <b>Delivery Token Bias (g0015):</b> Customer asked to update address; presence of 'delivery' and 'order' led model to misclassify as Delivery Issue, auto-replying with tracking info. <i>Fix:</i> Hierarchical rule giving address entities strict precedence.", bullet_style))
    story.append(Paragraph("2. <b>Empty Box Phrasing (g0035):</b> Customer received empty box; lacked keyword 'damaged'. Misclassified as delivery transit issue, but safely escalated due to low retrieval score (0.27 &lt; 0.28).", bullet_style))
    story.append(Paragraph("3. <b>Latent Prime Fee vs. Card Charge (g0112):</b> Customer complained of unprompted $99 deduction without saying 'Prime'. Misclassified as Unauthorized Charge; correctly escalated due to high-risk policy.", bullet_style))
    story.append(Paragraph("4. <b>Non-Romance Script Misses (g0043):</b> Japanese query bypassed regex split heuristics; safely escalated downstream by Unicode character detection in the escalation policy.", bullet_style))
    story.append(Paragraph("5. <b>Sarcastic Product Venting (g0044):</b> Sarcastic comment about trailer hitch compatibility lacked explicit verbs; low retrieval score prevented sending irrelevant templates.", bullet_style))

    # Section 7: Misleading Headline Number
    story.append(Paragraph("7. Mandatory Section: \"What is Misleading About My Headline Number?\"", h1_style))
    story.append(Paragraph(
        "1. <b>Validation Split Illusion (98.12% vs. 79.62%):</b> Citing only 98.12% validation accuracy is deceptive: validation labels were regex-derived. On messy human-annotated ground truth, accuracy is 79.62%.<br/>"
        "2. <b>Dominant Class Gravity:</b> In-scope accuracy (79.62%) hides variance: high-volume intents perform well (Account Access: 0.9375 F1), but rare intents (Prime Billing: 0.5714 F1) fail frequently.<br/>"
        "3. <b>The 57.14% Auto-Handle Fallacy:</b> Single-turn success ignores multi-turn conversational reality. Misdirecting a customer on turn 1 causes significant customer frustration.<br/>"
        "4. <b>Judge Calibration Limits:</b> The 84.5% LLM judge score reflects politeness compliance, but has only moderate correlation (r=0.4027) with human judgment.",
        body_style
    ))

    # Section 8 & 9: One More Week & Decision Log
    story.append(Paragraph("8. What We Would Do With One More Week", h1_style))
    story.append(Paragraph(
        "• <b>Multi-Turn Context Tracker:</b> Track customer entity revisions across turns 2–5.<br/>"
        "• <b>Fine-Tuned Bi-Encoder Dense Retrieval:</b> Fine-tune BGE-small / MiniLM on the 59k pairs using Multiple Negatives Ranking Loss.<br/>"
        "• <b>Active Learning & Hard Negatives:</b> Expand golden set to 500 instances focused on ambiguous charge/prime boundaries.<br/>"
        "• <b>Tool-Use Execution Sandbox:</b> Function calling for order lookups and return label generation in a mock API environment.",
        body_style
    ))

    story.append(Paragraph("9. 15-Point Engineering Decision Log (Summary)", h1_style))
    story.append(Paragraph(
        "1. <b>AmazonHelp Selected:</b> Highest volume (82k threads) & 94.2% response rate. | 2. <b>Component-Level Split:</b> 0% cross-turn contamination. | 3. <b>8 Data-Derived Intents:</b> Empirical cluster discovery over generic Banking77. | 4. <b>Merged Delivery/Tracking:</b> Unified identical resolution workflows. | 5. <b>Pruned Sparse Intents:</b> Removed Kindle device issues (&lt;0.4%). | 6. <b>Isolated Non-English Pool:</b> Preserved multilingual queries for escalation testing. | 7. <b>Stratified Golden Set:</b> Oversampled rare intents for statistical test power. | 8. <b>Train-Only Retrieval Corpus:</b> 100% leak-free evaluation. | 9. <b>Hybrid TF-IDF + Dense Search:</b> Lexical precision + dense morphology. | 10. <b>Intent Concordance Reranking:</b> Resolution coherence. | 11. <b>Multi-Signal Escalation:</b> 6 orthogonal risk signals. | 12. <b>Mandatory Security Escalation:</b> Guaranteed human review for sensitive accounts. | 13. <b>6-Dimension Rubric:</b> Separated factual safety from tone. | 14. <b>Empirical Judge Validation:</b> Double-blind human validation (r=0.4027, MAE=2.35). | 15. <b>CLI & Reproducibility:</b> Clean API and &lt;90s local reproduction over disposable UI.",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated PDF report: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
