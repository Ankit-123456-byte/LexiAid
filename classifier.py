"""
classifier.py — Deterministic NLP clause classifier.

Two-layer approach:
  Layer 1 — Rule-based regex  (instant, deterministic, always runs)
  Layer 2 — Zero-shot NLI     (semantic, catches paraphrases, lazy-loaded)

Results from both layers are merged.  The classifier risk level can also
*upgrade* the LLM-assigned risk when it spots something the LLM missed.
"""

import re
from typing import Optional

# ── Clause taxonomy ───────────────────────────────────────────────────────────

CLAUSE_PATTERNS: dict = {
    "arbitration": [
        r"arbitrat\w*",
        r"dispute.{0,25}resolution",
        r"\bJAMS\b",
        r"\bAAA\b.{0,10}arbitration",
        r"binding\s+arbitration",
    ],
    "auto_renewal": [
        r"auto.?renew\w*",
        r"automatically\s+renew",
        r"evergreen\s+clause",
        r"renews?\s+unless\s+cancelled",
        r"rollover\s+term",
    ],
    "indemnification": [
        r"indemnif\w+",
        r"hold\s+harmless",
        r"defend\s+and\s+indemnif",
        r"indemnitor",
    ],
    "limitation_of_liability": [
        r"limit\w{0,6}\s+of\s+liability",
        r"in\s+no\s+event.{0,40}liable",
        r"shall\s+not\s+be\s+liable",
        r"aggregate\s+liability.{0,30}shall\s+not\s+exceed",
        r"cap\s+on\s+liability",
    ],
    "waiver_of_rights": [
        r"waive.{0,25}right",
        r"class\s+action\s+waiver",
        r"jury\s+trial\s+waiver",
        r"waiver\s+of\s+jury",
        r"you\s+waive\s+any",
    ],
    "liquidated_damages": [
        r"liquidated\s+damages",
        r"penalty.{0,20}breach",
        r"penalty\s+clause",
        r"agreed.{0,10}damages",
    ],
    "unilateral_modification": [
        r"reserves?.{0,25}right.{0,25}modif",
        r"may\s+change\s+.{0,25}terms?\s+at\s+any\s+time",
        r"sole\s+discretion.{0,30}modif",
        r"update\s+these\s+terms?\s+at\s+any\s+time",
        r"without\s+notice.{0,20}modif",
    ],
    "governing_law": [
        r"governed\s+by.{0,20}law",
        r"jurisdiction.{0,20}court",
        r"laws?\s+of\s+the\s+state\s+of",
        r"courts?\s+of\s+competent\s+jurisdiction",
    ],
    "non_compete": [
        r"non.?compet\w+",
        r"not.{0,20}compet.{0,20}with",
        r"restrictive\s+covenant",
        r"covenant\s+not\s+to\s+compet",
    ],
    "termination": [
        r"terminat\w+\s+without\s+cause",
        r"at.?will\s+terminat",
        r"may\s+terminate.{0,25}any\s+time",
        r"immediate\s+termination",
        r"terminate\s+this\s+agreement",
    ],
    "intellectual_property": [
        r"intellectual\s+property",
        r"\bIP\s+rights?\b",
        r"work\s+for\s+hire",
        r"assigns?\s+all\s+rights",
        r"moral\s+rights",
        r"assigns?\s+ownership",
    ],
    "confidentiality": [
        r"confidential\w*",
        r"non.?disclosur\w+",
        r"\bNDA\b",
        r"proprietary\s+information",
        r"trade\s+secret",
    ],
    "payment_terms": [
        r"payment\s+due",
        r"invoice\w*",
        r"fee\s+schedule",
        r"late\s+fee",
        r"interest\s+on\s+overdue",
        r"past.?due\s+amount",
    ],
    "force_majeure": [
        r"force\s+majeure",
        r"act\s+of\s+god",
        r"beyond.{0,25}reasonable\s+control",
        r"unforeseeable\s+circumstances",
    ],
    "data_privacy": [
        r"personal\s+data",
        r"data\s+protection",
        r"\bGDPR\b",
        r"\bPDPA\b",
        r"data\s+processing",
        r"personally\s+identifiable",
    ],
}

# Clause types that inherently threaten signer's rights / finances
HIGH_RISK_TYPES = {
    "arbitration",
    "waiver_of_rights",
    "auto_renewal",
    "unilateral_modification",
    "non_compete",
    "liquidated_damages",
}

MEDIUM_RISK_TYPES = {
    "indemnification",
    "limitation_of_liability",
    "termination",
    "intellectual_property",
}

# Human-readable display names
CLAUSE_LABELS: dict = {
    "arbitration":             "Arbitration Clause",
    "auto_renewal":            "Auto-Renewal",
    "indemnification":         "Indemnification",
    "limitation_of_liability": "Limitation of Liability",
    "waiver_of_rights":        "Waiver of Rights",
    "liquidated_damages":      "Liquidated Damages",
    "unilateral_modification": "Unilateral Modification",
    "governing_law":           "Governing Law",
    "non_compete":             "Non-Compete",
    "termination":             "Termination Clause",
    "intellectual_property":   "Intellectual Property",
    "confidentiality":         "Confidentiality / NDA",
    "payment_terms":           "Payment Terms",
    "force_majeure":           "Force Majeure",
    "data_privacy":            "Data Privacy",
}

# Candidate labels passed to the zero-shot model
ZS_CANDIDATES = list(CLAUSE_PATTERNS.keys())

# ── Zero-shot model (lazy singleton) ─────────────────────────────────────────
_zero_shot_pipeline = None


def _get_zero_shot():
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        from transformers import pipeline as hf_pipeline
        # facebook/bart-large-mnli is the canonical zero-shot classification model.
        # First run downloads ~1.6 GB; subsequent runs use the local cache.
        _zero_shot_pipeline = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,   # CPU; change to 0 for GPU
        )
    return _zero_shot_pipeline


# ── Layer 1: Rule-based regex ─────────────────────────────────────────────────

def _rule_based(text: str) -> list:
    """Return clause type keys detected via regex patterns."""
    text_lower = text.lower()
    return [
        ctype
        for ctype, patterns in CLAUSE_PATTERNS.items()
        if any(re.search(pat, text_lower, re.IGNORECASE) for pat in patterns)
    ]


# ── Layer 2: Zero-shot NLI ────────────────────────────────────────────────────

def _zero_shot(text: str, threshold: float = 0.60) -> list:
    """Return clause type keys above confidence threshold from NLI model."""
    try:
        clf = _get_zero_shot()
        result = clf(text[:512], candidate_labels=ZS_CANDIDATES, multi_label=True)
        return [
            label
            for label, score in zip(result["labels"], result["scores"])
            if score >= threshold
        ]
    except Exception:
        return []


# ── Risk resolution ───────────────────────────────────────────────────────────

def _resolve_risk(clause_types: list) -> str:
    if any(t in HIGH_RISK_TYPES for t in clause_types):
        return "high"
    if any(t in MEDIUM_RISK_TYPES for t in clause_types):
        return "medium"
    if clause_types:
        return "low"
    return "none"


# ── Public API ────────────────────────────────────────────────────────────────

def classify_clause(text: str, use_zero_shot: bool = True) -> dict:
    """
    Classify a single legal clause.

    Args:
        text:          Raw clause text (original + simplified combined works best)
        use_zero_shot: Set False to run rule-based only (faster, less accurate)

    Returns:
        {
          "clause_types":    ["arbitration", ...],          # merged keys
          "clause_labels":   ["Arbitration Clause", ...],   # human-readable
          "rule_based":      [...],                         # layer-1 only
          "zero_shot":       [...],                         # layer-2 only
          "classifier_risk": "high|medium|low|none",
          "is_high_risk":    bool
        }
    """
    rule_types = _rule_based(text)
    zero_types = _zero_shot(text) if use_zero_shot else []
    all_types  = list(set(rule_types) | set(zero_types))
    risk       = _resolve_risk(all_types)

    return {
        "clause_types":    all_types,
        "clause_labels":   [CLAUSE_LABELS.get(t, t) for t in all_types],
        "rule_based":      rule_types,
        "zero_shot":       zero_types,
        "classifier_risk": risk,
        "is_high_risk":    risk == "high",
    }


def classify_sections(sections: list, use_zero_shot: bool = True) -> list:
    """
    Run the classifier over all clause sections returned by the LLM.

    Adds a `classifier` key to each section, and upgrades the LLM-assigned
    `risk` level when the classifier detects a higher-severity clause type.

    Args:
        sections:      List of section dicts from simplify_document()
        use_zero_shot: Whether to use the HuggingFace NLI model

    Returns:
        Same list with `classifier` key added to each section
    """
    RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}

    for section in sections:
        combined = (section.get("original", "") + " " + section.get("simplified", "")).strip()
        classification = classify_clause(combined, use_zero_shot=use_zero_shot)
        section["classifier"] = classification

        # Upgrade risk if classifier found something more severe than LLM
        llm_risk = section.get("risk", "none")
        clf_risk = classification["classifier_risk"]
        if RISK_ORDER.get(clf_risk, 0) > RISK_ORDER.get(llm_risk, 0):
            section["risk"] = clf_risk
            if not section.get("risk_note") and classification["clause_labels"]:
                labels = ", ".join(classification["clause_labels"])
                section["risk_note"] = f"Classifier detected: {labels}"

    return sections
