"""
simplifier.py — Ollama-powered legal document simplification.

Pipeline:
  1. (Optional) Retrieve jurisdiction context from Pinecone RAG
  2. Send document + context to Ollama for Grade-5 simplification
  3. (Optional) Run NLP classifier over each returned clause section
"""

import os
import json
import re
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

MAX_CHARS = 12_000   # truncation limit sent to Ollama

# ── Prompts ───────────────────────────────────────────────────────────────────

SIMPLIFY_PROMPT = """\
You are LexiAid, an AI that helps ordinary people understand legal documents.

Your job: take the legal text below and return a structured JSON response with
simplified plain language. Imagine you are explaining this to a 15-year-old.

RULES:
- Use short sentences. No jargon.
- Keep ALL important information — do not omit obligations, deadlines, or rights.
- Identify clauses that could HARM the person signing (unfair terms, hidden fees,
  automatic renewals, waiver of rights, one-sided penalties, etc.).
- Mark the risk level of the overall document: "low", "medium", or "high".

INPUT DOCUMENT:
{document}

Respond ONLY with valid JSON in exactly this format (no markdown, no explanation):
{{
  "summary": "2-3 sentence plain-language summary of what this document does",
  "risk_level": "low|medium|high",
  "risk_reason": "one sentence explaining the overall risk level",
  "sections": [
    {{
      "heading": "short heading for this section",
      "original": "the original clause text (keep it short, max 200 chars)",
      "simplified": "plain language explanation",
      "risk": "none|low|medium|high",
      "risk_note": "what to watch out for (empty string if risk is none)"
    }}
  ],
  "key_obligations": ["list", "of", "things", "the signer must do"],
  "key_rights": ["list", "of", "rights", "the signer has"],
  "red_flags": ["list of the most dangerous clauses in plain language"]
}}
"""

SIMPLIFY_PROMPT_WITH_CONTEXT = """\
You are LexiAid, an AI that helps ordinary people understand legal documents.

Your job: take the legal text below and return a structured JSON response with
simplified plain language. Imagine you are explaining this to a 15-year-old.

RELEVANT LEGAL CONTEXT (retrieved from an Indian law knowledge base — use this
to add jurisdiction-specific warnings and rights):
{context}

RULES:
- Use short sentences. No jargon.
- Keep ALL important information — do not omit obligations, deadlines, or rights.
- Identify clauses that could HARM the person signing (unfair terms, hidden fees,
  automatic renewals, waiver of rights, one-sided penalties, etc.).
- Where relevant, reference the legal context above (e.g. "Under the Indian
  Contract Act, ...") to ground your explanations in Indian law.
- Mark the risk level of the overall document: "low", "medium", or "high".

INPUT DOCUMENT:
{document}

Respond ONLY with valid JSON in exactly this format (no markdown, no explanation):
{{
  "summary": "2-3 sentence plain-language summary of what this document does",
  "risk_level": "low|medium|high",
  "risk_reason": "one sentence explaining the overall risk level",
  "sections": [
    {{
      "heading": "short heading for this section",
      "original": "the original clause text (keep it short, max 200 chars)",
      "simplified": "plain language explanation",
      "risk": "none|low|medium|high",
      "risk_note": "what to watch out for (empty string if risk is none)"
    }}
  ],
  "key_obligations": ["list", "of", "things", "the signer must do"],
  "key_rights": ["list", "of", "rights", "the signer has"],
  "red_flags": ["list of the most dangerous clauses in plain language"]
}}
"""


from json_repair import repair_json

# ── JSON helpers ──────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Parse JSON from model output, repairing common issues (trailing commas, etc.)."""
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    # Extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group() if match else text
    repaired = repair_json(raw)
    return json.loads(repaired)


# ── Ollama helper ──────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Core pipeline ─────────────────────────────────────────────────────────────

def simplify_document(
    text: str,
    use_rag: bool = True,
    use_classifier: bool = True,
    classifier_zero_shot: bool = True,
) -> dict:
    """
    Full pipeline: RAG context retrieval → Ollama simplification → NLP classification.

    Args:
        text:                  Raw legal document text
        use_rag:               Retrieve Pinecone context (skipped if PINECONE_API_KEY unset)
        use_classifier:        Run NLP clause classifier over returned sections
        classifier_zero_shot:  Use HuggingFace zero-shot model (slower; needs torch)

    Returns:
        Parsed dict with summary, sections, risk level, obligations, rights, red_flags.
        Each section gains a `classifier` key when use_classifier=True.
    """
    if not text or not text.strip():
        raise ValueError("Document text is empty")

    was_truncated = len(text) > MAX_CHARS
    text_to_send  = text[:MAX_CHARS] + ("\n\n[Document truncated for length]" if was_truncated else "")

    # ── Step 1: RAG context retrieval ─────────────────────────────────────────
    context = ""
    if use_rag and os.environ.get("PINECONE_API_KEY"):
        try:
            from rag import retrieve_context
            context = retrieve_context(text_to_send[:512])
        except Exception:
            context = ""   # non-critical; continue without context

    # ── Step 2: Ollama simplification ─────────────────────────────────────────
    if context:
        prompt = SIMPLIFY_PROMPT_WITH_CONTEXT.format(
            document=text_to_send, context=context
        )
    else:
        prompt = SIMPLIFY_PROMPT.format(document=text_to_send)

    raw = _ollama_generate(prompt)

    try:
        result = _parse_json(raw)
    except Exception:
        raise ValueError(f"Ollama returned non-JSON: {raw[:200]}")

    result["was_truncated"] = was_truncated
    result["rag_used"]      = bool(context)

    # ── Step 3: NLP classifier ────────────────────────────────────────────────
    if use_classifier and result.get("sections"):
        try:
            from classifier import classify_sections
            result["sections"] = classify_sections(
                result["sections"],
                use_zero_shot=classifier_zero_shot,
            )
        except Exception:
            pass   # classifier is enhancement-only; never break the response

    return result


# ── File parsers ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file given as raw bytes."""
    import PyPDF2
    import io
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a .docx file given as raw bytes."""
    import docx
    import io
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
