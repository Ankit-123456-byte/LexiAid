"""
translator.py — Ollama-powered translation of simplified legal output.

Only translates human-readable text values; JSON keys and risk level
tokens ("low", "medium", "high", "none") are kept in English so the
frontend logic doesn't break.
"""

import os
import json
import re
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")

SUPPORTED_LANGUAGES: dict = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ur": "Urdu (اردو)",
}

TRANSLATE_PROMPT = """\
Translate the JSON values below into {language}.

Rules:
- Keep ALL JSON keys exactly as written — never translate keys.
- Keep risk tokens ("low", "medium", "high", "none") in English.
- Translate every other string value naturally and accurately.
- Preserve list structure; translate each list item.
- Return ONLY valid JSON — no markdown fences, no explanation.

JSON:
{json_data}
"""


from json_repair import repair_json

def _parse_json(text: str) -> dict:
    """Parse JSON from model output, repairing common issues."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group() if match else text
    return json.loads(repair_json(raw))


def _ollama_generate(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def translate_result(result: dict, target_lang: str) -> dict:
    """
    Translate the text fields of a simplify_document() result.

    Args:
        result:      Dict returned by simplify_document()
        target_lang: BCP-47 language code from SUPPORTED_LANGUAGES

    Returns:
        New dict with translated text fields; structure is identical.
        Falls back to the original result if translation fails.
    """
    if target_lang == "en" or target_lang not in SUPPORTED_LANGUAGES:
        return result

    language_name = SUPPORTED_LANGUAGES[target_lang]

    # Build a minimal translatable payload (only human-readable text)
    translatable = {
        "summary":          result.get("summary", ""),
        "risk_reason":      result.get("risk_reason", ""),
        "key_obligations":  result.get("key_obligations", []),
        "key_rights":       result.get("key_rights", []),
        "red_flags":        result.get("red_flags", []),
        "sections": [
            {
                "heading":    s.get("heading", ""),
                "simplified": s.get("simplified", ""),
                "risk_note":  s.get("risk_note", ""),
            }
            for s in result.get("sections", [])
        ],
    }

    prompt = TRANSLATE_PROMPT.format(
        language=language_name,
        json_data=json.dumps(translatable, ensure_ascii=False, indent=2),
    )

    raw = _ollama_generate(prompt)

    try:
        translated = _parse_json(raw)
    except Exception:
        raise ValueError(f"Translation returned non-JSON: {raw[:300]}")

    # Merge translated fields back into a full result copy
    merged = result.copy()
    merged["summary"]         = translated.get("summary",         result["summary"])
    merged["risk_reason"]     = translated.get("risk_reason",     result["risk_reason"])
    merged["key_obligations"] = translated.get("key_obligations", result["key_obligations"])
    merged["key_rights"]      = translated.get("key_rights",      result["key_rights"])
    merged["red_flags"]       = translated.get("red_flags",       result["red_flags"])
    merged["translated_to"]   = target_lang

    t_sections = translated.get("sections", [])
    for i, section in enumerate(merged.get("sections", [])):
        if i < len(t_sections):
            section["heading"]    = t_sections[i].get("heading",    section["heading"])
            section["simplified"] = t_sections[i].get("simplified", section["simplified"])
            section["risk_note"]  = t_sections[i].get("risk_note",  section["risk_note"])

    return merged
