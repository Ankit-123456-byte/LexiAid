"""
main.py — LexiAid FastAPI application.

Routes:
  GET  /              → web UI (index.html)
  GET  /health        → health check
  POST /simplify      → analyse a document (text or file upload)
  POST /translate     → translate an existing result to a target language
  GET  /admin         → document ingestion admin page
  POST /admin/ingest  → add a legal reference document to Pinecone
  GET  /admin/docs    → list all ingested documents
  DELETE /admin/docs/{doc_id} → remove a document from Pinecone
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from simplifier import simplify_document, extract_text_from_pdf, extract_text_from_docx
from translator import translate_result, SUPPORTED_LANGUAGES

app = FastAPI(title="LexiAid", description="Legal document simplifier")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Models ─────────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    result: dict
    target_lang: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    pinecone_configured = bool(os.environ.get("PINECONE_API_KEY"))
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    return {
        "status": "ok",
        "model": f"ollama/{ollama_model}",
        "rag": pinecone_configured,
        "classifier": True,
        "translation": True,
    }


# ── /simplify ──────────────────────────────────────────────────────────────────

@app.post("/simplify")
async def simplify(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    use_classifier: bool = Form(True),
    classifier_zero_shot: bool = Form(False),   # False = rule-based only (fast default)
):
    """
    Accept either:
      - text: raw pasted legal text (Form field)
      - file: uploaded .pdf or .docx

    Optional flags:
      - use_classifier:        run NLP clause classifier (default True)
      - classifier_zero_shot:  use HuggingFace zero-shot model (default False = rule-based only)

    Returns: structured JSON with simplified document analysis.
    """
    document_text = ""

    if file and file.filename:
        file_bytes = await file.read()
        filename   = file.filename.lower()

        if filename.endswith(".pdf"):
            try:
                document_text = extract_text_from_pdf(file_bytes)
            except Exception as e:
                raise HTTPException(400, f"Could not read PDF: {e}")

        elif filename.endswith(".docx"):
            try:
                document_text = extract_text_from_docx(file_bytes)
            except Exception as e:
                raise HTTPException(400, f"Could not read DOCX: {e}")

        else:
            raise HTTPException(400, "Only .pdf and .docx files are supported")

    elif text and text.strip():
        document_text = text.strip()

    else:
        raise HTTPException(400, "Provide either 'text' or a file upload")

    if len(document_text.strip()) < 50:
        raise HTTPException(400, "Document is too short to analyse")

    try:
        result = simplify_document(
            document_text,
            use_rag=True,
            use_classifier=use_classifier,
            classifier_zero_shot=classifier_zero_shot,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Simplification failed: {e}")

    return JSONResponse(result)


# ── /translate ─────────────────────────────────────────────────────────────────

@app.post("/translate")
async def translate(req: TranslateRequest):
    """
    Translate an existing simplify_document() result to a target language.

    Body JSON:
      { "result": { ...full result dict... }, "target_lang": "hi" }

    Returns: translated result dict (same structure).
    """
    if req.target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            400,
            f"Unsupported language '{req.target_lang}'. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}",
        )
    try:
        translated = translate_result(req.result, req.target_lang)
    except Exception as e:
        raise HTTPException(500, f"Translation failed: {e}")

    return JSONResponse(translated)


# ── /admin ─────────────────────────────────────────────────────────────────────

ADMIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>LexiAid Admin — Ingest Legal Documents</title>
  <style>
    body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0d1117;color:#e6edf3;padding:40px;max-width:800px;margin:0 auto}
    h1{color:#58a6ff;margin-bottom:8px}
    .sub{color:#8b949e;margin-bottom:32px}
    label{display:block;margin-bottom:6px;font-size:14px;font-weight:600;color:#8b949e}
    input,select,textarea{width:100%;padding:10px 12px;background:#161b22;border:1px solid #30363d;
      border-radius:6px;color:#e6edf3;font-size:14px;margin-bottom:16px;font-family:inherit}
    textarea{height:200px;resize:vertical}
    button{background:#58a6ff;color:#000;border:none;padding:12px 24px;border-radius:6px;
      font-size:15px;font-weight:700;cursor:pointer}
    button:hover{opacity:.85}
    #msg{margin-top:20px;padding:12px 16px;border-radius:6px;display:none}
    #msg.ok{background:rgba(63,185,80,.1);border:1px solid rgba(63,185,80,.3);color:#3fb950}
    #msg.err{background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);color:#f85149}
    table{width:100%;border-collapse:collapse;margin-top:32px}
    th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #21262d;font-size:14px}
    th{color:#8b949e;font-weight:600}
    .del-btn{background:#f85149;color:#fff;border:none;padding:4px 10px;border-radius:4px;
      cursor:pointer;font-size:12px}
  </style>
</head>
<body>
<h1>LexiAid Admin</h1>
<p class="sub">Ingest Indian legal reference documents into Pinecone for RAG-enhanced analysis.</p>

<form id="form">
  <label>Document Name / Source</label>
  <input type="text" id="doc_name" placeholder="e.g. Indian Contract Act — Section 73" required/>

  <label>Document Type</label>
  <select id="doc_type">
    <option value="act">Central Act</option>
    <option value="regulation">Regulation</option>
    <option value="guideline">Guideline / Circular</option>
    <option value="judgment">Court Judgment</option>
    <option value="other">Other</option>
  </select>

  <label>Jurisdiction (optional)</label>
  <input type="text" id="jurisdiction" placeholder="e.g. India, Maharashtra, Pan-India"/>

  <label>Document Text</label>
  <textarea id="doc_text" placeholder="Paste the full text of the legal document or section here..." required></textarea>

  <button type="submit">Ingest into Pinecone</button>
</form>

<div id="msg"></div>

<h2 style="margin-top:40px;font-size:18px;color:#8b949e">Ingested Documents</h2>
<table id="docs-table">
  <thead><tr><th>Document ID</th><th>Source</th><th>Type</th><th></th></tr></thead>
  <tbody id="docs-body"><tr><td colspan="4" style="color:#8b949e">Loading...</td></tr></tbody>
</table>

<script>
async function loadDocs() {
  try {
    const r = await fetch('/admin/docs');
    const docs = await r.json();
    const tbody = document.getElementById('docs-body');
    if (!docs.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#8b949e">No documents ingested yet.</td></tr>';
      return;
    }
    tbody.innerHTML = docs.map(d => `
      <tr>
        <td style="font-family:monospace;font-size:12px">${d.doc_id}</td>
        <td>${d.source || '-'}</td>
        <td>${d.type || '-'}</td>
        <td><button class="del-btn" onclick="deleteDoc('${d.doc_id}')">Delete</button></td>
      </tr>`).join('');
  } catch(e) {
    document.getElementById('docs-body').innerHTML =
      '<tr><td colspan="4" style="color:#f85149">Could not load (Pinecone not configured?)</td></tr>';
  }
}

document.getElementById('form').addEventListener('submit', async e => {
  e.preventDefault();
  const msg = document.getElementById('msg');
  msg.style.display = 'none';
  const body = new FormData();
  body.append('doc_name', document.getElementById('doc_name').value.trim());
  body.append('doc_type', document.getElementById('doc_type').value);
  body.append('jurisdiction', document.getElementById('jurisdiction').value.trim());
  body.append('text', document.getElementById('doc_text').value.trim());
  try {
    const r = await fetch('/admin/ingest', { method: 'POST', body });
    const data = await r.json();
    if (r.ok) {
      msg.className = 'ok'; msg.style.display = 'block';
      msg.textContent = `Ingested ${data.chunks} chunks for "${data.doc_id}"`;
      document.getElementById('form').reset();
      loadDocs();
    } else {
      msg.className = 'err'; msg.style.display = 'block';
      msg.textContent = data.detail || 'Ingestion failed';
    }
  } catch(e) {
    msg.className = 'err'; msg.style.display = 'block';
    msg.textContent = 'Network error';
  }
});

async function deleteDoc(doc_id) {
  if (!confirm(`Delete all chunks for "${doc_id}"?`)) return;
  const r = await fetch(`/admin/docs/${encodeURIComponent(doc_id)}`, { method: 'DELETE' });
  if (r.ok) loadDocs();
}

loadDocs();
</script>
</body>
</html>"""


@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    return HTMLResponse(ADMIN_PAGE)


@app.post("/admin/ingest")
async def admin_ingest(
    text:         str  = Form(...),
    doc_name:     str  = Form(...),
    doc_type:     str  = Form("other"),
    jurisdiction: str  = Form("India"),
):
    """
    Ingest a legal reference document into Pinecone.

    Form fields:
      text:         Full document text
      doc_name:     Human-readable name (used as source label)
      doc_type:     Category (act / regulation / guideline / judgment / other)
      jurisdiction: Jurisdiction tag (default "India")
    """
    if not os.environ.get("PINECONE_API_KEY"):
        raise HTTPException(503, "PINECONE_API_KEY is not configured on this server")

    if not text.strip():
        raise HTTPException(400, "Document text is required")
    if not doc_name.strip():
        raise HTTPException(400, "doc_name is required")

    # Stable doc_id: slug of doc_name
    import re as _re
    doc_id = _re.sub(r"[^a-z0-9]+", "-", doc_name.lower().strip()).strip("-")[:80]

    try:
        from rag import ingest_document
        chunks = ingest_document(
            text=text.strip(),
            doc_id=doc_id,
            metadata={
                "source":       doc_name.strip(),
                "type":         doc_type,
                "jurisdiction": jurisdiction.strip() or "India",
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Pinecone ingestion failed: {e}")

    return JSONResponse({"doc_id": doc_id, "chunks": chunks, "status": "ok"})


@app.get("/admin/docs")
async def admin_list_docs():
    """List all ingested documents (deduplicated by doc_id)."""
    if not os.environ.get("PINECONE_API_KEY"):
        return JSONResponse([])
    try:
        from rag import list_documents
        return JSONResponse(list_documents())
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/admin/docs/{doc_id}")
async def admin_delete_doc(doc_id: str):
    """Delete all Pinecone chunks for a given doc_id."""
    if not os.environ.get("PINECONE_API_KEY"):
        raise HTTPException(503, "PINECONE_API_KEY is not configured")
    try:
        from rag import delete_document
        delete_document(doc_id)
        return JSONResponse({"status": "deleted", "doc_id": doc_id})
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
