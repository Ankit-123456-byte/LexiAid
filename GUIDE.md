# LexiAid — Step-by-Step Build Guide

**What it is:** An AI tool that takes any legal document — contract, rental agreement,
NDA, terms of service — and explains it in plain language, flags risky clauses,
and tells you what to watch out for.

**Tech stack:** FastAPI · Claude API · Python · HTML/JS (no framework)

---

## ✅ Phase 1 — Core Simplifier (DONE — run it now)

You have a working app. To start:

```bash
cd tools/lexiaid

# Copy .env and add your Anthropic API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# Start the server
source .env
python3 main.py
```

Open http://localhost:8000 in your browser.

**What works now:**
- Paste any legal text → Claude simplifies it
- Upload a PDF or DOCX file
- See: plain-language summary, risk level, red flags, obligations, rights
- Clause-by-clause breakdown with risk badges
- High/medium risk clauses auto-expand

**Test it with:** Paste a sample terms of service or rental agreement.
You can find sample contracts at: https://commonpaper.com (free, open-source contracts)

---

## 🔜 Phase 2 — Language Translation

**Goal:** Add a language selector. After simplification, translate the output to
Hindi, Bengali, Tamil, or any Indian language.

**How:**
1. Add a language dropdown to `templates/index.html`
2. After getting Claude's JSON result, call Claude again with a translation prompt
3. Or: use `deep-translator` (free, no API key needed) for quick translation

**Files to edit:** `simplifier.py` (add `translate_result()` function), `main.py` (new `/translate` endpoint), `templates/index.html` (add language selector)

**Install:** `pip install deep-translator`

**Key prompt for Claude translation:**
```
Translate the following simplified legal summary to {language}.
Keep the JSON structure the same. Only translate the text values, not the keys.
{json_result}
```

---

## 🔜 Phase 3 — RAG: Jurisdiction-Specific Legal Context

**Goal:** Add a knowledge base of Indian law (Indian Contract Act, Rent Control Acts,
Consumer Protection Act) so LexiAid can say "This clause may be unenforceable under
the Indian Consumer Protection Act 2019".

**How:**
1. Create `rag/` folder with markdown files containing legal summaries
2. Use `sentence-transformers` to embed them into FAISS vector store
3. When simplifying, retrieve relevant law snippets and include in Claude's prompt

**Install:**
```bash
pip install sentence-transformers faiss-cpu
```

**Files to create:**
- `rag/knowledge_base/indian_contract_act.md`
- `rag/knowledge_base/consumer_protection.md`
- `rag/knowledge_base/rent_control.md`
- `rag/embeddings.py` — build + save FAISS index
- `rag/retriever.py` — query FAISS, return top-k relevant chunks

**Modified simplify prompt:** Add retrieved chunks as context:
```
RELEVANT INDIAN LAW (use this to flag illegal/unenforceable clauses):
{retrieved_legal_context}

DOCUMENT TO SIMPLIFY:
{document}
```

---

## 🔜 Phase 4 — Better UI & Shareable Reports

**Goal:** Let users download a simplified version as a PDF or share a link.

**How:**
1. Add "Download as PDF" button — use `reportlab` (you already know how to use this!)
2. Add "Copy simplified text" button
3. Optional: save results to a local SQLite database so users can revisit

**Install:** `pip install reportlab` (already in your environment)

---

## 🔜 Phase 5 — Deploy to the Internet

**Goal:** Make LexiAid publicly accessible so you can show it to clients, put it on your
LinkedIn/GitHub, and let real users test it.

**Easiest free option: Railway.app**

```bash
# 1. Create Procfile in tools/lexiaid/
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# 2. Push to GitHub (make a new repo: lexiaid)
git init && git add . && git commit -m "Initial LexiAid"
git remote add origin https://github.com/YOUR_USERNAME/lexiaid.git
git push -u origin main

# 3. Go to railway.app → New Project → Deploy from GitHub
# 4. Add environment variable: ANTHROPIC_API_KEY=sk-ant-...
# 5. Done — you get a public URL like lexiaid.up.railway.app
```

**Alternative: Render.com** (also free, similar process)

---

## File Structure

```
tools/lexiaid/
├── main.py           ← FastAPI app (routes)
├── simplifier.py     ← Claude simplification logic + PDF/DOCX parsing
├── templates/
│   └── index.html    ← Web UI (all frontend code)
├── static/           ← (future: CSS files, images)
├── rag/              ← (Phase 3: legal knowledge base + FAISS)
├── requirements.txt
├── .env.example
└── GUIDE.md          ← this file
```

---

## Troubleshooting

**Server won't start:**
- Check `ANTHROPIC_API_KEY` is set: `echo $ANTHROPIC_API_KEY`
- Check you're in the right directory: `pwd` should end in `tools/lexiaid`
- Check port 8000 is free: `lsof -i :8000`

**Claude returns an error:**
- Check API key is valid and has credits
- Try with a shorter document first

**PDF upload fails:**
- Make sure the PDF has extractable text (not a scanned image)
- Image-based PDFs need OCR (Phase 4 enhancement)

---

## What to build next after LexiAid?

Once you have Phase 1–3 working and deployed, this project becomes your portfolio
flagship. What to do with it:
1. Post a demo GIF on LinkedIn: "I built an AI that explains legal documents in plain language"
2. Make a GitHub repo with a clear README — pin it on your profile
3. Share it in the IndiaAI community and AI-for-good Slack groups
4. Pitch it to NGOs working with migrant workers or rural communities
5. Then build FarmSense or AquaAlert using the same FastAPI + Claude + RAG pattern
