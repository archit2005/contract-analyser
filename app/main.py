# app/main.py - FastAPI backend: upload a contract, ask questions about it

import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.parser import read_pdf
from app.pipeline import process_contract
from app.rag import build_law_library, answer

app = FastAPI(title="Contract Analyzer API")
build_law_library()                                   # make sure the law library exists


class Question(BaseModel):
    doc_id: str
    question: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    name = file.filename.lower()
    if name.endswith(".pdf"):
        text = read_pdf(file.file)
    elif name.endswith(".txt"):
        text = file.file.read().decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Please upload a PDF or TXT file.")

    if len(text.strip()) < 50:
        raise HTTPException(status_code=400,
                            detail="Couldn't read any text. Is this a scanned PDF?")

    doc_id = uuid.uuid4().hex[:8]                     # unique id for this document
    report = process_contract(text, doc_id)
    risky = sum(1 for r in report if r["risk"] == "Risky")

    return {"doc_id": doc_id, "filename": file.filename,
            "total": len(report), "risky": risky, "clauses": report}


@app.post("/ask")
def ask(q: Question):
    reply, c_hits, l_hits = answer(q.question, q.doc_id)

    sources = [{"label": f"Clause {h['meta']['clause_no']}", "text": h["text"], "score": h["score"]}
               for h in c_hits]
    sources += [{"label": f"{h['meta']['source']}, page {h['meta']['page']}", "text": h["text"],
                 "score": h["score"]} for h in l_hits]

    return {"answer": reply, "sources": sources}