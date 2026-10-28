# app/rag.py - RAG pipeline: store documents, search by meaning, answer with Gemini

import os
import re
import pdfplumber
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()
# LLMs to try, in order. If one fails, the next one is used.
LLM_CHAIN = [
    ("gemini", "gemini-3.8-flash"),
    ("groq",   "openai/gpt-oss-120b"),
    ("groq",   "qwen/qwen3.8-27b"),
    ("groq",   "openai/gpt-oss-20b"),
]
THRESHOLD = 0.25                    # minimum similarity to count as "relevant"

# ---------- Setup ----------
db = chromadb.PersistentClient(path="chroma_db")              # saves data in a folder
embed = embedding_functions.DefaultEmbeddingFunction()        # text -> meaning codes (MiniLM)
gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=120_000),  # give up after 120 seconds
)
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",        # Groq's OpenAI-style address
    timeout=60,                                       # seconds
)


def get_collection(name):
    """Open (or create) a collection - like a table in the vector database."""
    return db.get_or_create_collection(
        name, embedding_function=embed, metadata={"hnsw:space": "cosine"})


# ---------- Reading and chunking ----------
def read_pdf_pages(path):
    """Return a list of (page_number, text) for every page of a PDF."""
    with pdfplumber.open(path) as pdf:
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]


def chunk_text(text, size=800, overlap=150):
    """Cut long text into overlapping pieces of about `size` characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def split_clauses(text):
    """Split a contract into clauses at numbers like '1.' or '2)' at the start of a line."""
    parts = re.split(r"\n\s*(?=\d{1,2}[.)]\s)", "\n" + text)
    return [p.strip().replace("\n", " ") for p in parts if len(p.strip()) > 30]


# ---------- Building the knowledge stores ----------
def add_in_batches(col, ids, docs, metas, batch=100):
    for i in range(0, len(ids), batch):
        col.add(ids=ids[i:i + batch], documents=docs[i:i + batch],
                metadatas=metas[i:i + batch])


def build_law_library(folder="data/laws"):
    """Read every Act PDF once, chunk it and store it in the 'indian_law' collection."""
    col = get_collection("indian_law")
    if col.count() > 0:
        print("Law library already built:", col.count(), "chunks")
        return
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".pdf"):
            continue
        act = file.replace(".pdf", "").replace("_", " ")
        ids, docs, metas = [], [], []
        for page_no, text in read_pdf_pages(os.path.join(folder, file)):
            for j, chunk in enumerate(chunk_text(text)):
                ids.append(f"{file}_p{page_no}_{j}")
                docs.append(chunk)
                metas.append({"source": act, "page": page_no})
        add_in_batches(col, ids, docs, metas)
        print(f"Added {len(docs)} chunks from {act}")


def index_contract(text, doc_id):
    """Split one user's contract into clauses and store them in their own collection."""
    name = f"contract_{doc_id}"
    try:
        db.delete_collection(name)      # start fresh if this document was indexed before
    except Exception:
        pass
    col = get_collection(name)
    clauses = split_clauses(text)
    add_in_batches(col,
                   ids=[f"c{i + 1}" for i in range(len(clauses))],
                   docs=clauses,
                   metas=[{"clause_no": i + 1} for i in range(len(clauses))])
    return clauses


# ---------- Retrieval (the R in RAG) ----------
def search(collection_name, question, k=3):
    """Find the k most similar chunks; keep only those above THRESHOLD."""
    col = get_collection(collection_name)
    if col.count() == 0:
        return []
    res = col.query(query_texts=[question], n_results=min(k, col.count()))
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "meta": meta, "score": round(1 - dist, 3)})
    return [h for h in hits if h["score"] >= THRESHOLD]


# ---------- LLM calls with fallback ----------
def ask_gemini(model, prompt):
    response = gemini_client.models.generate_content(model=model, contents=prompt)
    return response.text


def ask_groq(model, prompt):
    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,                  # low = more factual, less creative
    )
    return response.choices[0].message.content


def call_llm(prompt):
    """Try each LLM in LLM_CHAIN until one answers."""
    for provider, model in LLM_CHAIN:
        try:
            if provider == "gemini":
                text = ask_gemini(model, prompt)
            else:
                text = ask_groq(model, prompt)
            if text:                                  # got a real answer
                print(f"[LLM] answered by {provider}/{model}")
                return text
        except Exception as e:                        # retired, rate-limited, busy, timeout...
            print(f"[LLM] {provider}/{model} failed: {str(e)[:120]}")
    return "Sorry, the AI service is unavailable right now. Please try again in a minute."


# ---------- Augment + Generate (the A and G in RAG) ----------
def answer(question, doc_id):
    contract_hits = search(f"contract_{doc_id}", question)
    law_hits = search("indian_law", question)

    if not contract_hits and not law_hits:
        return "I couldn't find anything relevant in your document or the law library.", [], []

    contract_ctx = "\n".join(
        f"[Clause {h['meta']['clause_no']}] {h['text']}" for h in contract_hits) or "Nothing relevant found."
    law_ctx = "\n".join(
        f"[{h['meta']['source']}, page {h['meta']['page']}] {h['text']}" for h in law_hits) or "Nothing relevant found."

    prompt = f"""You are a contract assistant for people in India. You help ordinary people
understand documents they are asked to sign, such as rent / leave-and-licence agreements,
offer letters and appointment letters, and similar contracts.

Rules:
- Answer ONLY using the context below. Do not use outside knowledge.
- If the context does not contain the answer, say clearly that it is not in the document or the law library.
- First work out what kind of document this is (for example rent agreement or offer letter) and answer for that situation.
- Look at it from the point of view of the person signing it (the tenant or the employee).
- Write two short sections: "Your document says" and "The law says".
  If no relevant law was found, write "No matching law was found in the library" under "The law says".
- Cite clause numbers like [Clause 4] and law sources like [Act name, page X].
- If a clause looks one-sided or unfair to the signer, point it out briefly.
- Use simple English that a non-lawyer can understand.
- End with: "This is general information, not legal advice."

YOUR DOCUMENT:
{contract_ctx}

APPLICABLE LAW:
{law_ctx}

QUESTION: {question}"""

    reply = call_llm(prompt)
    return reply, contract_hits, law_hits
