# app/parser.py - read documents and split them into clauses

import re
import pdfplumber

# Patterns for personal numbers (Indian formats)
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")                 # e.g. ABCDE1234F
AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")             # 12 digits, spaces optional
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b")           # 10-digit mobile
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Pieces containing any of these are party details or registration headers, not clauses
NON_CLAUSE_RE = re.compile(
    r"\bPAN\b|Aadhaar|Aadhar|Residing at|\bAge\s*:|\bGRN\b|Stamp Duty|Transaction Id",
    re.IGNORECASE,
)


def read_pdf(file):
    """Read all text from a PDF. `file` can be a path or an uploaded file."""
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def mask_personal_info(text):
    """Replace personal numbers so they are never stored or sent to an LLM."""
    text = PAN_RE.sub("[PAN]", text)
    text = AADHAAR_RE.sub("[AADHAAR]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return text


def is_clause(piece):
    """False for party details and stamp/registration headers."""
    return not NON_CLAUSE_RE.search(piece)


def split_clauses(text):
    """Split a contract into clauses at '1.', '2)' or 'Clause 3.' at the start of a line."""
    parts = re.split(r"\n\s*(?=(?:clause\s+)?\d{1,2}[.)]\s?)", "\n" + text, flags=re.IGNORECASE)
    pieces = [p.strip().replace("\n", " ") for p in parts if len(p.strip()) > 30]

    # Fallback: no numbering found -> split into sentences instead
    if len(pieces) <= 1:
        sentences = re.split(r"(?<=[.;])\s+", text.replace("\n", " "))
        pieces = [s.strip() for s in sentences if len(s.strip()) > 30]

    clauses = [mask_personal_info(p) for p in pieces if is_clause(p)]
    return clauses