# app/parser.py - read documents and split them into clauses

import re
import pdfplumber


def read_pdf(file):
    """Read all text from a PDF. `file` can be a path or an uploaded file."""
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def split_clauses(text):
    """Split a contract into clauses at '1.', '2)' or 'Clause 3.' at the start of a line."""
    parts = re.split(r"\n\s*(?=(?:clause\s+)?\d{1,2}[.)]\s)", "\n" + text, flags=re.IGNORECASE)
    clauses = [p.strip().replace("\n", " ") for p in parts if len(p.strip()) > 30]

    # Fallback: no numbering found -> split into sentences instead
    if len(clauses) <= 1:
        sentences = re.split(r"(?<=[.;])\s+", text.replace("\n", " "))
        clauses = [s.strip() for s in sentences if len(s.strip()) > 30]

    return clauses