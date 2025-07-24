"""
pdf_reader.py
-------------
Extracts text from PDF and TXT files for downstream embedding.
- Uses PyMuPDF (fitz) for PDFs
- Handles plain text files
"""

import fitz  # PyMuPDF
import os
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.
    Returns a single concatenated string of all pages.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError("File is not a PDF.")
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """
    Extracts all text from a plain text file.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File not found: {txt_path}")
    if not txt_path.lower().endswith(".txt"):
        raise ValueError("File is not a TXT file.")
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_texts_from_folder(folder_path: str, exts: List[str] = [".pdf", ".txt"]) -> List[str]:
    """
    Extracts text from all PDF and TXT files in a folder.
    Returns a list of extracted texts (one per file).
    """
    texts = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if fname.lower().endswith(".pdf"):
            texts.append(extract_text_from_pdf(fpath))
        elif fname.lower().endswith(".txt"):
            texts.append(extract_text_from_txt(fpath))
    return texts

if __name__ == "__main__":
    # Example usage
    pdf_path = "docs/domain_guide.pdf"  # Update as needed
    txt_path = "docs/notes.txt"         # Update as needed
    if os.path.exists(pdf_path):
        print(f"PDF text sample:\n{extract_text_from_pdf(pdf_path)[:500]}\n...")
    if os.path.exists(txt_path):
        print(f"TXT text sample:\n{extract_text_from_txt(txt_path)[:500]}\n...")
