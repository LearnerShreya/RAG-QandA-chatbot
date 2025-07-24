"""
embedder.py
-----------
Embeds all text data (CSV + docs) and builds a FAISS index for semantic retrieval.
- Uses all-MiniLM-L6-v2 (sentence-transformers)
- Saves FAISS index for later use
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List
from src.preprocess import load_and_clean_csv, dataframe_to_chunks
from src.pdf_reader import extract_texts_from_folder

# Move all paths and chunk size to constants
DATA_CSV = "data/loan_data.csv.csv"  # Update if needed
DOCS_FOLDER = "docs/"
FAISS_INDEX_PATH = "embeddings/faiss_index"
CHUNK_SIZE = 300


def get_all_text_chunks() -> List[str]:
    """
    Loads and combines all text chunks from CSV and docs.
    """
    # CSV chunks
    df = load_and_clean_csv(DATA_CSV)
    fields = [col for col in df.columns if col != "loan_id"]
    csv_chunks = dataframe_to_chunks(df, fields, max_length=CHUNK_SIZE)
    # Docs chunks
    doc_texts = extract_texts_from_folder(DOCS_FOLDER)
    doc_chunks = []
    for text in doc_texts:
        # Chunk long doc text for embedding
        doc_chunks.extend([text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)])
    return csv_chunks + doc_chunks


def build_and_save_faiss_index(texts: List[str], index_path: str = FAISS_INDEX_PATH):
    """
    Embeds texts and saves a FAISS index to disk.
    """
    print(f"Loading embedding model (all-MiniLM-L6-v2)...")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"Embedding {len(texts)} chunks...")
    vectorstore = FAISS.from_texts(texts, embedding=embedding)
    print(f"Saving FAISS index to {index_path} ...")
    vectorstore.save_local(index_path)
    print("Done!")

if __name__ == "__main__":
    all_chunks = get_all_text_chunks()
    print(f"Total text chunks to embed: {len(all_chunks)}")
    build_and_save_faiss_index(all_chunks)
