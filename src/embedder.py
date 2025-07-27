"""
embedder.py
-----------
Embeds all text data (CSV + docs) and builds a FAISS index for semantic retrieval.
- Uses all-MiniLM-L6-v2 (sentence-transformers)
- Saves FAISS index for later use
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import sys
from typing import List

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_clean_csv, dataframe_to_chunks
from src.pdf_reader import extract_texts_from_folder


DATA_CSV = "data/loan_data.csv.csv"
DOCS_FOLDER = "docs/"
FAISS_INDEX_PATH = "embeddings"
CHUNK_SIZE = 300


def get_all_text_chunks() -> List[str]:
    """
    Loads and combines all text chunks from CSV and docs.
    """
    df = load_and_clean_csv(DATA_CSV)
    fields = [col for col in df.columns if col != "loan_id"]
    csv_chunks = dataframe_to_chunks(df, fields, max_length=CHUNK_SIZE)
    doc_texts = extract_texts_from_folder(DOCS_FOLDER)
    doc_chunks = []
    for text in doc_texts:
        doc_chunks.extend([text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)])
    return csv_chunks + doc_chunks


def build_and_save_faiss_index(texts: List[str], index_path: str = FAISS_INDEX_PATH):
    """
    Embeds texts and saves a FAISS index to disk.
    """
    # Set device explicitly to avoid meta tensor issues
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts, embedding=embedding)
    vectorstore.save_local(index_path)

if __name__ == "__main__":
    all_chunks = get_all_text_chunks()
    build_and_save_faiss_index(all_chunks)
