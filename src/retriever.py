"""
retriever.py
------------
Loads the FAISS index and retrieves top-k relevant chunks for a query.
- Uses the same embedding model as for indexing
- Easy to use in RAG pipeline
"""

from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import os

FAISS_INDEX_PATH = "embeddings/faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_faiss_retriever(index_path: str = FAISS_INDEX_PATH, model_name: str = EMBED_MODEL):
    """
    Loads the FAISS index and returns a retriever object.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run embedder.py first.")
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()


def retrieve_top_k(query: str, k: int = 5) -> List[str]:
    """
    Retrieves the top-k most relevant chunks for a query.
    Returns a list of text chunks.
    """
    retriever = load_faiss_retriever()
    docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in docs[:k]]

if __name__ == "__main__":
    # Example usage
    test_query = "What increases the chances of getting a home loan?"
    print(f"Query: {test_query}")
    results = retrieve_top_k(test_query, k=3)
    for i, chunk in enumerate(results, 1):
        print(f"\nResult {i}:\n{chunk[:500]}\n...")