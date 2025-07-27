"""
retriever.py
------------
Loads the FAISS index and retrieves top-k relevant chunks for a query.
- Uses the same embedding model as for indexing
- Easy to use in RAG pipeline
- Enhanced retrieval for better RAG + LLM performance
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os

FAISS_INDEX_PATH = "embeddings"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_faiss_retriever(index_path: str = FAISS_INDEX_PATH, model_name: str = EMBED_MODEL):
    """
    Loads the FAISS index and returns a retriever object.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run 'python src/embedder.py' to build the index.")
    
    embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()


def retrieve_top_k(query: str, k: int = 8) -> List[str]:
    """
    Enhanced retrieval for RAG + LLM. Retrieves more chunks for better context coverage.
    Returns a list of text chunks with improved relevance.
    """
    retriever = load_faiss_retriever()
    docs = retriever.invoke(query)
    
    # Enhanced retrieval strategy:
    # 1. Get more chunks for better coverage
    # 2. Filter out very short or irrelevant chunks
    # 3. Ensure diverse context for the LLM
    
    relevant_chunks = []
    for doc in docs[:k]:
        content = doc.page_content.strip()
        # Filter out very short chunks that might not be useful
        if len(content) > 20 and not content.isspace():
            relevant_chunks.append(content)
    
    # If we don't have enough relevant chunks, return what we have
    if not relevant_chunks:
        return [doc.page_content for doc in docs[:k]]
    
    return relevant_chunks

if __name__ == "__main__":
    pass