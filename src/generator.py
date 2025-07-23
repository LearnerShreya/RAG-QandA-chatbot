"""
generator.py
------------
Gemini LLM integration for answer generation in the RAG pipeline.
- Uses official Google GenerativeAI SDK
- Reads API key from environment variable
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List

load_dotenv()

def get_gemini_llm():
    """
    Instantiates the Gemini LLM using the API key from the environment, using the gemini-1.5-flash model for faster responses.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")

def generate_answer(llm, question: str, context: List[str]) -> str:
    """
    Generates an answer using the LLM, given a question and a list of context strings.
    """
    system_prompt = (
        "You are a helpful loan approval assistant. Use the provided context to answer the user's question. "
        "If the answer is not in the context, say 'I don't know based on the provided information.'"
    )
    context_str = "\n\n".join(context)
    prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {question}\nAnswer:"
    response = llm.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    # Example usage
    try:
        llm = get_gemini_llm()
        sample_context = [
            "A good credit history increases your chances of loan approval.",
            "Stable income and low existing debt are also important factors."
        ]
        question = "What increases the chances of getting a home loan?"
        answer = generate_answer(llm, question, sample_context)
        print(f"Q: {question}\nA: {answer}")
    except Exception as e:
        print(f"Error: {e}")
