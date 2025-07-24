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

def generate_answer(llm, question: str, context: list, chat_history: list = None, language: str = "English") -> str:
    """
    Generates an answer using the LLM. If context is empty or weak, uses Gemini's general knowledge; otherwise, uses RAG context. Includes recent chat history for multi-turn memory. Answers in the selected language.
    """
    chat_history = chat_history or []
    # Format chat history as a string
    history_str = ""
    if chat_history:
        history_str = "\n".join([
            f"User: {q}\nBot: {a}" for q, a in chat_history[-4:]
        ])
    context_str = "\n\n".join(context).strip()
    context_is_weak = not context_str or len(context_str) < 30
    lang_instruction = f"Please answer in {language}."
    if context_is_weak:
        prompt = (
            f"You are an expert loan advisor. Even if the provided information is insufficient, always try to answer the user's question using your own knowledge and reasoning. "
            f"Be specific, friendly, and proactive. If you don't know the exact answer, provide your best possible explanation or general information. {lang_instruction}\n\n"
            f"Recent chat:\n{history_str}\n\nUser: {question}"
        )
        response = llm.generate_content(prompt)
        return response.text.strip()
    else:
        system_prompt = (
            f"You are a helpful loan approval assistant. Use the provided context and recent conversation to answer the user's question. {lang_instruction} "
            "If the answer is not in the context, say 'I don't know based on the provided information.'"
        )
        prompt = f"{system_prompt}\n\nRecent chat:\n{history_str}\n\nContext:\n{context_str}\n\nUser: {question}\nAnswer:"
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
