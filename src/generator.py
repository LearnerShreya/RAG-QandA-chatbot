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
import requests

load_dotenv()

def get_gemini_llm():
    """
    Instantiates the Gemini LLM using the API key from the environment.
    Uses GOOGLE_API_KEY for the chatbot functionality.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please set GOOGLE_API_KEY in your .env file.")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("test")
        return model
    except Exception as e:
        raise ValueError(f"Google API connection failed: {str(e)}")

def generate_answer(llm, question: str, context: list, chat_history: list = None, language: str = "English") -> str:
    """
    Generates an answer using the LLM. If context is empty or weak, uses Gemini's general knowledge; otherwise, uses RAG context. Includes recent chat history for multi-turn memory. Answers in the selected language.
    """
    chat_history = chat_history or []
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

def execute_code_judge0(source_code, language_id, stdin=None):
    """
    Executes code using the Judge0 API. Returns the output or error message.
    Requires JUDGE0_API_KEY in .env.
    """
    api_key = os.getenv("JUDGE0_API_KEY")
    if not api_key:
        raise ValueError("JUDGE0_API_KEY environment variable not set.")
    url = "https://judge0-ce.p.rapidapi.com/submissions?base64_encoded=false&wait=true"
    headers = {
        "Content-Type": "application/json",
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com"
    }
    payload = {
        "source_code": source_code,
        "language_id": language_id,
    }
    if stdin:
        payload["stdin"] = stdin
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 201:
        raise Exception(f"Judge0 API error: {response.status_code} {response.text}")
    result = response.json()
    if result.get("stderr"):
        return result["stderr"]
    return result.get("stdout", "")

if __name__ == "__main__":
    pass
