"""
generator.py
------------
Gemini LLM integration for answer generation in the RAG pipeline.
- Uses official Google GenerativeAI SDK
- Reads API key from environment variable
- Enhanced RAG + LLM integration with structured formatting
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
    Enhanced RAG + LLM answer generation with structured formatting.
    Combines retrieved context with LLM's knowledge for concise, 
    well-structured responses with clear sections and bullet points.
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
    
    # Enhanced RAG + LLM prompt engineering with concise formatting
    if context_is_weak:
        # Fallback to LLM's general knowledge when RAG context is insufficient
        prompt = (
            f"You are an expert loan advisor. {lang_instruction}\n\n"
            f"IMPORTANT: Keep responses CONCISE and FOCUSED. Maximum 150-200 words.\n\n"
            f"FORMATTING REQUIREMENTS:\n"
            f"1. Use clear headings with ##\n"
            f"2. Use bullet points (•) for lists\n"
            f"3. Use bold text (**text**) for emphasis\n"
            f"4. Keep sections short and to the point\n"
            f"5. Avoid unnecessary explanations\n\n"
            f"Recent conversation:\n{history_str}\n\n"
            f"User Question: {question}\n\n"
            f"Provide a CONCISE, well-structured response:"
        )
    else:
        # Enhanced RAG + LLM integration with concise formatting
        prompt = (
            f"You are a knowledgeable loan assistant. Use the provided context information "
            f"AND your own expertise to give comprehensive but CONCISE answers. {lang_instruction}\n\n"
            f"IMPORTANT: Keep responses CONCISE and FOCUSED. Maximum 150-200 words.\n\n"
            f"FORMATTING REQUIREMENTS:\n"
            f"1. Use clear headings with ##\n"
            f"2. Use bullet points (•) for lists\n"
            f"3. Use bold text (**text**) for emphasis\n"
            f"4. Keep sections short and to the point\n"
            f"5. Avoid unnecessary explanations\n"
            f"6. Focus on the most important information\n\n"
            f"Retrieved Context:\n{context_str}\n\n"
            f"Recent conversation:\n{history_str}\n\n"
            f"User Question: {question}\n\n"
            f"Instructions:\n"
            f"1. Use the context information as your primary source\n"
            f"2. Supplement with your own knowledge if needed\n"
            f"3. Be specific and concise\n"
            f"4. Focus on the most relevant information\n"
            f"5. Keep the response under 200 words\n"
            f"6. Use clear, structured formatting\n\n"
            f"Provide a CONCISE, well-structured answer:"
        )
    
    try:
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Fallback response if LLM fails
        return f"I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

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
