import streamlit as st
from src.retriever import retrieve_top_k
from src.generator import get_gemini_llm, generate_answer
from src.chat_memory import get_memory, reset_memory
from dotenv import load_dotenv
import os

# Load environment variables (for Gemini API key)
load_dotenv()

st.set_page_config(page_title="Loan Q&A Bot", page_icon="🏦")
st.title("🏦 Loan Approval Chatbot (RAG + Gemini)")
st.markdown("""
Ask any question about loan approval, eligibility, or process.\n
*Powered by RAG, FAISS, and Gemini LLM.*
""")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = get_memory()

# Option to reset chat
if st.button("🧹 New Conversation"):
    st.session_state.chat_history = []
    reset_memory()
    st.session_state.memory = get_memory()
    st.rerun()

# User input
query = st.text_input("Ask a question about loan approval:", key="user_input")

if query:
    # Retrieve relevant context
    with st.spinner("Retrieving relevant information..."):
        context = retrieve_top_k(query, k=5)
    # Get LLM
    llm = get_gemini_llm()
    # Generate answer
    with st.spinner("Generating answer..."):
        answer = generate_answer(llm, query, context)
    # Save to chat history and memory
    st.session_state.chat_history.append((query, answer))
    st.session_state.memory.save_context({"input": query}, {"output": answer})

# Display chat history (most recent last)
st.markdown("---")
st.subheader("Conversation")
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
