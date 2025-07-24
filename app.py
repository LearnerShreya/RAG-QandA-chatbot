import streamlit as st
from src.retriever import retrieve_top_k, load_faiss_retriever
from src.generator import get_gemini_llm, generate_answer
from src.chat_memory import get_memory, reset_memory
from src.pdf_reader import extract_text_from_pdf, extract_text_from_txt
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit.components.v1 as components
import tempfile
import time
import os
import io
import json

# Add fade-in animation CSS (at the top, global)
st.markdown("""
<style>
.fade-in {
    animation: fadeIn 0.7s;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Smart Loan Assistant", page_icon="💬", layout="wide")

# ------------------------ SESSION SETUP ------------------------ #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = get_memory()
if "bot_typing" not in st.session_state:
    st.session_state.bot_typing = False
if "custom_vectorstore" not in st.session_state:
    st.session_state.custom_vectorstore = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "context_history" not in st.session_state:
    st.session_state.context_history = []
if "language" not in st.session_state:
    st.session_state.language = "English"

# ------------------------ SIDEBAR ------------------------ #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/681/681494.png", width=100)
st.sidebar.title("💼 Loan Chatbot")
st.sidebar.write("Ask questions related to loan eligibility, approvals, interest, etc.")
# Dark mode toggle
mode = st.sidebar.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if mode else "light"

if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    reset_memory()
    st.session_state.memory = get_memory()
    st.session_state.custom_vectorstore = None
    st.rerun()

if st.sidebar.button("🧼 Clear Memory"):
    reset_memory()
    st.session_state.memory = get_memory()
    st.sidebar.success("Conversation memory cleared!")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Upload new document")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    with st.spinner("Processing uploaded file..."):
        # Save to temp file
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        # Extract text
        if suffix == ".pdf":
            new_text = extract_text_from_pdf(tmp_path)
        else:
            new_text = extract_text_from_txt(tmp_path)
        # Embed and create a new vectorstore for this session
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if st.session_state.custom_vectorstore is None:
            base_retriever = load_faiss_retriever()
            base_docs = [doc.page_content for doc in base_retriever.get_relevant_documents("")]
            st.session_state.custom_vectorstore = FAISS.from_texts(base_docs, embedding=embedding)
        st.session_state.custom_vectorstore.add_texts([new_text])
        # Track uploaded doc name
        st.session_state.uploaded_docs.append(uploaded_file.name)
        st.success("Document uploaded and added to knowledge base!")
        os.unlink(tmp_path)

# Sidebar summary panel
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Session Summary")
if st.session_state.uploaded_docs:
    st.sidebar.markdown("**Uploaded Docs:**")
    for doc in st.session_state.uploaded_docs:
        icon = "📄" if doc.lower().endswith(".pdf") else "📝"
        st.sidebar.markdown(f"- {icon} {doc}")
else:
    st.sidebar.markdown("_No docs uploaded this session._")
st.sidebar.markdown(f"**Questions asked:** {len(st.session_state.chat_history)}")
up_count = sum(1 for v in st.session_state.feedback.values() if v == "up")
down_count = sum(1 for v in st.session_state.feedback.values() if v == "down")
st.sidebar.markdown(f"**Feedback:** 👍 {up_count} &nbsp;&nbsp; 👎 {down_count}")

languages = [
    "English", "Hindi", "Bengali", "Telugu", "Marathi", "Tamil", "Urdu", "Gujarati", "Kannada", "Odia", "Malayalam", "Punjabi", "French", "Spanish", "German", "Chinese", "Japanese", "Arabic",
    "Assamese", "Maithili", "Santali", "Kashmiri", "Nepali", "Konkani", "Sindhi", "Dogri", "Manipuri", "Bodo", "Sanskrit"
]
st.session_state.language = st.sidebar.selectbox("🌐 Language", languages, index=languages.index(st.session_state.language) if st.session_state.language in languages else 0)

# ------------------------ MAIN HEADER ------------------------ #
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        html, body, .stApp, .stSidebar, .stSidebarContent, .stHeader, .st-emotion-cache-1avcm0n {
            background-color: #181b20 !important; color: #f5f6fa !important; font-size: 1.08em !important;
        }
        .stHeader, .st-emotion-cache-1avcm0n { background: #181b20 !important; color: #fff !important; border-bottom: none !important; }
        .chat-box { background: #23262b !important; color: #fff !important; min-height: 80px !important; overflow-y: auto; border: 1px solid #23262b !important; font-size: 1.09em !important; border-radius: 14px !important; box-shadow: 0 4px 24px #0004; }
        .user-msg { background: linear-gradient(90deg, #2563eb 80%, #1e90ff 100%); color: #fff !important; box-shadow: 0 2px 8px #2563eb44; font-size: 1.12em !important; border-radius: 12px; }
        .bot-msg { background: #23262b !important; color: #fff !important; box-shadow: 0 2px 8px #1e90ff22; font-size: 1.12em !important; border-radius: 12px; }
        .user-avatar { background: linear-gradient(135deg, #2563eb 60%, #1e90ff 100%) !important; color: #fff !important; }
        .bot-avatar { background: #23262b !important; color: #1e90ff !important; }
        .welcome-hero-img, .main-title, .main-subtitle, .welcome-hero-text { color: #fff !important; background: transparent !important; box-shadow: none !important; }
        .main-title .emoji, .main-subtitle .emoji, .welcome-hero-img .emoji { color: #fff !important; }
        .stSidebar img { background: #fff !important; border-radius: 12px !important; padding: 6px !important; box-shadow: 0 2px 8px #0003; }
        .stButton>button, .stDownloadButton>button, .stFileUploader .st-bw { background: #23262b !important; color: #fff !important; border-radius: 8px !important; border: 1.5px solid #2563eb !important; box-shadow: 0 2px 8px #1e90ff22; font-size: 1.08em !important; transition: background 0.2s, color 0.2s, border 0.2s; }
        .stButton>button:hover, .stDownloadButton>button:hover, .stFileUploader .st-bw:hover { background: #2563eb !important; color: #fff !important; border: 1.5px solid #1e90ff !important; }
        .stTextInput>div>div>input, .stTextInput>div>div>div>input { background: #23262b !important; color: #fff !important; border: 1.5px solid #2563eb !important; border-radius: 8px !important; font-size: 1.08em !important; box-shadow: 0 2px 8px #1e90ff22; }
        .stTextInput>div>div>input:focus, .stTextInput>div>div>div>input:focus { outline: 2px solid #1e90ff !important; }
        .stTextInput input::placeholder { color: #b0b3b8 !important; opacity: 1 !important; }
        .stExpander, .stExpanderHeader { background: #23262b !important; color: #fff !important; border: 1px solid #2563eb !important; border-radius: 10px !important; box-shadow: 0 2px 8px #1e90ff22; }
        .stSelectbox>div>div>div>div, .stSelectbox>div>div>div>div>div, .stSelectbox label { background: #23262b !important; color: #fff !important; border: none !important; border-radius: 8px !important; }
        .stSelectbox svg { color: #fff !important; fill: #fff !important; }
        .st-cq, .st-cp, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz { background: #23262b !important; color: #fff !important; border-radius: 10px !important; }
        /* File uploader container and text */
        .stFileUploader, .stFileUploader label, .stFileUploader .css-1mna7ka, .stFileUploader .css-1u2g0t6, .stFileUploader .css-1v0mbdj, .stFileUploader .css-1y4p8pa, .stFileUploader .css-1b1m3c7 {
            background: #23262b !important; color: #fff !important; border: 1.5px solid #23262b !important; border-radius: 12px !important; box-shadow: 0 2px 8px #1e90ff22;
        }
        .stFileUploader label, .stFileUploader div, .stFileUploader p, .stFileUploader span, .stFileUploader .stFileUploaderLabel, .stFileUploader div[role="alert"] {
            color: #fff !important;
        }
        .stFileUploader .css-1mna7ka, .stFileUploader .css-1u2g0t6, .stFileUploader .css-1v0mbdj, .stFileUploader .css-1y4p8pa, .stFileUploader .css-1b1m3c7 {
            background: #23262b !important; color: #fff !important; border: none !important; border-radius: 12px !important; }
        .stFileUploader .css-1v0mbdj, .stFileUploader .css-1u2g0t6, .stFileUploader .css-1mna7ka, .stFileUploader .css-1y4p8pa {
            color: #fff !important;
        }
        .stFileUploader .css-1b1m3c7 { color: #fff !important; background: #23262b !important; }
        .stFileUploader .css-1v0mbdj:hover, .stFileUploader .css-1u2g0t6:hover, .stFileUploader .css-1mna7ka:hover, .stFileUploader .css-1y4p8pa:hover, .stFileUploader .css-1b1m3c7:hover {
            background: #2563eb !important; color: #fff !important; border: 1.5px solid #1e90ff !important;
        }
        .stFileUploader .st-bw { background: #23262b !important; color: #fff !important; border: 1.5px solid #2563eb !important; border-radius: 8px !important; }
        .stFileUploader .st-bw:hover { background: #2563eb !important; color: #fff !important; border: 1.5px solid #1e90ff !important; }
        .stToggle label, .stToggle span, .stToggle { color: #fff !important; }
        ::-webkit-scrollbar { width: 8px; background: #181b20; }
        ::-webkit-scrollbar-thumb { background: #23262b; border-radius: 8px; }
        ::selection { background: #2563eb; color: #fff; }
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div { color: #fff !important; }
        .stExpanderContent { color: #fff !important; }
        .stCaption, .stCaption span, .stCaption p { color: #b0b3b8 !important; }
        .stMarkdown .secondary, .stMarkdown .feedback, .stMarkdown .placeholder { color: #b0b3b8 !important; }
        hr { border: 0; border-top: 1px solid #23262b; margin: 1em 0; }
        .stFileUploader .stFileUploaderLabel {
            color: #b0b3b8 !important;
            font-weight: 500 !important;
        }
        .stFileUploader small,
        .stFileUploader .stFileUploaderDetails,
        .stFileUploader .stMarkdown,
        .stFileUploader .stFileUploaderLabel + div,
        .stFileUploader .stFileUploaderLabel + span {
            color: #b0b3b8 !important;
            font-size: 0.98em !important;
        }
        /* Sidebar collapse/expand icon */
        [data-testid="collapsedControl"] svg {
            color: #fff !important;
            fill: #fff !important;
            opacity: 1 !important;
        }
        /* Sidebar toggle label and emoji */
        .stSidebar label, .stSidebar span, .stSidebar [data-testid="stSidebarUserContent"] label, .stSidebar [data-testid="stSidebarUserContent"] span {
            color: #fff !important;
        }
        /* File uploader dropzone and text */
        .stFileUploader, .stFileUploader label, .stFileUploader .stFileUploaderDropzone, .stFileUploader .stFileUploaderLabel, .stFileUploader .stFileUploaderDetails, .stFileUploader .stMarkdown, .stFileUploader small, .stFileUploader div[role="alert"] {
            background: #23262b !important;
            color: #fff !important;
            border: 1.5px solid #23262b !important;
            border-radius: 12px !important;
        }
        .stFileUploader .stFileUploaderLabel, .stFileUploader .stFileUploaderDetails, .stFileUploader .stMarkdown, .stFileUploader small, .stFileUploader div[role="alert"] {
            color: #b0b3b8 !important;
        }

        /* File uploader Browse files button */
        .stFileUploader .st-bw, .stFileUploader button {
            background: #23262b !important;
            color: #fff !important;
            border: 1.5px solid #2563eb !important;
            border-radius: 8px !important;
        }
        .stFileUploader .st-bw:hover, .stFileUploader button:hover {
            background: #2563eb !important;
            color: #fff !important;
            border: 1.5px solid #1e90ff !important;
        }

        /* Chat submit button (arrow) */
        .stButton>button, .stDownloadButton>button {
            background: #23262b !important;
            color: #fff !important;
            border-radius: 8px !important;
            border: 1.5px solid #2563eb !important;
            box-shadow: 0 2px 8px #1e90ff22;
            font-size: 1.08em !important;
            transition: background 0.2s, color 0.2s, border 0.2s;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background: #2563eb !important;
            color: #fff !important;
            border: 1.5px solid #1e90ff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        /* File uploader dropzone and button - improved dark mode */
        .stFileUploaderDropzone {
            background: #23262b !important; /* dark gray, not black */
            color: #f5f6fa !important;      /* off-white for high contrast */
            padding: 1.1rem 1.2rem !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px #0003 !important;
            min-height: 70px !important;
            display: flex !important;
            align-items: center !important;
            gap: 1em !important;
        }
        .stFileUploaderDropzone svg {
            color: #b0b3b8 !important;
            fill: #b0b3b8 !important;
            width: 28px !important;
            height: 28px !important;
            margin-right: 0.7em !important;
        }
        .stFileUploaderDropzone div,
        .stFileUploaderDropzone label,
        .stFileUploaderDropzone span {
            color: #f5f6fa !important;
            font-weight: 500 !important;
            font-size: 1.01rem !important;
            opacity: 1 !important;
        }
        .stFileUploaderDropzone small,
        .stFileUploaderDropzone .stFileUploaderDetails,
        .stFileUploaderDropzone .stMarkdown {
            background: #fff !important;
            color: #222 !important;
            padding: 2px 8px !important;
            border-radius: 6px !important;
            display: inline-block !important;
            font-size: 0.97rem !important;
            font-weight: 500 !important;
            margin-top: 0.3em !important;
        }
        .stFileUploader .st-bw, .stFileUploader button {
            background: #23262b !important;
            color: #f5f6fa !important;
            border: 1.2px solid #2563eb !important;
            border-radius: 7px !important;
            font-size: 0.98em !important;
            font-weight: 500 !important;
            padding: 0.38em 1em !important; /* smaller button */
            box-shadow: 0 1px 4px #1e90ff22;
        }
        .stFileUploader .st-bw:hover, .stFileUploader button:hover {
            background: #2563eb !important;
            color: #fff !important;
            border: 1.2px solid #1e90ff !important;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        html, body { background-color: #f5f8fc; font-family: 'Segoe UI', sans-serif; }
        .chat-box {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            min-height: 80px;
            overflow-y: auto;
        }
        .user-msg, .bot-msg {
            padding: 0.8rem 1.2rem;
            margin: 0.6rem 0;
            border-radius: 10px;
            font-size: 1.02rem;
            line-height: 1.5;
            max-width: 80%;
        }
        .user-msg {
            background: linear-gradient(90deg, #0e76a8 80%, #3b8beb 100%);
            color: white;
            margin-left: auto;
            text-align: right;
            box-shadow: 0 2px 8px rgba(14,118,168,0.10);
            display: flex;
            align-items: center;
            gap: 0.5em;
        }
        .bot-msg {
            background-color: #f1f1f1;
            color: #333;
            margin-right: auto;
            box-shadow: 0 2px 8px rgba(14,118,168,0.06);
            display: flex;
            align-items: center;
            gap: 0.5em;
        }
        .user-avatar, .bot-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4em;
            margin-right: 0.5em;
        }
        .user-avatar {
            background: linear-gradient(135deg, #0e76a8 60%, #3b8beb 100%);
            color: #fff;
        }
        .bot-avatar {
            background: #eaf6fb;
            color: #0e76a8;
        }
        .typing-msg {
            color: #0e76a8;
            font-style: italic;
            animation: blink 1.2s infinite;
        }
        @keyframes blink {
            0% { opacity: 0.2; }
            50% { opacity: 1; }
            100% { opacity: 0.2; }
        }
        .welcome-hero {
            display: flex;
            align-items: center;
            gap: 1.2em;
            margin-bottom: 1.2em;
        }
        .welcome-hero-img {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #eaf6fb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.2em;
            box-shadow: 0 2px 8px rgba(14,118,168,0.08);
        }
        .welcome-hero-text {
            font-size: 1.15em;
            color: #0e76a8;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.3em;
            font-weight: 800;
            color: #1e90ff;
            text-align: left;
            margin-bottom: 0.1em;
            letter-spacing: 0.01em;
        }
        .main-subtitle {
            font-size: 1.1em;
            color: #e0e0e0;
            text-align: left;
            margin-bottom: 1.2em;
            font-weight: 400;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.3em;
            font-weight: 800;
            color: #0e276a;
            text-align: left;
            margin-bottom: 0.1em;
            letter-spacing: 0.01em;
        }
        .main-subtitle {
            font-size: 1.1em;
            color: #333333;
            text-align: left;
            margin-bottom: 1.2em;
            font-weight: 400;
        }
        </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="main-title">💬 Smart Loan Q&amp;A Chatbot</div>
    <div class="main-subtitle">AI-powered assistant for loan eligibility, approval, rejections, documents, and financial tips</div>
""", unsafe_allow_html=True)

# Welcome message with illustration
st.markdown("""
<div class="welcome-hero">
  <div class="welcome-hero-img">🏦</div>
  <div class="welcome-hero-text">
    Welcome!<br>
    <span style='font-weight:400;color:#262730;'>Ask any question about loan approval, eligibility, or process.<br>Powered by RAG, FAISS, and Gemini LLM.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Example prompts
with st.expander("💡 Example questions", expanded=False):
    st.markdown("""
    - What are the current home loan interest rates?
    - Why was my loan rejected even with good income?
    - Is credit history important for loan approval?
    - What increases the chances of getting a home loan?
    """)

# ------------------------ CHAT WINDOW ------------------------ #
with st.container():
    if not st.session_state.chat_history and not st.session_state.bot_typing:
        st.markdown("""
        <div style='text-align:center; color:#888; margin-top:2em;'>
            <div style='font-size:3em;'>💬</div>
            <div style='font-size:1.2em; margin-top:0.5em;'>Start the conversation!<br>Type your question below or use the mic.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for idx, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"<div class='user-msg fade-in'><span class='user-avatar'>🧑‍💼</span>{q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-msg fade-in'><span class='bot-avatar'>🤖</span>{a}</div>", unsafe_allow_html=True)
            # Feedback buttons for each bot answer
            fb_key = f"feedback_{idx}"
            if idx not in st.session_state.feedback:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍", key=fb_key+"_up"):
                        st.session_state.feedback[idx] = "up"
                with col2:
                    if st.button("👎", key=fb_key+"_down"):
                        st.session_state.feedback[idx] = "down"
            else:
                fb_val = st.session_state.feedback[idx]
                st.markdown(f"<span style='color:#0e76a8;font-size:1.1em;'>Feedback: {'👍' if fb_val=='up' else '👎'}</span>", unsafe_allow_html=True)
            # Context viewer
            if idx < len(st.session_state.context_history):
                with st.expander("🔎 Show retrieved context", expanded=False):
                    context_chunks = st.session_state.context_history[idx]
                    if context_chunks:
                        for i, chunk in enumerate(context_chunks):
                            st.markdown(f"**Chunk {i+1}:**\n{chunk}")
                    else:
                        st.markdown("_No context retrieved for this answer._")
        if st.session_state.bot_typing:
            st.markdown("<div class='bot-msg typing-msg fade-in'><span class='bot-avatar'>🤖</span>Typing...</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Download chat as TXT
    if st.session_state.chat_history:
        chat_lines = []
        for q, a in st.session_state.chat_history:
            chat_lines.append(f"You: {q}\nBot: {a}\n")
        chat_txt = "\n".join(chat_lines)
        st.download_button(
            label="📥 Download Chat as TXT",
            data=chat_txt,  # Pass string directly
            file_name="loan_chatbot_conversation.txt",
            mime="text/plain"
        )

# Voice input button (above the form, single column)
mic_result = st.button("🎤 Speak", key="mic_button")
if mic_result:
    components.html('''
    <script>
    const streamlitInput = window.parent.document.querySelector('input[data-testid="stTextInput"]');
    if (window.hasRunVoiceInput !== true) {
        window.hasRunVoiceInput = true;
        var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'en-US';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            streamlitInput.value = transcript;
            streamlitInput.dispatchEvent(new Event('input', { bubbles: true }));
        };
        recognition.start();
    }
    </script>
    ''', height=0)
    st.info("Listening... Please speak your question.")

# ------------------------ INPUT FORM ------------------------ #
with st.form(key="chat_form", clear_on_submit=True):
    input_col, button_col = st.columns([8, 1])
    user_input = input_col.text_input("", placeholder="Ask your loan-related question here...", label_visibility="collapsed", key="text_input")
    submitted = button_col.form_submit_button("➤")

# ------------------------ HANDLE SUBMIT ------------------------ #
def custom_retrieve_top_k(query, k=5):
    if st.session_state.custom_vectorstore is not None:
        docs = st.session_state.custom_vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    else:
        return retrieve_top_k(query, k=k)

if submitted and user_input:
    st.session_state.bot_typing = True
    st.session_state.chat_history.append((user_input, "..."))  # Temporary
    st.session_state.context_history.append([])  # Placeholder for context
    st.rerun()

# # ------------------------ GENERATE RESPONSE ------------------------ #
# if st.session_state.bot_typing:
#     time.sleep(1.0)
#     with st.spinner("Generating answer..."):
#         context = custom_retrieve_top_k(user_input, k=5)
#         llm = get_gemini_llm()
#         chat_hist = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 1 else []
#         raw_answer = generate_answer(llm, user_input, context, chat_history=chat_hist, language=st.session_state.language)
#         final_answer = raw_answer.strip()
#         if not final_answer or "i don't know" in final_answer.lower():
#             final_answer = "I'm not sure based on that input. Could you try rephrasing your question or give more details?"
#     st.session_state.chat_history[-1] = (user_input, final_answer)
#     st.session_state.context_history[-1] = context
#     st.session_state.memory.save_context({"input": user_input}, {"output": final_answer})
#     st.session_state.bot_typing = False
#     st.rerun()


if st.session_state.bot_typing:
    time.sleep(1.0)
    with st.spinner("Generating answer..."):
        context = custom_retrieve_top_k(user_input, k=5)
        llm = get_gemini_llm()
        chat_hist = st.session_state.chat_history[-4:] if len(st.session_state.chat_history) > 1 else []
        
        answer = generate_answer(llm=llm, query=user_input, context_docs=context, history=chat_hist)
        final_answer = answer.strip() if answer else "I'm not sure based on that input. Could you try rephrasing your question or give more details?"

    st.session_state.chat_history[-1] = (user_input, final_answer)
    st.session_state.context_history[-1] = context
    st.session_state.memory.save_context({"input": user_input}, {"output": final_answer})
    st.session_state.bot_typing = False
    st.rerun()
