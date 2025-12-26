import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import streamlit as st

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Google Embeddings (YOUR REQUESTED MODEL)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Gemini for answering
import google.generativeai as genai


# ------------------------ SETUP ------------------------

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please add GOOGLE_API_KEY to your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

DB_DIR = "chroma_gemini_embeddings"


# ------------------------ WEBSITE LOADER (403 FIX) ------------------------

def load_website_content(url):
    """Loads website text while bypassing 403 Forbidden."""
    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()   # Raises 403/404 errors

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.extract()

    text = soup.get_text(separator="\n")
    cleaned_text = "\n".join(
        line.strip() for line in text.splitlines() if line.strip()
    )

    return [Document(page_content=cleaned_text, metadata={"source": url})]


# ------------------------ CHUNKING ------------------------

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
    )
    return splitter.split_documents(docs)


# ------------------------ EMBEDDINGS (YOUR MODEL) ------------------------

def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",   # <--- YOUR REQUEST
        google_api_key=GOOGLE_API_KEY
    )


# ------------------------ VECTORSTORE BUILD/LOAD ------------------------

def build_vectorstore(docs, url):
    """Create a Chroma DB using Gemini embeddings."""
    for d in docs:
        d.metadata["source"] = url

    # NOTE: from_documents uses embedding=
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=get_gemini_embeddings(),   # <--- CORRECT FOR from_documents
        persist_directory=DB_DIR,
    )
    vectordb.persist()
    return vectordb


def load_existing_vectorstore():
    """Load existing Chroma DB (if exists)."""
    if not os.path.exists(DB_DIR):
        return None

    # NOTE: direct Chroma() uses embedding_function=
    return Chroma(
        embedding_function=get_gemini_embeddings(),   # <--- CORRECT FOR Chroma()
        persist_directory=DB_DIR,
    )


# ------------------------ GEMINI ANSWER GENERATION ------------------------

def chat_with_context(question, context_text):
    """Generate answer from Gemini Flash using retrieved context."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not found, say:
"I could not find this information on the website."

CONTEXT:
{context_text}

QUESTION:
{question}
"""

    response = model.generate_content(prompt)
    return response.text


# ------------------------ STREAMLIT UI ------------------------

st.set_page_config(page_title="Chat with Website (Gemini)", layout="wide")

st.markdown(
    """
    <div style="background:#1e3c72;padding:20px;border-radius:12px;margin-bottom:20px;">
        <h2 style="color:white;">🌐 Chat with Website — Gemini Embeddings Enabled</h2>
        <p style="color:#ddd;">Using model: <b>models/gemini-embedding-001</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1, 2])


# ------------------------ LEFT PANEL (LOAD WEBSITE) ------------------------

with left:
    st.subheader("1️⃣ Load Website URL")

    url = st.text_input("Enter a website URL", placeholder="https://example.com")

    # Load DB on first run
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = load_existing_vectorstore()

    if st.button("📥 Fetch & Index Website"):
        if not url:
            st.warning("Please enter a URL first.")
        else:
            try:
                with st.spinner("Fetching and indexing website..."):
                    docs = load_website_content(url)
                    chunks = split_documents(docs)
                    vectordb = build_vectorstore(chunks, url)
                    st.session_state.vectordb = vectordb

                st.success("Website indexed successfully!")
                st.info(f"Total chunks stored: {len(chunks)}")

            except Exception as e:
                st.error(f"Error while loading: {e}")


# ------------------------ RIGHT PANEL (CHAT) ------------------------

with right:
    st.subheader("2️⃣ Chat with Website Content")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    question = st.chat_input("Ask something...")

    if question:
        if not st.session_state.vectordb:
            st.warning("Please index a website first.")
        else:
            # user message
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    vectordb = st.session_state.vectordb

                    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

                    # NEW API (fix)
                    docs = retriever.invoke(question)

                    # Extract text from documents
                    context = "\n\n".join([d.page_content for d in docs])

                    answer = chat_with_context(question, context)

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
