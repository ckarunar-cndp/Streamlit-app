import os
import io
import pickle
import uuid
import streamlit as st
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np

# LangChain & vectorstore
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Transformers pipelines for DEV mode
from transformers import pipeline

# ---------------- Config ----------------
DEV_MODE = True  # Set False to switch to OpenAI
DOCUMENTS_DIR = Path("documents")
VECTORSTORE_PATH = Path("vectorstore_mpnet.faiss")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 Company Knowledge Assistant (RAG)")

# ---------------- Prompt Template ----------------
system_prompt = """
You are a helpful company assistant.
Answer politely and concisely using ONLY the context provided.
If the requested information is not in the context, reply exactly:
"I'm sorry, I don't have that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(input_variables=["question", "context"], template=system_prompt)

# ---------------- File Readers ----------------
def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"Could not read PDF {path.name}: {e}")
        return ""

def read_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.warning(f"Could not read DOCX {path.name}: {e}")
        return ""

def read_xlsx(path: Path) -> str:
    try:
        df = pd.read_excel(path)
        return df.to_string(index=False)
    except Exception as e:
        st.warning(f"Could not read XLSX {path.name}: {e}")
        return ""

def load_documents_from_folder(folder: Path) -> List[str]:
    texts = []
    for f in sorted(folder.iterdir()):
        if f.name.startswith("~$") or f.name.startswith("."):
            continue
        if f.suffix.lower() == ".pdf":
            texts.append(read_pdf(f))
        elif f.suffix.lower() == ".docx":
            texts.append(read_docx(f))
        elif f.suffix.lower() in [".xlsx", ".xls"]:
            texts.append(read_xlsx(f))
        elif f.suffix.lower() == ".txt":
            try:
                texts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                texts.append(f.read_text(encoding="latin-1", errors="ignore"))
    return [t for t in texts if t.strip()]

# ---------------- Vectorstore ----------------
@st.cache_resource(show_spinner="Building vectorstore...")
def build_vectorstore(dev_mode: bool = True):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL) if dev_mode else OpenAIEmbeddings(model="text-embedding-3-small")

    if VECTORSTORE_PATH.exists():
        try:
            with open(VECTORSTORE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    texts = load_documents_from_folder(DOCUMENTS_DIR)
    if not texts:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents(texts)

    vs = FAISS.from_documents(docs, embeddings)
    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vs, f)
    return vs

# ---------------- DEV Pipeline ----------------
@st.cache_resource(show_spinner="Loading local HF models...")
def load_dev_pipeline():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    except Exception:
        return pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)

# ---------------- Smart Retrieval ----------------
def get_semantic_docs(query, retriever, embeddings, top_k=TOP_K):
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return []

    # Re-rank results based on cosine similarity
    query_vec = np.array(embeddings.embed_query(query))
    doc_vecs = np.array([embeddings.embed_query(d.page_content[:512]) for d in docs])
    sims = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
    sorted_idx = np.argsort(sims)[::-1][:top_k]
    return [docs[i] for i in sorted_idx]

# ---------------- Load Resources ----------------
vectorstore = build_vectorstore(DEV_MODE)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K}) if vectorstore else None
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
hf_qa_pipe = load_dev_pipeline() if DEV_MODE else None

if not DEV_MODE and retriever:
    llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, max_tokens=512)
    qa_chain_prod = ConversationalRetrievalChain.from_llm(
        llm=llm_openai, retriever=retriever, combine_docs_chain_kwargs={"prompt": PROMPT}
    )
else:
    qa_chain_prod = None

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "👋 Hello! Ask me anything."}]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings & Uploads")
    st.markdown(f"**Mode:** {'DEV (local HF)' if DEV_MODE else 'PROD (OpenAI)'}")

    uploaded = st.file_uploader("Upload documents", accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            dest = DOCUMENTS_DIR / up.name
            if dest.exists():
                dest = DOCUMENTS_DIR / f"{Path(up.name).stem}-{uuid.uuid4().hex[:6]}{Path(up.name).suffix}"
            with open(dest, "wb") as f:
                f.write(up.getbuffer())
        if VECTORSTORE_PATH.exists():
            VECTORSTORE_PATH.unlink()
        vectorstore = build_vectorstore(DEV_MODE)
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K}) if vectorstore else None
        st.success("✅ Index rebuilt successfully.")

    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "👋 Hello! Ask me anything."}]
        st.rerun()

# ---------------- Chat Display ----------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- Main Input ----------------
def generate_answer(user_query):
    if retriever is None:
        return "No indexed documents found. Please upload some files."

    docs = get_semantic_docs(user_query, retriever, embeddings, top_k=TOP_K)
    context = "\n\n".join([d.page_content for d in docs])

    if DEV_MODE:
        prompt = system_prompt.format(question=user_query, context=context)
        out = hf_qa_pipe(prompt)
        if isinstance(out, list) and "generated_text" in out[0]:
            return out[0]["generated_text"]
        return str(out)
    else:
        try:
            result = qa_chain_prod({"question": user_query, "chat_history": []})
            return result.get("answer", "I'm sorry, I don't have that information.")
        except Exception as e:
            return f"Error: {e}"

# ---------------- Chat Input ----------------
if user_input := st.chat_input("Type your question..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.spinner("Generating answer..."):
        answer_text = generate_answer(user_input)
    st.session_state["messages"].append({"role": "assistant", "content": answer_text})
    st.rerun()

# ---------------- FAQ Section ----------------
st.markdown("### 💡 Frequently Asked Questions")
faq_list = [
    "What services does the company provide?",
    "Which industries has the company served?",
    "What are the company’s core strengths?",
    "Does the company offer cloud solutions?",
    "Where is the company located?"
]

cols = st.columns(5)
for i, q in enumerate(faq_list):
    if cols[i % 5].button(q):
        st.session_state["messages"].append({"role": "user", "content": q})
        with st.spinner("Retrieving answer..."):
            answer_text = generate_answer(q)
        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
        st.rerun()
