
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
import numpy as np

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Assistant")
st.caption("Ask questions about your PDF using local AI — no paid API required.")

@st.cache_resource
def load_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.2", temperature=0)
    return embeddings, llm

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages

def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = []
    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            chunks.append({
                "page": page["page"],
                "text": chunk_text,
            })

    return chunks

def retrieve_chunks(question, chunks, embeddings, top_k=4):
    question_vector = np.array(
        embeddings.embed_query(question),
        dtype=float,
    )
    question_norm = np.linalg.norm(question_vector)

    scored_chunks = []

    for chunk in chunks:
        chunk_vector = np.array(
            chunk["embedding"],
            dtype=float,
        )
        denominator = question_norm * np.linalg.norm(chunk_vector)

        if denominator == 0:
            score = 0
        else:
            score = float(
                np.dot(question_vector, chunk_vector) / denominator
            )

        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
)

if uploaded_file:
    try:
        pages = extract_pdf(uploaded_file)

        if not pages:
            st.error(
                "No selectable text was found. "
                "Scanned PDFs may require OCR."
            )
            st.stop()

        st.success(f"PDF loaded: {len(pages)} page(s)")

        with st.spinner("Preparing your document for local AI..."):
            embeddings, llm = load_models()

            chunks = split_pages(pages)
            chunk_vectors = embeddings.embed_documents(
                [chunk["text"] for chunk in chunks]
            )

            for chunk, vector in zip(chunks, chunk_vectors):
                chunk["embedding"] = vector

        st.success(f"Document ready: {len(chunks)} text sections indexed.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about your PDF...")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the PDF and generating an answer..."):
                    relevant_chunks = retrieve_chunks(
                        question,
                        chunks,
                        embeddings,
                    )
                    with st.expander("Debug: Retrieved PDF text"):
                        for chunk in relevant_chunks:
                            st.write(f"Page {chunk['page']}")
                            st.write(chunk["text"])
                            st.divider()

                    context = "\n\n".join(
                        f"[Page {chunk['page']}]\n{chunk['text']}"
                        for chunk in relevant_chunks
                    )

                    prompt = f"""
You are an assistant answering questions about an uploaded PDF.

Use only the document excerpts below to answer.
If the answer is not present, say that you could not find it in the PDF.
Do not invent facts. Cite the relevant page number(s).

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}
"""

                    response = llm.invoke(prompt)
                    answer = response.content

                st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

    except Exception as error:
        st.error("Something went wrong while processing the PDF.")
        st.exception(error)

else:
    st.info("Upload a PDF to get started.")
