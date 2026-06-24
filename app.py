import streamlit as st
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

st.set_page_config(page_title="PDF Chatbot")
st.title("📄 PDF-Based AI Chatbot")

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )
    return embed_model, llm

embed_model, llm = load_models()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    chunks = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        for j in range(0, len(text), 500):
            chunks.append({
                "page": i + 1,
                "text": text[j:j+500]
            })

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts)

    query = st.text_input("Ask a question about the PDF")

    if query:
        query_vec = embed_model.encode([query])[0]
        scores = np.dot(embeddings, query_vec)

        top_idx = scores.argsort()[-3:][::-1]

        context = ""
        pages = set()
        for idx in top_idx:
            context += f"(Page {chunks[idx]['page']}) {chunks[idx]['text']}\n\n"
            pages.add(chunks[idx]["page"])

        prompt = f"""
        Answer using ONLY the context below.
        Mention page numbers.

        Context:
        {context}

        Question:
        {query}
        """

        answer = llm(prompt)[0]["generated_text"]

        st.subheader(" Answer")
        st.write(answer)

        st.subheader(" Sources")
        for p in sorted(pages):
            st.write(f"Page {p}")
