
import streamlit as st
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from vector_store import VectorStore
from retriever import Retriever
from llm import HRPolicyLLM
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 HR Policy Assistant")
st.write(
    "Ask questions about company policies, "
    "vacation, remote work, privacy, and benefits."
)

@st.cache_resource
def load_pipeline():
    vector_store = VectorStore()
    retriever = Retriever(vector_store, top_k=3)
    llm = HRPolicyLLM(model_name="llama3.1:latest")
    return RAGPipeline(retriever, llm)


rag = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"- {source}")


question = st.chat_input("Ask an HR policy question...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating an answer..."):
            try:
                result = rag.answer(question)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"- {source}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")
