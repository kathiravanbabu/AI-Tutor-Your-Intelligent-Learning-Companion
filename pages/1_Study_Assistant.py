import streamlit as st
import os
import docx
from uuid import uuid4
from typing_extensions import List, TypedDict, Annotated

from datetime import datetime
import os
from dotenv import load_dotenv
from utils.llm import get_llm, get_embeddings
from utils.files import get_pdf_text, get_docx_text, get_text_chunks
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Load environment variables

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables.")
    st.stop()

# Streamlit session state

if "success" not in st.session_state:
    st.session_state.success = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Initialize LLM & Embeddings

llm = get_llm()
embeddings = get_embeddings()

# Prompt Template

template = """
Use the following pieces of context to answer the question below
in a clear and structured manner.

If you do not know the answer, say that you do not know.
Do not make up information.

{context}

Question: {question}

Helpful Answer:
"""

prompt = PromptTemplate.from_template(template)



def get_retriever(text_chunks):
    index = faiss.IndexFlatL2(
        len(embeddings.embed_query("dimension check"))
    )

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    ids = [str(uuid4()) for _ in text_chunks]
    vector_store.add_texts(text_chunks, ids=ids)

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2}
    )

# LangGraph state definitions

class Search(BaseModel):
    query: str = Field(description="Search query")


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# Graph nodes

def analyze_query(state: State):
    parser = PydanticOutputParser(pydantic_object=Search)
    prompt = PromptTemplate(
        template="Generate a search query to retrieve relevant documents for the following user question.\n{format_instructions}\nQuestion: {question}",
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    query = chain.invoke({"question": state["question"]})
    return {"query": query}


def retrieve(state: State):
    retrieved_docs = st.session_state.retriever.invoke(
        state["query"].query
    )
    return {"context": retrieved_docs}


def generate(state: State):
    context_text = "\n\n".join(
        doc.page_content for doc in state["context"]
    )
    messages = prompt.invoke(
        {
            "question": state["question"],
            "context": context_text
        }
    )
    response = llm.invoke(messages)
    return {"answer": response.content}

# --------------------------------------------------
# UI Logic

def summary():
    st.title("Study Assistant")
    st.write("Upload documents to summarize content or ask specific questions.")

    docs = st.file_uploader(
        "Upload PDF or DOCX Files",
        accept_multiple_files=True,
        type=["pdf", "docx"],
        help="Upload one or more PDF or DOCX documents to use for study and question answering."
    )

    if st.button("Process Documents") and docs:
        with st.spinner("Processing files..."):
            raw_text = ""
            pdf_docs = [f for f in docs if f.name.endswith('.pdf')]
            docx_docs = [f for f in docs if f.name.endswith('.docx')]
            
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if docx_docs:
                raw_text += get_docx_text(docx_docs)
                
            chunks = get_text_chunks(raw_text)

            if not chunks:
                st.error("No text found in files. Are they scanned or empty?")
                st.stop()
            
            st.session_state.retriever = get_retriever(chunks)
            st.session_state.success = True
            st.success(
                f"Successfully processed {len(docs)} file(s) into {len(chunks)} chunks."
            )

    if st.session_state.success:
        user_query = st.text_area(
            "What would you like to know?",
            placeholder="Type your question here...",
            help="Enter your question or specific topic you want to learn about from the uploaded documents."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Ask Question", use_container_width=True):
                if user_query:
                    with st.spinner("Generating answer..."):
                        create_summary(user_query)
                else:
                    st.warning("Please enter a question.")
        
        with col2:
            if st.button("Summarize", use_container_width=True):
                with st.spinner("Generating summary..."):
                    # For summarization, we treat it as a query to summarize deeply
                    summary_prompt = f"Summarize the following topic or document based on context: {user_query}" if user_query else "Provide a comprehensive summary of the document."
                    create_summary(summary_prompt)

# Graph execution

def create_summary(topic):
    graph_builder = StateGraph(State)

    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)

    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.set_entry_point("analyze_query")

    graph = graph_builder.compile()
    result = graph.invoke({"question": topic})

    st.subheader("Assistant Response")
    st.write(result["answer"])

# Run app

summary()
