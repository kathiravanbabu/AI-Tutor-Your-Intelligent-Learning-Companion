import streamlit as st
import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_llm():
    """Initialize Groq LLM."""
    return init_chat_model(
        "llama-3.1-8b-instant",
        model_provider="groq"
    )

@st.cache_resource
def get_embeddings():
    """Initialize HuggingFace Embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
