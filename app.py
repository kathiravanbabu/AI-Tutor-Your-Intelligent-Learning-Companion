import streamlit as st

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Tutor – A RAG Based Document QA System",
    page_icon=None,
    layout="wide"
)

# -------------------- Header Section --------------------
st.title("AI Tutor – A RAG Based Document QA System")
st.markdown("### Your Intelligent Learning Companion")

# -------------------- Intro Section --------------------
st.markdown(
    """
    **AI Tutor** is an intelligent learning companion built using **RAG, FAISS, Hugging Face, and Groq**.
    It helps you understand content deeply by summarizing, testing, answering questions,
    and guiding you with structured learning roadmaps.
    """
)

st.divider()

# -------------------- Features Section --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Study Assistant")
    st.write("Upload documents to summarize content or ask specific questions.")

with col2:
    st.subheader("Quiz Generator")
    st.write("Automatically create quizzes and evaluate answers with AI-driven feedback.")

with col3:
    st.subheader("Learning Roadmaps")
    st.write("Get structured, field-based learning paths for careers in tech and AI.")

# -------------------- Navigation Hint --------------------
st.divider()
st.info("Use the sidebar to navigate between modules.")

# -------------------- Footer --------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
        Built by Sai • Powered by RAG, FAISS, Hugging Face & Groq
    </p>
    """,
    unsafe_allow_html=True
)
