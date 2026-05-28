import streamlit as st
import os
import json
import docx
from typing import List, Dict

from datetime import datetime
import os
from dotenv import load_dotenv
from utils.llm import get_llm, get_embeddings
from utils.files import get_pdf_text, get_docx_text, get_text_chunks
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in environment variables.")
    st.stop()

# Session State Initialization

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "questions_generated" not in st.session_state:
    st.session_state.questions_generated = False

if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

if "score" not in st.session_state:
    st.session_state.score = None

# Initialize Embeddings & LLM

embeddings = get_embeddings()
llm = get_llm()

def create_vector_store(chunks):
    return FAISS.from_texts(chunks, embeddings)

# Question Generation

class Question(BaseModel):
    question: str = Field(description="The question text")
    options: Dict[str, str] = Field(description="Dictionary of options with keys a, b, c, d")
    correct_answer: str = Field(description="The correct option key (a, b, c, or d)")

class Quiz(BaseModel):
    questions: List[Question] = Field(description="List of questions")

def generate_questions(topic: str, vector_store, num_questions: int = 10):
    docs = vector_store.similarity_search(topic, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    parser = JsonOutputParser(pydantic_object=Quiz)

    prompt = PromptTemplate(
        template="""
You are an expert educator.

Generate {num_questions} multiple-choice questions based on the context provided.
Ensure questions are clear, concise, most asked and written in standard English.
Avoid overly complex jargon where simple terms suffice.

Each question must have 4 options (a, b, c, d) and exactly one correct answer.

{format_instructions}

Context:
{context}

Topic:
{topic}
""",
        input_variables=["num_questions", "context", "topic"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    
    try:
        response = chain.invoke(
            {
                "num_questions": num_questions,
                "context": context,
                "topic": topic,
            }
        )
        return response
    except Exception as e:
        st.error(f"Failed to generate quiz: {str(e)}")
        return None

# Quiz Display & Evaluation

def display_quiz(questions_data):
    st.session_state.user_answers = {}

    for i, question in enumerate(questions_data["questions"], 1):
        st.write(f"**Question {i}:** {question['question']}")
        options = question["options"]

        st.session_state.user_answers[f"q{i}_correct"] = question["correct_answer"]

        user_answer = st.radio(
            f"Select your answer for Question {i}:",
            options=["a", "b", "c", "d"],
            key=f"q{i}",
            format_func=lambda x: f"{x}) {options[x]}",
        )

        st.session_state.user_answers[f"q{i}_user"] = user_answer
        st.write("---")


def evaluate_answers():
    correct = 0
    total = len(
        [k for k in st.session_state.user_answers if k.endswith("_correct")]
    )

    for i in range(1, total + 1):
        if (
            st.session_state.user_answers.get(f"q{i}_user")
            == st.session_state.user_answers.get(f"q{i}_correct")
        ):
            correct += 1

    score = (correct / total) * 100 if total > 0 else 0
    st.session_state.score = score
    return score

# Streamlit UI

def main():
    st.title("Quiz Generator")
    st.write("Upload your study materials to generate a practice quiz.")

    # Step 1: Upload Files
    with st.expander("Step 1: Upload Documents", expanded=True):
        files = st.file_uploader(
            "Upload PDF or DOCX Files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Select documents to generate quiz questions from."
        )

        if files and st.button("Process Documents"):
            with st.spinner("Processing files..."):
                text = ""
                pdf_files = [f for f in files if f.name.endswith('.pdf')]
                docx_files = [f for f in files if f.name.endswith('.docx')]
                
                if pdf_files:
                    text += get_pdf_text(pdf_files)
                if docx_files:
                    text += get_docx_text(docx_files)
                
                chunks = get_text_chunks(text)
                
                if not chunks:
                    st.error("No text found in files. Are they scanned or empty?")
                    st.stop()

                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.pdf_processed = True
                st.success("Files processed successfully.")

    # Step 2: Generate Questions
    if st.session_state.pdf_processed:
        with st.expander("Step 2: Create Quiz", expanded=True):
            topic = st.text_input(
                "Enter a topic from the documents:",
                placeholder="e.g., Machine Learning, History, Biology",
                help="Enter the specific topic you want to be tested on (e.g., 'Machine Learning')."
            )

            if topic and st.button("Create Quiz"):
                with st.spinner("Generating quiz..."):
                    questions_data = generate_questions(
                        topic, st.session_state.vector_store
                    )
                    if questions_data:
                        st.session_state.questions_data = questions_data
                        st.session_state.questions_generated = True
                        st.success("Quiz generated successfully.")

    # Step 3: Take Quiz
    if st.session_state.questions_generated:
        with st.expander("Step 3: Take the Quiz", expanded=True):
            display_quiz(st.session_state.questions_data)

            if st.button("Submit Answers"):
                score = evaluate_answers()
                st.success(f"Your score: {score:.1f}%")

                st.write("### Correct Answers")
                for i, question in enumerate(
                    st.session_state.questions_data["questions"], 1
                ):
                    correct = st.session_state.user_answers[f"q{i}_correct"]
                    st.write(
                        f"Question {i}: {correct}) "
                        f"{question['options'][correct]}"
                    )

    # Reset
    if st.session_state.pdf_processed and st.button("Start Over"):
        st.session_state.clear()
        st.rerun()


# Run App

main()
