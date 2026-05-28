import docx
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_pdf_text(pdf_docs):
    """Extract text from one or more PDF files."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_docx_text(docx_docs):
    """Extract text from one or more DOCX files."""
    text = ""
    for doc_file in docx_docs:
        doc = docx.Document(doc_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def get_text_chunks(text):
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        add_start_index=True
    )
    return splitter.split_text(text)
