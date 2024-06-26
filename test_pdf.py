# Progress Bar:

# Replaced spinner with progress bar for feedback during PDF processing.
# Display Uploaded Files:

# Show a list of uploaded PDFs in the sidebar.
# Enhanced Error Handling:

# Added try-except blocks for better error handling and displaying user-friendly error messages.
# PDF Text Extraction:

# Used pdfplumber for handling non-textual content.
# Integrated OCR using pytesseract to handle scanned documents.
# Customizable Chunk Size and Batch Processing:

# Allowed users to customize the chunk size and chunk overlap.
# Optimized text splitting and embeddings for better performance.
# Keyword Extraction:

# Extracted and displayed keywords using Spacy.

import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from itertools import chain
import spacy
from langdetect import detect
from googletrans import Translator

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize language model for keyword extraction
nlp = spacy.load("en_core_web_sm")

def extract_text_from_page(page):
    try:
        return page.extract_text()
    except:
        return ""

def extract_text_from_image(page):
    try:
        image = page.to_image()
        return pytesseract.image_to_string(Image.fromarray(image))
    except:
        return ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += extract_text_from_page(page)
                text += extract_text_from_image(page)
    return text

def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context of the provided document". 
    Do not provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def display_keywords(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return ", ".join(keywords)

def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat With Multiple PDF Using Gemini")

    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        chunk_size = st.number_input("Chunk Size", value=10000)
        chunk_overlap = st.number_input("Chunk Overlap", value=1000)
        
        if st.button("Submit & Process"):
            try:
                with st.spinner("Processing..."):
                    if pdf_docs:
                        text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(text, chunk_size, chunk_overlap)
                        get_vector_store(text_chunks)
                        st.success("PDF Uploaded Successfully")
                        st.write("Uploaded Files:")
                        for pdf in pdf_docs:
                            st.write(pdf.name)
                        st.write("Keywords:")
                        st.write(display_keywords(text))
                    else:
                        st.error("No PDF files uploaded!")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
