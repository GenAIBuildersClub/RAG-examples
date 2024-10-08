import streamlit as st
import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# read the first line of the text file Open AI coding.txt with encoding utf-8

with open(r'C:\Users\dries.faems\Biografie\Open AI coding.txt', 'r', encoding = 'utf-8') as file:
    api_key = str(file.readline())[1:-1]

# environment variable `OPENAI_API_KEY`
os.environ["OPENAI_API_KEY"] = api_key


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Streamlit UI
def main():
    st.title("PDF Chatbot with Streamlit")
    st.write("Upload a PDF and ask questions!")

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if pdf_file is not None:
        # Extract text from the uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(pdf_file)
            st.success("Text extracted successfully!")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text)

        # Generate embeddings
        embeddings = OpenAIEmbeddings()

        # Create a vector store using FAISS
        vector_store = FAISS.from_texts(texts, embeddings)

        # Set up RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )

        # User input for questions
        st.write("### Ask your questions below:")
        user_question = st.text_input("Your question:")

        if user_question:
            # Get the answer from the QA chain
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(user_question)
                st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()