import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Cache directory for vector store
CACHE_DIR = tempfile.gettempdir()
VECTOR_STORE_PATH = os.path.join(CACHE_DIR, "faiss_index")

# Functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {page_number}]\n{page_text}"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

def load_or_create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        get_vector_store(chunks)
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the provided
    context, respond with "The answer is not available in the context." Adhere to specific instructions provided in the
    question, but allow context-driven follow-ups without repetition of keywords.

    Context:
    {context}
    Question:
    {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Upload a PDF and ask me anything about it."}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def search_pdf(keyword, text):
    results = []
    for line in text.split("\n"):
        if keyword.lower() in line.lower():
            results.append(line)
    return results

# Main App
def main():
    # Page configuration
    st.set_page_config(page_title="Gemini PDF Chatbot", page_icon="🤖", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("📁 Gemini PDF Chatbot")
        st.write("Explore PDFs with ease. Upload, process, and chat with your documents.")

        st.header("1️⃣ Upload PDF")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        if pdf_docs:
            st.subheader("PDF Previews:")
            for pdf in pdf_docs:
                try:
                    st.write(f"**Preview of {pdf.name}:**")
                    pdf_reader = PdfReader(pdf)
                    first_page = pdf_reader.pages[0]
                    st.write(first_page.extract_text()[:700])
                except Exception as e:
                    st.error(f"Error processing {pdf.name}: {e}")

        if st.button("Process PDF(s)"):
            if pdf_docs:
                with st.spinner("Processing PDF(s)..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    load_or_create_vector_store(text_chunks)
                    st.success("Processing complete!")
                    st.session_state.raw_text = raw_text
            else:
                st.error("Please upload at least one PDF to proceed.")

        st.header("2️⃣ Tools")
        if st.button("Clear Chat History"):
            clear_chat_history()

        keyword = st.text_input("Search PDF for keyword")
        if st.button("Search"):
            if "raw_text" in st.session_state:
                results = search_pdf(keyword, st.session_state.raw_text)
                if results:
                    st.success("Search Results:")
                    for result in results:
                        st.write(result)
                else:
                    st.warning("No matches found.")
            else:
                st.error("Please process a PDF first.")

    # Chat area
    st.title("🤖 Chat with your PDFs")

    # Initialize chat history
    if "messages" not in st.session_state:
        clear_chat_history()

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

    # User input
    if prompt := st.chat_input(placeholder="Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Bot response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()
