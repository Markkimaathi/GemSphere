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
import speech_recognition as sr
from gtts import gTTS
import io
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define function to read PDF files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Get embeddings for chunks and store them
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Define chat functionality
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF and ask a question"}]

# Handle user input for the chatbot
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    return chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

# Convert text to speech using gTTS and play it
def speak_text(text):
    tts = gTTS(text)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio = AudioSegment.from_file(audio_buffer, format="mp3")
    play(audio)

# Function to capture voice input from user
def capture_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening for your question...")
        audio = recognizer.listen(source)
        try:
            user_input_text = recognizer.recognize_google(audio)
            st.write(f"Recognized: {user_input_text}")
            return user_input_text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand your voice.")
        except sr.RequestError as e:
            st.write(f"Could not request results; {e}")
    return None

# Main function for the Streamlit app
def main():
    # Set up page configuration
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("📁 Gemini PDF Chatbot")
        st.write("Process PDFs and ask questions about their contents.")

        st.header("1️⃣ Upload PDF")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        if pdf_docs:
            st.subheader("Preview of PDFs")
            for pdf in pdf_docs:
                st.write(f"Preview of {pdf.name}:")
                pdf_reader = PdfReader(pdf)
                first_page = pdf_reader.pages[0]
                st.write(first_page.extract_text()[:700])

        if st.button("Process PDF(s)"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF processing complete!")
            else:
                st.error("Please upload a PDF to proceed.")

    # Main content area for displaying chat messages
    st.title("🤖 Chat with your PDFs")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF and ask a question"}]

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.write(content)

    # Option for voice input
    if st.button("Ask a question with voice"):
        user_question = capture_voice_input()
        if user_question:  # Check if voice input was successfully captured
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            # Process the captured question
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(user_question)
                    full_response = "".join(response['output_text'])
                    st.write(full_response)
                    speak_text(full_response)  # Convert the bot's response to speech
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.write("Sorry, I could not capture your voice input. Please try again.")

    # Chat input from user
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Bot response to the user
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                full_response = "".join(response['output_text'])
                st.write(full_response)
                speak_text(full_response)  # Convert the bot's response to speech
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Button to clear chat history
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# Run the app
if __name__ == "__main__":
    main()
