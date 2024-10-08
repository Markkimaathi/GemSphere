# 📖 Gemini PDF Chatbot 🤖
GemSphere is an intelligent chatbot designed to extract relevant information from PDFs. Upload your PDF files, ask questions, and Gemini will provide detailed answers by analyzing the content in real-time. Powered by LangChain, Google Generative AI, and FAISS for text vectorization and retrieval, this application enables seamless interaction with document content.


# 🚀 Features
PDF Upload and Processing: Upload multiple PDF files and let the chatbot process them to extract content.
Intelligent Q&A: Ask detailed questions about the uploaded PDFs and receive context-based answers.
PDF Content Preview: Get a snippet preview of the content of your uploaded PDFs before submitting.
Interactive Chat Interface: Chat-like interface with chat history that saves user and assistant responses.
Powered by FAISS: Uses FAISS for efficient document similarity search and vector storage.

# 🛠️ Technologies Used
Python: The core language of the application.
Streamlit: Used for building the UI and providing the interactive web interface.
LangChain: For managing large language model chains and embeddings.
Google Generative AI: For embedding generation and chat response.
FAISS (Facebook AI Similarity Search): For efficient document retrieval based on similarity.
PyPDF2: For reading and extracting text from PDF files.
# 📋 Prerequisites
Before running the chatbot locally, ensure you have the following installed:

Python 3.8+
pip (Python package manager)
Streamlit (pip install streamlit)
FAISS (pip install faiss-cpu)
Google Generative AI SDK (pip install google-generativeai)
You will also need an API key from Google Generative AI to enable embeddings and chat generation. Follow this guide to get your API key.

# 📦 Installation
Clone the repository:

git clone https://github.com/Markkimaathi/GemSphere.git
cd gemini-pdf-chatbot
Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt

Configure environment variables:
Create a .env file in the root directory.
Add your Google API Key in the .env file:
GOOGLE_API_KEY=your-google-api-key-here
Run the application:
streamlit run app.py
This will start the Streamlit server, and you can access the chatbot interface at http://localhost:8501.

# 🎨 Usage
1. Upload PDFs:
Click on Upload PDF Files in the sidebar to upload multiple PDFs.
A preview of the first few lines of each PDF will be displayed for confirmation.
2. Ask Questions:
After the PDF is processed, type your question in the input box and hit enter.
Gemini will provide answers based on the content of the uploaded PDFs.
3. Chat History:
All questions and answers are stored in the chat history for easy reference. You can clear the history using the Clear Chat History button.
# 🖥️ Example
> What are the key topics covered in this PDF?
Response:
The PDF discusses topics including machine learning models, optimization techniques, and their applications in healthcare and finance.
# 📚 Code Structure
📂 gemini-pdf-chatbot/
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables file (ignored in git)
├── README.md            # Project documentation
└── assets/              # Folder for images and assets (optional)

# Key Functions:
get_pdf_text(): Extracts text from PDF files.
get_text_chunks(): Splits large text content into smaller, manageable chunks.
get_vector_store(): Converts text chunks into vector embeddings and stores them using FAISS.
user_input(): Handles user input and generates a response using the conversational chain.

# ⚙️ Configuration
API Keys: Ensure your .env file contains the correct Google API Key for embedding and chat generation.

# 🔧 Future Improvements
Support for Other Document Types: Extend support to handle other file formats such as .docx or .txt.
Enhanced UI: Further refine the UI with more theming options, including night mode and chat bubbles.
Deployment: Dockerize the application for easy deployment on cloud platforms like Heroku or AWS.

# 👥 Contributing
Contributions are welcome! If you'd like to improve the chatbot, please fork the repository and submit a pull request.

# 📧 Contact
For any questions, feel free to reach out via email or create an issue on the GitHub repository.
