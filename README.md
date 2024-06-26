# Chat With Multiple PDFs Using Gemini

This application allows you to upload multiple PDF documents, process them, and then ask questions about the content of these documents. The application uses Google's Generative AI for embedding and conversational AI to answer questions based on the provided context from the PDFs.

## Features

- Upload multiple PDF documents.
- Process and extract text from the PDFs.
- Split the extracted text into manageable chunks for efficient processing.
- Create a vector store using FAISS for quick similarity searches.
- Use Google Generative AI for embedding and conversational responses.
- Ask questions about the uploaded PDFs and get detailed answers based on the context.

## Requirements

- Streamlit
- PyPDF2
- Langchain
- Google Generative AI
- FAISS
- Python Dotenv

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chat-with-pdfs.git
   cd chat-with-pdfs
2. Install the required packages:
   ```bash
    pip install streamlit PyPDF2 langchain google-generativeai langchain_community faiss-cpu python-dotenv
3. Set up your Google API key:
   Create a .env file in the root of your project.
   Add your Google API key to the .env file:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
## Usage
1. Run the Streamlit application:
   ```bash
    streamlit run app.py
2. Open your web browser and go to http://localhost:8501.

3. In the sidebar, upload your PDF documents.

4. Click the "Submit & Process" button to process the PDFs.

5. Enter a question about your documents in the text input field and press Enter to get a detailed response based on the context of the uploaded PDFs.
![image](https://github.com/Subhradyuti/Chat-With-Multiple-PDFs-Using-Gemini/assets/133640355/36d303af-4255-4949-aab9-dd4edeecdbd6)

## Code Overview

### PDF Processing

- **get_pdf_text(pdf_docs):** Extracts text from the uploaded PDF documents.
- **get_text_chunks(text):** Splits the extracted text into manageable chunks using `RecursiveCharacterTextSplitter`.
- **get_vector_store(text_chunks):** Creates a vector store from the text chunks using FAISS and Google Generative AI embeddings.

### Conversational AI

- **get_conversational_chain():** Sets up the conversational AI chain using Google Generative AI and a custom prompt template.
- **user_input(user_question):** Handles user questions, performs a similarity search in the vector store, and generates a response using the conversational AI chain.

### Streamlit Application

- **main():** Sets up the Streamlit interface, handles file uploads, and processes user questions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
