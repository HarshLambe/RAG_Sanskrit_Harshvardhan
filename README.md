# Sanskrit Document RAG System

This project is an end-to-end Retrieval-Augmented Generation (RAG) system for Sanskrit documents, strictly adhering to CPU-based constraints for local operations, using an optimal local HuggingFace embedding model for vector representation and a lightweight local LLM (`Qwen1.5-0.5B-Chat`) for generation without any external API calls.

## Project Structure
- `/code/`: Contains the implementation scripts (`rag_pipeline.py`, `app.py`).
- `/data/`: Contains the sample Sanskrit document corpus (`Rag-docs.txt`).
- `/report/`: Contains the generated PDF report detailing architecture and performance.

## Features
- **100% Offline & Local**: The system requires no API keys and makes no external API calls after the initial model download.
- **Strict CPU Compliance**: Both Vector search (`paraphrase-multilingual-MiniLM-L12-v2`) and the Generation LLM (`Qwen1.5-0.5B-Chat`) run completely on the local CPU, eliminating GPU dependencies.
- **Multilingual Pipeline**: Capable of ingesting raw Sanskrit text and splitting appropriately.
- **Interactive Interface**: Powered by Streamlit for clean query execution and context verification.

## Prerequisites
1. Python 3.9+

## Setup Instructions

1. **Environment Setup**:
   Install all necessary dependencies.
   ```bash
   pip install -r requirements.txt
   ```
   *Alternatively:*
   ```bash
   pip install langchain langchain-community langchain-huggingface sentence-transformers chromadb python-dotenv langchain-google-genai streamlit fpdf
   ```

2. **Running the Application**:
   Navigate to the `code/` directory and run the Streamlit app.
   ```bash
   cd code
   python -m streamlit run app.py
   ```
   
   On the first run, the system will automatically ingest `Rag-docs.txt`, split it into chunks, embed them using the CPU model, and create a local Vector Store database in `chroma_db/`.

## Usage
- Open the provided Streamlit local URL (usually `http://localhost:8501`).
- Enter a query in Sanskrit, English, or transliterated text.
- View the generated answer alongside the exact text chunks retrieved from the document corpus.

