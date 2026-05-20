import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# 1. Document Loader and Preprocessor (Dynamic)
def load_and_preprocess_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
        
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", "।", " ", ""]
    )
    return text_splitter.split_documents(docs)

# 2. Retriever (Vector based)
def get_vector_store():
    # CPU-based HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectorstore

# 3. Generator (LLM-based text generator) - COMPLETELY LOCAL CPU
def get_llm():
    print("Loading local LLM... this may take a few minutes the first time to download.")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        low_cpu_mem_usage=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        repetition_penalty=1.1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# 4. End-to-End Pipeline
class RAGPipeline:
    def __init__(self):
        self.llm = get_llm()
        self.vectorstore = get_vector_store()
        
    def add_document(self, file_path):
        splits = load_and_preprocess_file(file_path)
        if splits:
            self.vectorstore.add_documents(splits)
            return len(splits)
        return 0

    def invoke(self, inputs):
        query = inputs.get("input", "")
        # Retrieve docs (increased K to fetch more precise chunks)
        docs = self.vectorstore.similarity_search(query, k=5)
        if not docs:
            return {"answer": "No relevant context found in the database. Please upload a document first.", "context": []}
            
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # Build prompt strictly formatted for Qwen Chat
        prompt = (
            "<|im_start|>system\n"
            "You are an expert reading comprehension assistant. Read the provided Sanskrit context carefully. "
            "Answer the user's question ONLY using the facts from the context. "
            "Do not invent names, events, or details. Keep the answer extremely brief.\n\n"
            f"Context:\n{context}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        response = self.llm.invoke(prompt)
        
        # Extract only the assistant's answer
        answer = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
        
        return {
            "answer": answer,
            "context": docs
        }

def build_rag_pipeline():
    return RAGPipeline()
