import streamlit as st
import os
from rag_pipeline import build_rag_pipeline, DATA_DIR

st.set_page_config(page_title="Sanskrit RAG System", page_icon="📜", layout="wide")

st.title("📜 Dynamic Sanskrit Document RAG System")
st.markdown("**100% Offline CPU-Based Retrieval-Augmented Generation**")

@st.cache_resource
def load_rag_pipeline():
    # Load LLM and existing Vector DB
    return build_rag_pipeline()

try:
    with st.spinner("Initializing Local LLM and Vector Store... (This might take a minute initially)"):
        rag_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Error initializing system: {str(e)}")
    st.stop()

# --- SIDEBAR FOR DYNAMIC DOCUMENT INGESTION ---
with st.sidebar:
    st.header("1. Upload Documents")
    st.markdown("Upload Sanskrit `.txt` or `.pdf` files to index them into the database.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document"):
            with st.spinner("Processing document and generating local embeddings..."):
                try:
                    os.makedirs(DATA_DIR, exist_ok=True)
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    
                    # Save file to disk
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    # Add to vector store dynamically
                    chunks_added = rag_chain.add_document(file_path)
                    st.success(f"Successfully indexed {chunks_added} chunks from '{uploaded_file.name}'!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

# --- MAIN CHAT INTERFACE ---
st.header("2. Ask Questions")
query = st.text_input("Enter your query in English, Sanskrit, or transliterated text:")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Retrieving context and generating response via Local LLM..."):
            try:
                response = rag_chain.invoke({"input": query})
                
                st.subheader("Answer")
                st.write(response["answer"])
                
                with st.expander("View Retrieved Context"):
                    if response["context"]:
                        for idx, doc in enumerate(response["context"]):
                            st.markdown(f"**Chunk {idx+1}:**")
                            st.write(doc.page_content)
                            st.markdown("---")
                    else:
                        st.write("No context found.")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
