import streamlit as st
import chromadb
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="VaultParse | Enterprise RAG", layout="centered")
st.title("VaultParse Intelligence")
st.caption("Offline-First Document Extraction & Query System")

# --- 2. CACHE THE BACKEND ---
# @st.cache_resource ensures we only load the DB once, not every time the user types a letter.
@st.cache_resource
def load_vaultparse_engine():

    # Load environment variables from .env file
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    
    # Connect to your existing hard-drive database
    db = chromadb.PersistentClient(path="./vaultparse_db")
    chroma_collection = db.get_collection("enterprise_docs") 
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=3)

query_engine = load_vaultparse_engine()

# --- 3. THE CHAT INTERFACE ---
# Initialize session state to remember chat history on the screen
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. HANDLING USER INPUT ---
if prompt := st.chat_input("Ask about the enterprise documents..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching VaultParse Database..."):
            sys_prompt = "You are a VaultParse AI Assistant, your job is to tell the user about VaultParse using the provided data."
            response = query_engine.query(prompt)
            
            # Print the main answer
            st.markdown(response.response)
            
            # Print the citations beautifully
            st.divider()
            st.markdown("**Sources Retrieved:**")
            for i, node in enumerate(response.source_nodes):
                source_file = node.node.metadata.get("source_file", "Unknown")
                page_num = node.node.metadata.get("page_number", "Unknown")
                st.caption(f"📄 {source_file} (Page {page_num}) - Match: {node.score:.2f}")
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response.response})