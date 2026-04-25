import os
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

print("Loading the models...")

# Load environment variables from .env file
load_dotenv()

# Replace hardcoded API key with environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = Groq(model="llama-3.3-70b-versatile")

# llm = Ollama(model="batiai/gemma4-e2b:q4", request_timeout=10000.0, context_window=2048, additional_kwargs={"num_ctx": 2048})
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Setting these as defaults for Ollama
Settings.llm = llm 
Settings.embed_model = embed_model

print("Connecting to chroma vector db...")
db = chromadb.PersistentClient(path=r'./vaultparse_db')
collection = db.get_collection('enterprise_docs')

vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store) # loading our vectors from the vector store

print("\nInitializing query engine...")
query_engine = index.as_query_engine(similarity_top_k=3)

print("\nVaultParse is ready! Type \'exit\' to quit.")

while True:
    query = input("\nAsk VaultParse: ")
    if (query.lower() == 'exit'): break
    
    print("\nThinking...")
    response = query_engine.query(query)

    print(f"\n{response.response}")
    print("\n--- Source Citations ---")
    # Here we crack open the response object to prove Metadata Inheritance worked
    for i, node in enumerate(response.source_nodes):
        # node.node.metadata contains the dictionary we attached during ingestion
        source_file = node.node.metadata.get("source_file", "Unknown File")
        page_num = node.node.metadata.get("page_number", "Unknown Page")
        
        # node.score is the exact math score of the similarity match
        print(f"Citation {i+1}: {source_file} (Page {page_num}) - Match Score: {node.score:.4f}")