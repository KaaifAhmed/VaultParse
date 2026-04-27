'''
We first take our PDFs, extract text from it in pages, and save those pages into an array.
When saving pages, we save them as Document object, including the actual text, and the metadata. That metadata helps in referencing.
Then we pass those pages-array to VectorStoreIndex, which applies transformations to it 
(splitting text into chunks of ~512 tokens and an overlap of ~50 tokens, while trying to preserve sentences), 
use our embedding model which we send it to create vector embeddings of our chunks, and store those vectors in our chroma vector db.
'''


import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import chromadb

# LlamaIndex Imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_docs (file_path):
    print(f"\nExtracting from {file_path}")

    file = fitz.open(file_path)
    docs = []

    for i, page in enumerate(file):
        page = file[i]
        text = page.get_text().strip()
        doc = Document(
                text=text,
                metadata={
                    "source_file": file_path,
                    "page_number": i + 1
                }
            )
        docs.append(doc)
    
    print(f"\nExtracted {len(docs)} pages.")
    return docs

def build_vector_db(docs):
    print("\nBuilding the databases...")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text") # Our Ollama embedding model

    db = chromadb.PersistentClient(path=r".\vaultparse_db") # Creating a chromadb on hard-drive (persistent)
    chroma_collection = db.get_or_create_collection("enterprise_docs") # Creating a collection (like table in a db)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # ChromaDB collection into LlamaIndex understandable format
    storage_context = StorageContext.from_defaults(vector_store=vector_store) # It tells LlamaIndex where to store what in the chromadb

    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50) # Intelligent splitter, that'd split our docs into chunks of 512 tokens (trying to preserve sentences/paragraphs)

    index = VectorStoreIndex.from_documents( # Connect everything together
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter]
    )

    print("\nDatabase built!")
    return index

file_path_1 = 'VaultParse_Project_Documentation.pdf'
documents = extract_docs(file_path_1)
index = build_vector_db(documents)

file_path_2 = 'VaultParse_Technical_Details.pdf'
documents = extract_docs(file_path_2)
index = build_vector_db(documents)