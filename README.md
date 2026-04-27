# VaultParse: Offline-First Enterprise Document Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-black.svg)](https://www.llamaindex.ai/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local_AI-Ollama-white.svg)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

## Overview
**VaultParse** is an edge-capable, multi-modal Retrieval-Augmented Generation (RAG) pipeline designed for highly sensitive enterprise data (e.g., technical audits, insurance claims, legal contracts). 

Unlike standard cloud-based RAG applications that leak sensitive prompt data to third-party APIs (like OpenAI or Anthropic), VaultParse operates with **100% data privacy**. It ingests messy, mixed-format data, applies localized processing, and utilizes quantized local Large Language Models (LLMs) to securely query and generate structured reports entirely behind the corporate firewall.

**Business Outcome:** Solved the "messy data" problem for legacy enterprise environments by converting unstructured documents into actionable insights, operating with zero network-boundary data leaks through edge computing.

---

## Key Architectural Features

* **Zero-Data-Leak RAG Pipeline:** The entire vectorization (Nomic), storage (ChromaDB), and generation (Gemma 4) process occurs locally on device. No context data is ever sent across the public internet.
* **Smart Document Routing Engine:** Employs a dynamic ingestion architecture. Uses blistering-fast digital text extraction (`PyMuPDF`) as a primary pass, automatically falling back to heavy image-to-text processing (`Tesseract OCR`) *only* for scanned, non-digital pages. This dramatically reduces compute time.
* **Metadata Inheritance & Verifiable Citations:** Built to combat LLM hallucinations. Document chunks are tracked at the page level. When answering queries, the system returns mathematically verified Cosine Similarity scores and exact Source Page numbers for enterprise trust.
* **Optimized Edge Memory Management:** Implements customized KV-cache restrictions and quantized models (4-bit compression) to run complex reasoning tasks on consumer-grade hardware (<4GB RAM footprint) without crashing the host OS.

---

## Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestrator** | `LlamaIndex` | Data framework, chunking (`SentenceSplitter`), and pipeline routing. |
| **Vector Database** | `ChromaDB` | Persistent local storage & HNSW (Approximate Nearest Neighbors) graph search. |
| **Embedding Model** | `Nomic-Embed-Text` | Generates 768-dimensional contextual vectors via Mean Pooling. |
| **Local LLM Engine** | `Gemma 4` (via `Ollama`) | Quantized local intelligence and report synthesis. |
| **Cloud LLM (Optional)** | `Groq` (`Gemma 2 9B`) | Hybrid fallback for ultra-low latency inference testing. |
| **Data Ingestion** | `PyMuPDF` & `Tesseract` | Multi-modal text and image extraction layer. |
| **Frontend UI** | `Streamlit` | Modern, responsive chat interface. |

---

## How It Works (The Pipeline)

1.  **Ingestion & Routing:** A user uploads a PDF. VaultParse reads it page-by-page. Digital pages are extracted instantly; images/scanned pages are routed to Tesseract OCR.
2.  **Semantic Chunking:** The extracted text is sliced into 512-token nodes with a 50-token overlap to preserve contextual boundaries. Page numbers are attached as metadata.
3.  **Embedding & Storage:** Chunks are passed through a local Transformer model to generate vector embeddings. These are stored in a persistent, local ChromaDB instance.
4.  **Retrieval:** A user asks a question. The prompt is vectorized, and ChromaDB calculates Cosine Similarity to find the top 3 most relevant context chunks.
5.  **Synthesis:** The LLM receives the prompt and the retrieved chunks, generating a factual answer backed by specific document citations.

---

## Quick Start & Installation

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed locally.
* Tesseract OCR and Poppler binaries installed on your system PATH.

### 1. Setup the Environment
```bash
git clone [https://github.com/yourusername/VaultParse.git](https://github.com/yourusername/VaultParse.git)
cd VaultParse
python -m venv .venv
source .venv/bin/activate  # On Windows use `\.venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Pull the Local Models
```bash
ollama pull nomic-embed-text
ollama pull gemma:2b  # Or your preferred edge model
```

### 3. Build the Database (Ingest Documents)
Place your enterprise PDFs in the `data/` directory, then run the indexing engine:
```bash
python vaultparse_indexer.py
```
*(This will generate a persistent `vaultparse_db` folder containing your vector graph).*

### 4. Launch the UI
```bash
streamlit run app.py
```

---

## Developer
Built by **Kaaif Ahmed Khan** *Aspiring AI Engineer*

linkedin.com/in/KaaifAhmedKhan