# VaultParse
VaultParse: Offline-First Enterprise Document Intelligence. Designed and implemented a completely offline, multi-modal extraction system for highly sensitive enterprise data, such as insurance claims and technical audits. The pipeline ingests messy, mixed-format data (PDFs, text, tables, and images), applies localized OCR, and utilizes a custom RAG architecture powered by quantized local LLMs to securely query and generate structured reports.

## Architecture
### The LLM Brain:
We are using **Gemma 4 E4B** as our LLM Brain. As it is light-weight and can give all the necessary performance.

### Vectorizer (Embedding Model):
We'll be using **Nomic-Embed-Text v1.5** for vectorization.

*Note that we are using Ollama for hosting the models locally*

### Ollama and Models Setup:
Download Ollama: 
```irm https://ollama.com/install.ps1 | iex```
Pull the models:
```ollama pull gemma```
```ollama pull nomic-embed-text```