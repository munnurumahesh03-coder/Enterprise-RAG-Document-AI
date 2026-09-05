# 📄 Enterprise Document AI & RAG Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg )](https://www.python.org/ )
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B.svg )](https://enterprise-rag-document-ai-aebfomlonaecv4gwvijn9g.streamlit.app/ )
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg )](https://langchain.com/ )
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg )](https://www.trychroma.com/ )

### 🚀 Live Application: [Try the Enterprise RAG Engine Here](https://enterprise-rag-document-ai-aebfomlonaecv4gwvijn9g.streamlit.app/ )

## 📌 Overview
This project is a Universal Retrieval-Augmented Generation (RAG) SaaS platform designed to bridge the gap between Large Language Models and private, unstructured enterprise data. It allows users to upload any PDF document (research papers, HR manuals, financial reports) and instantly query it using natural language. The system is strictly engineered to eliminate LLM hallucinations by grounding all answers exclusively in the retrieved vector context.

## 🎯 Business Use Cases
* **Legal & Compliance:** Instantly query massive contracts for specific clauses without manual reading.
* **Technical Research:** Extract exact mathematical formulas, parameters, and methodologies from dense AI research papers.
* **Enterprise HR:** Allow employees to chat with company policy documents to get instant, accurate answers.

## 🧠 The RAG Architecture Pipeline
1. **Document Ingestion:** The user uploads a PDF. `PyPDFLoader` extracts the raw text while preserving document structure.
2. **Semantic Chunking:** The text is passed through a `RecursiveCharacterTextSplitter`. It is chopped into 1000-character chunks with a 200-character overlap to ensure sentences are not cut in half and semantic context is preserved.
3. **Vector Embedding:** Each chunk is processed by HuggingFace's `all-MiniLM-L6-v2` embedding model, translating the human text into high-dimensional mathematical vectors.
4. **Vector Storage & Search:** These vectors are stored in **ChromaDB**. When a user asks a question, the query is embedded, and ChromaDB performs a rapid similarity search to retrieve the top 3 most relevant chunks.
5. **Contextual Generation:** The retrieved chunks are injected into a strict LangChain prompt template. The Groq LLM (`qwen/qwen3.8-27b`) reads the context and synthesizes a final, hallucination-free answer.

## ⚙️ Key Engineering Decisions
| Component | Technology Chosen | Engineering Rationale |
| :--- | :--- | :--- |
| **Vector Database** | ChromaDB | Lightweight, in-memory vector store perfect for rapid prototyping and Streamlit Cloud deployment without external database hosting. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Open-source HuggingFace model that provides an optimal balance between semantic accuracy and CPU-based inference speed. |
| **LLM Inference** | Groq (Qwen 27B) | Groq's LPU architecture provides ultra-low latency generation, ensuring the chat interface feels instantaneous. |
| **Anti-Hallucination** | Strict Prompting | Engineered the system prompt to explicitly command the LLM to output *"I cannot answer this"* if the vector search returns irrelevant context. |

## 📁 Repository Structure
* `app.py`: The main Streamlit application containing the UI, file uploader, and LangChain LCEL (LangChain Expression Language) pipeline.
* `Enterprise_RAG_Architecture.ipynb`: The backend prototyping notebook demonstrating the step-by-step vector math and document chunking logic.
* `requirements.txt`: Dynamically generated dependencies optimized for Streamlit Community Cloud deployment.

## 💻 Local Installation & Setup
If you wish to run this RAG pipeline locally on your own machine:

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Enterprise-RAG-Document-AI.git
cd Enterprise-RAG-Document-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your Groq API Key
mkdir .streamlit
echo 'GROQ_API_KEY = "your_api_key_here"' > .streamlit/secrets.toml

# 4. Launch the application
streamlit run app.py

