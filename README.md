<div align="center">

# 🎓 Academic QA RAG Chatbot

A local Retrieval-Augmented Generation (RAG) chatbot for asking questions over academic PDFs and YouTube lecture transcripts.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-blue?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama3.1-red?style=for-the-badge)

</div>

---

# 📌 Overview

This project is a local RAG-based chatbot that answers questions using information retrieved from embedded academic documents and YouTube lecture transcripts.

The chatbot uses:
- **ChromaDB** for persistent vector storage
- **HuggingFace embeddings** for semantic search
- **Groq LLM** for response generation
- **LangGraph** for conversational memory
- **Streamlit** for the chat interface

It supports:
- Multi-turn conversations
- Semantic document retrieval
- Source citations
- Persistent vector database loading

---

# ✨ Features

- 💬 Conversational chat interface using Streamlit
- 🧠 Semantic retrieval using dense embeddings
- 📚 Question answering over PDFs and transcripts
- 🗄️ Persistent Chroma vector database
- 🧵 Conversation memory using LangGraph
- ⚡ Fast inference using Groq API
- 🔖 Source document and page references

---

# 🏗️ Architecture

```mermaid
flowchart LR
    A[User Question] --> B[Streamlit UI]
    B --> C[LangGraph Memory]
    C --> D[ChromaDB Retrieval]
    D --> E[Groq LLM]
    E --> B
