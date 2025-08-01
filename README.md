# alwasaet-qa-document
Agentic RAG application that enables users to upload documents, interact through a chat interface, and receive accurate answers with citations including page numbers from the original sources.



# 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │  Document       │    │  Vector Store   │
│   (Interface)   │◄──►│  Processor      │◄──►│  (Qdrant)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Agentic RAG    │    │  RAG Engine     │    │  Gemini LLM &   │
│  (Orchestrator) │◄──►│  (Retrieval)    │◄──►│  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```