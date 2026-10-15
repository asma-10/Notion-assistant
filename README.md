# Notion Assistant

A chat assistant connected to a Notion database. Ask questions about your Notion content through a RAG pipeline, and (planned) let AI agents create and write content back to Notion.

## Overview

This project links a chat interface to a Retrieval-Augmented Generation (RAG) system built on top of a Notion database. Instead of searching through pages manually, you ask a question in the chat and get an answer based on the content of your database.

The next stage adds agents that can act on the database: writing and creating new content in Notion.

## Project Status

**Done**
- RAG system built with LlamaIndex
- Backend API built with FastAPI
- Notion connected as the data source
- Supabase connected as the vector store
- Ollama connected for local LLM and embeddings
- Chat system working
- Prompt versioning system

**Planned**
- Agentic system with LangGraph
- Agents that run tasks on the Notion database (write and create content)

## Tech Stack

| Tool | Role | Why |
|------|------|-----|
| **LlamaIndex** | RAG pipeline | Provides a solid RAG architecture out of the box: data connectors to load content from sources like Notion, indexing, vector store integrations, and retrieval and query components. This avoids building each stage from scratch. |
| **Supabase** | Vector store | Gives good vector store management. The dataset is small, which makes it a good fit for this use case. |
| **Notion** | Data source, and the tool the agents will manage | The content lives in a Notion database. Its API lets the system read the data now and, later, let agents write to it. |
| **LangGraph** | Agentic system (planned) | A popular, widely adopted tool for building agents. |
| **Ollama** | Runs the models locally: `nomic-embed-text` (embeddings) and `qwen2:0.5b` (LLM) | Free to use, with no paid LLM API needed. Models run locally. |
| **FastAPI** | Backend API | Lightweight Python framework for exposing the RAG system and chat as API endpoints, with automatic interactive documentation. |

## Techniques

- **Retrieval-Augmented Generation (RAG):** answers are generated from the content of the Notion database rather than from the model alone.
- **Vector search:** Notion content is embedded with `nomic-embed-text` and stored in Supabase.
- **Local LLM inference:** embeddings and answer generation run through Ollama, at no API cost.
- **Prompt versioning:** prompts are managed with a version config file, and the active version is loaded at runtime.

## Project Structure

```
Notion-social-media-assistant/
├── app/
│   ├── routes/
│   │   ├── chat.py          # Chat endpoint
│   │   └── ingest.py        # Ingestion endpoint
│   ├── embed.py             # RAG functions
│   ├── ingestion.py         # RAG functions
│   ├── query.py             # RAG functions
│   ├── main.py              # FastAPI entry point
│   ├── notion_loader.py     # Loads data from Notion
│   └── schemas.py           # Request/response schemas
├── config/
│   ├── config.py            # Environment settings values
│   ├── prompt_loader.py     # Loads the active prompt version
│   └── prompts.yaml         # Prompt definitions
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## To Be Continued
