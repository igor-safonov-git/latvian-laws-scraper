# RAG with LLM Integration

A simple solution for answering questions about Latvian laws using Retrieval-Augmented Generation (RAG) with an LLM.

## Overview

This module combines:
1. RAG system to retrieve relevant context from the database
2. LLM integration to generate answers based on the retrieved context

## Features

- Asynchronous implementation
- Contextual responses from RAG
- OpenAI API integration with search tool enabled
- Simple prompt formatting
- Basic logging

## Requirements

- Python 3.7+
- OpenAI API key
- PostgreSQL with pgvector extension
- Required Python packages:
  - aiohttp
  - asyncio
  - python-dotenv
  - asyncpg

## Setup

1. Ensure your `.env` file contains:
```
OPENAI_API_KEY=your-api-key
MODEL=gpt-4  # default, can be changed
DATABASE_URL=your-database-url
```

2. Make sure PostgreSQL with pgvector extension is running and properly set up

## Usage

```bash
# Ask a question directly
python rag_with_llm.py "What are the regulations for digital signatures in Latvia?"

# Or run interactively
python rag_with_llm.py
```

## How It Works

1. The question is sent to the RAG system to retrieve relevant passages from the database
2. Retrieved passages are formatted into context
3. The question and context are sent to the LLM
4. The LLM generates an answer based on the provided context
5. Answer and sources are displayed to the user

## Implementation

The system is built with two main components:

1. `rag.py` - Handles vector search and retrieval
2. `llm_client/` - Provides the LLM integration

The `rag_with_llm.py` script connects these components to provide a complete solution.