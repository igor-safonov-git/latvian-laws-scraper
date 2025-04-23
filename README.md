# Latvian Legal Documents RAG System

An end-to-end system for scraping, translating, and searching Latvian legal documents with a Telegram chatbot interface.

## Features

- Scheduled scraping (daily at midnight UTC)
- On-demand translation via DeepL API
- On-demand vector embedding generation
- Asynchronous fetching using aiohttp
- HTML to text conversion with BeautifulSoup
- PostgreSQL storage with content change detection
- Automatic Latvian to English translation via DeepL API
- Vector embeddings generation via OpenAI API
- Enhanced embeddings with document chunking and summarization
- Vector search capabilities with pgvector
- Retrieval Augmented Generation (RAG) for context-aware answers
- Telegram bot interface for user interaction
- Web-based monitoring dashboard
- Detailed logging
- Automatic testing and statistics tracking

## Configuration

The application uses environment variables for configuration:

- `DATABASE_URL`: PostgreSQL connection string (set automatically on Heroku)
- `LINKS_FILE`: Path to a file containing URLs to scrape (default: `links.txt`)
- `DEEPL_API_KEY`: API key for DeepL translation service
- `OPENAI_API_KEY`: API key for OpenAI embedding and LLM services
- `TELEGRAM_API_KEY`: API key for Telegram bot
- `POLL_INTERVAL`: Seconds between translation checks (default: `60`)
- `BATCH_SIZE`: Number of documents to translate per batch (default: `10`)
- `MAX_TOKENS`: Maximum tokens for embedding generation (default: `8192`)
- `MAX_CONTEXT_TOKENS`: Maximum tokens for RAG context (default: `8000`)
- `EMBEDDING_MODEL`: OpenAI model to use (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (default: `1536`)
- `MODEL`: OpenAI LLM model to use (default: `gpt-4`)
- `SIMILARITY_THRESHOLD`: Threshold for vector similarity (default: `0.3`)
- `TOP_K`: Maximum number of context snippets to retrieve (default: `5`)
- `CHUNK_SIZE`: Characters per document chunk (default: `3000`)
- `CHUNK_OVERLAP`: Character overlap between chunks (default: `500`)
- `SUMMARY_LENGTH`: Target character length for summaries (default: `3000`)

## Database Schema

### Raw Documents Table

```sql
CREATE TABLE IF NOT EXISTS raw_docs (
  id               TEXT PRIMARY KEY,
  url              TEXT NOT NULL,
  fetched_at       TIMESTAMPTZ NOT NULL,
  raw_text         TEXT NOT NULL,
  processed        BOOLEAN DEFAULT FALSE,
  translated_text  TEXT
);
```

- `id`: SHA-256 hash of the URL
- `url`: Source URL
- `fetched_at`: Timestamp of retrieval
- `raw_text`: Plain text extracted from HTML in Latvian
- `processed`: Flag indicating if the document has been processed/translated
- `translated_text`: English translation of the raw text

### Vector Embeddings Tables

```sql
-- Main document embeddings
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS docs (
  id        TEXT PRIMARY KEY,
  metadata  JSONB,
  embedding VECTOR(3072)
);

-- Document chunk embeddings
CREATE TABLE IF NOT EXISTS doc_chunks (
  id          TEXT NOT NULL,
  chunk_id    TEXT PRIMARY KEY,
  chunk_index INTEGER NOT NULL,
  chunk_text  TEXT NOT NULL,
  metadata    JSONB,
  embedding   VECTOR(3072)
);

-- Document summary embeddings
CREATE TABLE IF NOT EXISTS doc_summaries (
  id           TEXT PRIMARY KEY,
  summary_text TEXT NOT NULL,
  metadata     JSONB,
  embedding    VECTOR(3072)
);
```

- `docs.id`: SHA-256 hash of the URL (same as raw_docs.id)
- `docs.metadata`: JSON with URL, timestamp, and preview of the document
- `docs.embedding`: Vector representation (3072 dimensions) of the full document

- `doc_chunks.id`: Reference to the document ID
- `doc_chunks.chunk_id`: Unique chunk identifier (hash of document ID + chunk index)
- `doc_chunks.chunk_index`: Sequential index of the chunk in the document
- `doc_chunks.chunk_text`: Text content of this document chunk
- `doc_chunks.metadata`: JSON with URL, timestamp, chunk index, and text preview
- `doc_chunks.embedding`: Vector representation of the chunk text

- `doc_summaries.id`: SHA-256 hash of the URL (same as docs.id)
- `doc_summaries.summary_text`: AI-generated summary of the document
- `doc_summaries.metadata`: JSON with URL, timestamp, and summary length
- `doc_summaries.embedding`: Vector representation of the summary text

## Usage

1. Add URLs to `links.txt` (one per line)
2. The scraper will:
   - Run daily at 00:00 UTC
   - Fetch and parse each URL
   - Store raw content in PostgreSQL database
   - Log results to the console and to a logfile
3. To translate documents:
   - Run the translator manually when needed:
   ```bash
   python translator.py
   ```
   - This will process a batch of untranslated documents
   - Translate text from Latvian to English using DeepL API
   - Store translated text and mark document as processed
4. To generate embeddings:
   - Run the embedder manually when needed:
   ```bash
   python embedder.py --once
   ```
   - This will **completely clear all vector tables** and regenerate all embeddings
   - Generate AI-powered summaries of each document
   - Split documents into overlapping chunks with semantic boundary detection
   - Generate embeddings for the full document, each chunk, and the summary

## Local Development

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with configuration
echo "DATABASE_URL=postgresql://username:password@localhost/database" > .env
echo "LINKS_FILE=links.txt" >> .env
echo "DEEPL_API_KEY=your_deepl_api_key" >> .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env

# Run each service in a separate terminal
python scraper.py
python embedder_enhanced.py

# Test each component
python test_db.py                 # Test database and scraper
python test_translator.py         # Test translator functionality
python test_embedder_enhanced.py  # Test embedder functionality

# For detailed output with samples
python test_db.py --verbose
python test_translator.py --verbose
python test_embedder_enhanced.py --verbose

# Additional test options
python test_translator.py --check-one  # Test a specific document's translation
python test_embedder_enhanced.py --sample  # Show sample embedding values
```

## Testing

### Testing the Scraper
To verify that the scraper is working correctly:

```bash
python test_db.py
```

This will:
- Check that the database tables exist
- Verify that records have been created
- Ensure that records have actual content
- Check for URL uniqueness

### Testing the Translator
To verify that the translator is working correctly:

```bash
python test_translator.py
```

This will:
- Check that the translated_text column exists
- Verify that documents have been translated
- Ensure that translated documents are marked as processed
- Check for translation quality (length validation)

To test a specific document's translation:

```bash
python test_translator.py --check-one
```

### Testing the Embedder
To verify that the embedder is working correctly:

```bash
python test_embedder_enhanced.py
```

This will:
- Check that the vector extension is enabled
- Verify that all embedding tables exist (docs, doc_chunks, doc_summaries)
- Confirm that embeddings have been generated
- Check that document chunks have been generated
- Verify that document summaries have been generated
- Check the distribution of chunks per document
- Validate embedding dimensions across all tables
- Show summary statistics for all storage components

For detailed output with embedding samples:

```bash
python test_embedder_enhanced.py --verbose --sample
```

### Testing Embedder Edge Cases
To verify that the embedder handles challenging scenarios correctly:

```bash
python test_embedder_edge_cases.py
```

This will test:
- Extremely long documents (>100K characters)
- Very small documents (<100 characters)
- Documents with unusual special characters
- NULL or placeholder embeddings
- Orphaned records or reference integrity issues
- Consistency of embedding dimensions

For detailed analysis:

```bash
python test_embedder_edge_cases.py --verbose
```

## Deployment

This application is deployed on Heroku with:
- PostgreSQL database add-on
- Dyno types:
  - `scraper`: Scheduled web scraping (daily at midnight UTC)
  - `web`: Flask-based status dashboard
  - `bot`: Telegram bot interface for the RAG-LLM system
- On-demand processes (run manually):
  - `translator.py`: Document translation service
  - `embedder.py --once`: Vector embeddings generation
- Environment variables for configuration:
  - `DATABASE_URL`: Set automatically by Heroku PostgreSQL add-on
  - `DEEPL_API_KEY`: For DeepL translation API
  - `OPENAI_API_KEY`: For OpenAI embeddings and LLM API
  - `TELEGRAM_API_KEY`: For Telegram bot interface
- PostgreSQL extensions:
  - `pgvector`: Enabled for vector storage and search
- URL: https://latvian-laws-06e89c613b8a.herokuapp.com/
- Telegram Bot: @latvian_laws_bot

### Running Manual Processes on Heroku

To run the translator on Heroku:
```bash
heroku run python translator.py --app latvian-laws
```

To run the embedder on Heroku:
```bash
heroku run python embedder.py --once --app latvian-laws
```
This command will completely clear all vector tables and regenerate ALL embeddings from scratch.

To see if there are untranslated documents:
```bash
heroku run python -c "import psycopg2, os; conn = psycopg2.connect(os.environ['DATABASE_URL']); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM raw_docs WHERE processed = FALSE AND raw_text IS NOT NULL'); print(f'Documents waiting for translation: {cur.fetchone()[0]}');" --app latvian-laws
```

To see if there are translated documents ready for embedding:
```bash
heroku run python -c "import psycopg2, os; conn = psycopg2.connect(os.environ['DATABASE_URL']); cur = conn.cursor(); cur.execute('SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NOT NULL'); print(f'Documents ready for embedding: {cur.fetchone()[0]}');" --app latvian-laws
```

### Monitoring

The application provides a simple monitoring dashboard:

- **`/`**: Basic status page showing the application is online
- **`/status`**: Detailed status information including:
  - Database connection status
  - Record counts
  - Latest document fetched
  - Recent scraper runs
  - Document change statistics

### Running Tests

To run tests on Heroku:

```bash
heroku run python tests/test_db.py --app latvian-laws
heroku run python tests/test_translator.py --app latvian-laws
heroku run python tests/test_embedder_enhanced.py --app latvian-laws
```

## Repository Structure

The codebase is organized as follows:

- Core components:
  - `scraper.py`: Law document scraper
  - `translator.py`: Document translator
  - `embedder_optimized.py`: Document embedder
  - `rag.py`: Retrieval augmented generation
  - `rag_with_llm.py`: Integration of RAG and LLM systems
  - `llm_client/`: LLM client modules
  - `telegram_llm_bot.py`: Telegram bot interface
  - `app.py`: Web interface
  
- Support directories:
  - `tests/`: Test scripts
  - `legacy/`: Previous versions and deprecated files
  - `development/`: Development and experimental files

## Telegram Bot Usage

The Telegram bot provides a simple interface to the RAG-LLM system:

1. Start the bot by sending `/start` to @latvian_laws_bot
2. Send a question about Latvian laws, for example:
   - "What are the VAT tax rates in Latvia?"
   - "How are digital signatures regulated in Latvia?"
   - "What are the requirements for personal income tax in Latvia?"
3. The bot will:
   - Search for relevant context in the database
   - Generate an answer based on the retrieved context
   - Provide source links for the information