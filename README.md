# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents with PostgreSQL storage.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching using aiohttp
- HTML to text conversion with BeautifulSoup
- PostgreSQL storage with content change detection
- Automatic Latvian to English translation via DeepL API
- Vector embeddings generation via OpenAI API
- Enhanced embeddings with document chunking and summarization
- Vector search capabilities with pgvector
- Detailed logging
- Web-based monitoring dashboard
- Automatic testing and statistics tracking

## Configuration

The application uses environment variables for configuration:

- `DATABASE_URL`: PostgreSQL connection string (set automatically on Heroku)
- `LINKS_FILE`: Path to a file containing URLs to scrape (default: `links.txt`)
- `DEEPL_API_KEY`: API key for DeepL translation service
- `OPENAI_API_KEY`: API key for OpenAI embedding service
- `POLL_INTERVAL`: Seconds between translation checks (default: `60`)
- `BATCH_SIZE`: Number of documents to translate per batch (default: `100`)
- `MAX_TOKENS`: Maximum tokens for embedding generation (default: `8192`)
- `EMBEDDING_MODEL`: OpenAI model to use (default: `text-embedding-3-large`)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (default: `3072`)
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
3. The translator will:
   - Run every 60 seconds (configurable)
   - Find untranslated documents (where processed = FALSE)
   - Translate text from Latvian to English using DeepL API
   - Store translated text and mark document as processed
   - Log translation results to the console and to a logfile
4. The embedder will:
   - Run daily at 00:30 UTC (after translator completes)
   - Check for new or modified translations
   - When new translations are found:
     - Clear previous embeddings
     - Generate AI-powered summaries of each document
     - Split documents into overlapping chunks with semantic boundary detection
     - Generate embeddings for the full document, each chunk, and the summary
     - Store all embeddings in their respective tables for multi-representation search
     - Record timestamp of successful run
   - Log embedding results to the console and to a logfile

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
python translator.py
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
- Multiple dyno types:
  - `scraper`: Scheduled web scraping (daily at midnight UTC)
  - `translator`: Background translation service (polls every 60s)
  - `embedder`: Vector embeddings with chunking and summarization (daily at 00:30 UTC)
  - `web`: Flask-based status dashboard
- Environment variables for configuration:
  - `DATABASE_URL`: Set automatically by Heroku PostgreSQL add-on
  - `DEEPL_API_KEY`: For DeepL translation API
  - `OPENAI_API_KEY`: For OpenAI embeddings API
- PostgreSQL extensions:
  - `pgvector`: Enabled for vector storage and search
- URL: https://latvian-laws-06e89c613b8a.herokuapp.com/

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
heroku run python test_db.py --app latvian-laws
heroku run python test_translator.py --app latvian-laws
heroku run python test_embedder_enhanced.py --app latvian-laws
```