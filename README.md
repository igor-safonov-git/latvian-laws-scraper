# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents with PostgreSQL storage.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching using aiohttp
- HTML to text conversion with BeautifulSoup
- PostgreSQL storage with content change detection
- Automatic Latvian to English translation via DeepL API
- Vector embeddings generation via OpenAI API
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
- `EMBEDDING_MODEL`: OpenAI model to use (default: `text-embedding-3-small`)
- `EMBEDDING_DIMENSIONS`: Vector dimensions (default: `512`)

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

### Vector Embeddings Table

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS docs (
  id        TEXT PRIMARY KEY,
  metadata  JSONB,
  embedding VECTOR(512)
);
```

- `id`: SHA-256 hash of the URL (same as raw_docs.id)
- `metadata`: JSON with URL, timestamp, and preview of the document
- `embedding`: Vector representation (512 dimensions) of the translated text

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
   - Check that all documents have been translated
   - Clear previous embeddings
   - Generate new embeddings using OpenAI API
   - Store embeddings in the docs table for vector search
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
python embedder.py

# Test each component
python test_db.py            # Test database and scraper
python test_translator.py    # Test translator functionality
python test_embedder.py      # Test embedder functionality

# For detailed output with samples
python test_db.py --verbose
python test_translator.py --verbose
python test_embedder.py --verbose

# Additional test options
python test_translator.py --check-one  # Test a specific document's translation
python test_embedder.py --sample       # Show sample embedding values
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
python test_embedder.py
```

This will:
- Check that the vector extension is enabled
- Verify that the docs table exists
- Confirm that embeddings have been generated
- Check that all translated documents have embeddings
- Verify the embedding dimensions
- Validate metadata format

For detailed output with embedding samples:

```bash
python test_embedder.py --verbose --sample
```

## Deployment

This application is deployed on Heroku with:
- PostgreSQL database add-on
- Multiple dyno types:
  - `scraper`: Scheduled web scraping (daily at midnight UTC)
  - `translator`: Background translation service (polls every 60s)
  - `embedder`: Vector embedding generation (daily at 00:30 UTC)
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
```