# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents with PostgreSQL storage.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching using aiohttp
- HTML to text conversion with BeautifulSoup
- PostgreSQL storage with content change detection
- Automatic Latvian to English translation via DeepL API
- Detailed logging
- Web-based monitoring dashboard
- Automatic testing and statistics tracking

## Configuration

The application uses environment variables for configuration:

- `DATABASE_URL`: PostgreSQL connection string (set automatically on Heroku)
- `LINKS_FILE`: Path to a file containing URLs to scrape (default: `links.txt`)
- `DEEPL_API_KEY`: API key for DeepL translation service
- `POLL_INTERVAL`: Seconds between translation checks (default: `60`)
- `BATCH_SIZE`: Number of documents to translate per batch (default: `100`)

## Database Schema

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

# Run the scraper
python scraper.py

# Run the translator in a separate terminal
python translator.py

# Test the database content
python test_db.py
# For detailed output
python test_db.py --verbose
```

## Testing

To verify that the scraper is working correctly, run the test script:

```bash
python test_db.py
```

This will:
- Check that the database tables exist
- Verify that records have been created
- Ensure that records have actual content
- Check for URL uniqueness

For more detailed output, use:

```bash
python test_db.py --verbose
```

## Deployment

This application is deployed on Heroku with:
- PostgreSQL database add-on
- Multiple dyno types:
  - `scraper`: Scheduled web scraping (daily at midnight UTC)
  - `translator`: Background translation service (polls every 60s)
  - `web`: Flask-based status dashboard
- Environment variables for configuration:
  - `DATABASE_URL`: Set automatically by Heroku PostgreSQL add-on
  - `DEEPL_API_KEY`: Must be set manually via Heroku dashboard or CLI
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