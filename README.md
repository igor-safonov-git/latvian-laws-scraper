# Latvian Laws Scraper

An asynchronous scraper for Latvian legal documents with PostgreSQL storage.

## Features

- Scheduled scraping (daily at midnight UTC)
- Asynchronous fetching using aiohttp
- HTML to text conversion with BeautifulSoup
- PostgreSQL storage with content change detection
- Detailed logging

## Configuration

The application uses environment variables for configuration:

- `DATABASE_URL`: PostgreSQL connection string (set automatically on Heroku)
- `LINKS_FILE`: Path to a file containing URLs to scrape (default: `links.txt`)

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS raw_docs (
  id          TEXT PRIMARY KEY,
  url         TEXT NOT NULL,
  fetched_at  TIMESTAMPTZ NOT NULL,
  raw_text    TEXT NOT NULL,
  processed   BOOLEAN DEFAULT FALSE
);
```

- `id`: SHA-256 hash of the URL
- `url`: Source URL
- `fetched_at`: Timestamp of retrieval
- `raw_text`: Plain text extracted from HTML
- `processed`: Flag indicating if the document has been processed

## Usage

1. Add URLs to `links.txt` (one per line)
2. The scraper will:
   - Run daily at 00:00 UTC
   - Fetch and parse each URL
   - Store content in PostgreSQL database
   - Log results to the console and to a logfile

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

# Run the scraper
python scraper.py

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
- Worker dyno for scheduled scraping
- Environment variables for configuration
- URL: https://latvian-laws-06e89c613b8a.herokuapp.com/

To run tests on Heroku:

```bash
heroku run python test_db.py --app latvian-laws
```