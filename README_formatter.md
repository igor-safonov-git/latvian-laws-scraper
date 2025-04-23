# Latvian Text Formatter

This tool formats Latvian legal texts into structured formats (bullet points, sections) using the EuroLLM-9B-Instruct multilingual model.

## Features

- Formats raw Latvian texts into bullet points or sections
- Memory-efficient processing for large documents
- Database integration for batch processing
- Standalone testing script for file-based formatting

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Database Batch Processing

The main script (`latvian_formatter.py`) processes documents directly from the database:

```
python latvian_formatter.py --format-type bullet_points --batch-size 10 --memory-limit 4000
```

Parameters:
- `--format-type`: Format style (bullet_points, sections, general)
- `--batch-size`: Number of documents to process in one batch
- `--memory-limit`: Memory limit in MB

### Standalone File Processing

For testing or one-off formatting, use the test script:

```
python test_formatter.py --input input.txt --output formatted.txt --format-type sections
```

Parameters:
- `--input`, `-i`: Input text file in Latvian
- `--output`, `-o`: Output file for formatted text
- `--format-type`, `-t`: Type of formatting to apply

## Environment Variables

Create a `.env` file with:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
BATCH_DELAY_MS=100
MEMORY_LIMIT_MB=4000
FORMATTER_MODEL_ID=utter-project/EuroLLM-9B-Instruct
```

## Database Schema

The script expects a database with a table structure:

```sql
CREATE TABLE raw_docs (
    id TEXT PRIMARY KEY,
    url TEXT,
    fetched_at TIMESTAMP,
    raw_text TEXT,
    structured_format TEXT,
    formatted_at TIMESTAMP
);
```

## Notes

- EuroLLM-9B-Instruct is optimized for Latvian and other European languages
- For large documents, the script truncates text to 6000 characters by default
- GPU acceleration is used when available, falling back to CPU 