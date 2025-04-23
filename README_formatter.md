# Latvian Text Formatter

This tool formats Latvian legal texts into structured formats (bullet points, sections) using the EuroLLM-9B-Instruct multilingual model via a Hugging Face endpoint.

## Features

- Formats raw Latvian texts into bullet points or sections
- Memory-efficient processing for large documents
- Database integration for batch processing
- Standalone testing script for file-based formatting
- Uses Hugging Face API endpoint instead of loading model locally

## Requirements

Install dependencies:
```
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:

```
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
BATCH_DELAY_MS=100
MEMORY_LIMIT_MB=4000
HF_ENDPOINT=https://uyba45g29g72pbos.us-east4.gcp.endpoints.huggingface.cloud/v1/
HF_API_KEY=your_huggingface_api_key
```

You must obtain a valid Hugging Face API key and set it in the `HF_API_KEY` environment variable.

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
- Using the Hugging Face endpoint reduces memory requirements and simplifies deployment 