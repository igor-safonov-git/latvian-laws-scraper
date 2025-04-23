#!/usr/bin/env python3
import os
import psycopg2
from datetime import datetime
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Generate a unique ID for the test document
test_text = "Šī ir jauna tulkošanas pārbaude. DeepL API atslēga darbojas labi."
doc_id = hashlib.sha256(test_text.encode()).hexdigest()
url = "https://example.com/new-test-document"
fetched_at = datetime.now().isoformat()

# Connect to the database
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("Error: DATABASE_URL not set")
    exit(1)

print(f"Connecting to database...")
conn = psycopg2.connect(database_url)
conn.autocommit = True

try:
    with conn.cursor() as cursor:
        # Insert test document
        cursor.execute("""
            INSERT INTO raw_docs (id, url, raw_text, fetched_at, processed) 
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT (id) DO UPDATE SET 
                raw_text = EXCLUDED.raw_text,
                processed = FALSE
        """, (doc_id, url, test_text, fetched_at))
        
        print(f"Added new test document with ID: {doc_id}")
        print(f"Text: {test_text}")
        
        # Check all documents
        cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE processed = FALSE")
        unprocessed_count = cursor.fetchone()[0]
        print(f"There are {unprocessed_count} unprocessed documents in the database")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()