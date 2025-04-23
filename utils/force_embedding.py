#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        # Clear the docs table to force re-embedding
        cursor.execute("""
            TRUNCATE TABLE docs
        """)
        
        print("Cleared docs table to force re-embedding of all documents")
        
        # Check translation status
        cursor.execute("""
            SELECT id, url, 
                   LENGTH(raw_text) as raw_length,
                   LENGTH(translated_text) as trans_length
            FROM raw_docs
            WHERE translated_text IS NOT NULL
        """)
        
        translated_docs = cursor.fetchall()
        print(f"\nFound {len(translated_docs)} translated documents:")
        
        for doc_id, url, raw_length, trans_length in translated_docs:
            # Calculate the compression ratio
            ratio = trans_length / raw_length if raw_length > 0 else 0
            
            print(f"  - {doc_id[:8]}... : {url}")
            print(f"    Raw: {raw_length} chars, Translated: {trans_length} chars")
            print(f"    Ratio: {ratio:.4f}")
        
        print("\nRun the embedder service to generate embeddings.")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()