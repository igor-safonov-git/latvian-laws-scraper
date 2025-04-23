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
        # Get a list of document IDs
        cursor.execute("""
            SELECT id, url FROM raw_docs
            ORDER BY fetched_at DESC
        """)
        docs = cursor.fetchall()
        
        for doc_id, url in docs:
            # Get the first 500 characters of the translated text
            cursor.execute("""
                SELECT 
                    SUBSTRING(raw_text, 1, 100) as raw_preview,
                    SUBSTRING(translated_text, 1, 500) as trans_preview,
                    LENGTH(raw_text) as raw_length,
                    LENGTH(translated_text) as trans_length,
                    fetched_at
                FROM raw_docs
                WHERE id = %s
            """, (doc_id,))
            
            row = cursor.fetchone()
            if row:
                raw_preview, trans_preview, raw_length, trans_length, fetched_at = row
                
                print(f"\nDocument ID: {doc_id[:8]}...")
                print(f"URL: {url}")
                print(f"Fetched at: {fetched_at}")
                print(f"Raw length: {raw_length}, Translated length: {trans_length}")
                print(f"Ratio: {trans_length/raw_length:.4f}")
                print("\nRaw preview:")
                print("-" * 50)
                print(raw_preview)
                print("-" * 50)
                print("\nTranslated preview:")
                print("-" * 50)
                print(trans_preview)
                print("-" * 50)
                
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()