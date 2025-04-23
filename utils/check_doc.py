#!/usr/bin/env python3
import os
import psycopg2
import psycopg2.extras
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
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        # Check the test document status
        cursor.execute("""
            SELECT id, url, processed, 
                   LENGTH(raw_text) as raw_length,
                   LENGTH(translated_text) as translated_length,
                   SUBSTRING(raw_text, 1, 100) as raw_preview,
                   SUBSTRING(translated_text, 1, 100) as translated_preview
            FROM raw_docs
            ORDER BY processed ASC, fetched_at DESC
            LIMIT 5
        """)
        
        rows = cursor.fetchall()
        
        # Print the results
        print(f"Recent documents in the database:")
        print(f"{'ID':<10} | {'Processed':<10} | {'Raw Length':<12} | {'Trans Length':<12} | {'URL':<50} | {'Preview':<50}")
        print("-" * 150)
        
        for row in rows:
            print(f"{row['id'][:8]:<10} | {str(row['processed']):<10} | {row['raw_length']:<12} | "
                  f"{row['translated_length'] if row['translated_length'] else 0:<12} | {row['url'][:50]:<50} | "
                  f"{row['raw_preview'][:50] if row['raw_preview'] else 'None'}")
            
            if row['processed'] is False:
                print(f"Document {row['id']} is not processed")
                print(f"Raw text: {row['raw_preview']}")
                print(f"Translated text: {row['translated_preview'] if row['translated_preview'] else 'None'}")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()