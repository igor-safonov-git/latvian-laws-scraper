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
        # Get total document count
        cursor.execute("SELECT COUNT(*) FROM raw_docs")
        total_before = cursor.fetchone()[0]
        print(f"Total documents before cleanup: {total_before}")
        
        # Delete test documents by URL pattern
        cursor.execute("""
            DELETE FROM raw_docs 
            WHERE url LIKE 'https://example.com/%'
            OR url LIKE '%test%'
            RETURNING id, url
        """)
        
        deleted_docs = cursor.fetchall()
        print(f"Deleted {len(deleted_docs)} test documents:")
        for doc_id, url in deleted_docs:
            print(f"  - {doc_id[:8]}... : {url}")
        
        # Get document count after cleanup
        cursor.execute("SELECT COUNT(*) FROM raw_docs")
        total_after = cursor.fetchone()[0]
        
        print(f"Total documents after cleanup: {total_after}")
        print(f"Total removed: {total_before - total_after}")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()