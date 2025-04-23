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
        # Check raw_docs table
        cursor.execute("""
            SELECT COUNT(*) FROM raw_docs
        """)
        total_docs = cursor.fetchone()[0]
        print(f"Total documents in raw_docs: {total_docs}")
        
        # Check which documents need processing
        cursor.execute("""
            SELECT COUNT(*) FROM raw_docs
            WHERE translated_text IS NOT NULL
        """)
        translated_docs = cursor.fetchone()[0]
        print(f"Documents with translations: {translated_docs}")
        
        # Check docs table (where embeddings go)
        cursor.execute("""
            SELECT COUNT(*) FROM pg_tables
            WHERE tablename = 'docs'
        """)
        docs_table_exists = cursor.fetchone()[0] > 0
        
        if docs_table_exists:
            cursor.execute("""
                SELECT COUNT(*) FROM docs
            """)
            embedding_docs = cursor.fetchone()[0]
            print(f"Documents with embeddings in docs table: {embedding_docs}")
            
            # Check documents with non-null embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM docs
                WHERE embedding IS NOT NULL
            """)
            non_null_embeddings = cursor.fetchone()[0]
            print(f"Documents with non-null embeddings: {non_null_embeddings}")
        else:
            print("The 'docs' table does not exist")
        
        # Print document IDs and URLs
        cursor.execute("""
            SELECT id, url, LENGTH(raw_text) as raw_length, 
                   LENGTH(translated_text) as translated_length
            FROM raw_docs
            ORDER BY fetched_at DESC
        """)
        
        print("\nDocuments in database:")
        print(f"{'ID':<10} | {'URL':<50} | {'Raw Length':<12} | {'Trans Length':<12}")
        print("-" * 100)
        
        for row in cursor.fetchall():
            doc_id, url, raw_length, translated_length = row
            print(f"{doc_id[:8]}... | {url[:50]:<50} | {raw_length:<12} | {translated_length if translated_length else 0:<12}")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()