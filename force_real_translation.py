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
        # Find documents with placeholder translations
        cursor.execute("""
            SELECT id, url 
            FROM raw_docs
            WHERE translated_text LIKE '%PLACEHOLDER TRANSLATION%'
               OR LENGTH(translated_text) < 1000
        """)
        
        placeholder_docs = cursor.fetchall()
        
        if not placeholder_docs:
            print("No documents with placeholder translations found.")
        else:
            print(f"Found {len(placeholder_docs)} documents with placeholder translations:")
            
            for doc_id, url in placeholder_docs:
                print(f"  - {doc_id[:8]}... : {url}")
                
                # Clear the translated_text field to force retranslation
                cursor.execute("""
                    UPDATE raw_docs
                    SET processed = FALSE, translated_text = NULL
                    WHERE id = %s
                """, (doc_id,))
                
                print(f"    Cleared translation and marked for retranslation")
            
            print("\nAll documents have been marked for retranslation.")
            print("Run the translator service to process them.")
        
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    conn.close()