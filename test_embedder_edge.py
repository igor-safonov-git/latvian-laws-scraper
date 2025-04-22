#!/usr/bin/env python3
import os
import sys
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

def test_edge_cases():
    """Test edge cases for embedder functionality."""
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL environment variable not set")
        sys.exit(1)

    try:
        # Connect to the database
        print("Connecting to database...")
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        # Edge Case 1: Check for embeddings with inconsistent dimensions
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM docs WHERE array_length(embedding, 1) <> 6280;")
            inconsistent_count = cursor.fetchone()[0]
            
            if inconsistent_count > 0:
                print(f"❌ Found {inconsistent_count} embeddings with inconsistent dimensions")
            else:
                print("✅ All embeddings have consistent dimensions (6280)")
        
        # Edge Case 2: Check for empty or null embeddings
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM docs WHERE embedding IS NULL OR array_length(embedding, 1) = 0;")
            empty_count = cursor.fetchone()[0]
            
            if empty_count > 0:
                print(f"❌ Found {empty_count} empty or NULL embeddings")
            else:
                print("✅ No empty or NULL embeddings found")
        
        # Edge Case 3: Check for placeholder embeddings (random vectors with small magnitudes)
        # Placeholder embeddings would have much smaller magnitudes (between -0.1 and 0.1)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM docs
                WHERE embedding <@ cube(array_fill(0.2, ARRAY[6280]), 0.2);
            """)
            placeholder_count = cursor.fetchone()[0]
            
            if placeholder_count > 0:
                print(f"ℹ️ Found {placeholder_count} possible placeholder embeddings (small magnitude)")
            else:
                print("✅ No placeholder embeddings detected")
        
        # Edge Case 4: Check for duplicated embeddings
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT e1.id, e2.id
                FROM docs e1, docs e2
                WHERE e1.id <> e2.id AND e1.embedding = e2.embedding
                LIMIT 1;
            """)
            duplicate = cursor.fetchone()
            
            if duplicate:
                print(f"❌ Found duplicate embeddings between documents {duplicate[0]} and {duplicate[1]}")
            else:
                print("✅ No duplicate embeddings found")
        
        # Edge Case 5: Check for documents with embeddings but no translated text
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM docs d 
                LEFT JOIN raw_docs r ON d.id = r.id
                WHERE r.translated_text IS NULL OR length(r.translated_text) = 0;
            """)
            invalid_count = cursor.fetchone()[0]
            
            if invalid_count > 0:
                print(f"❌ Found {invalid_count} embeddings for documents with no translated text")
            else:
                print("✅ All embeddings correspond to documents with valid translated text")
        
        # Edge Case 6: Check if any embeddings were generated for very long texts (near token limit)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM docs d 
                JOIN raw_docs r ON d.id = r.id
                WHERE length(r.translated_text) > 32000;
            """)
            long_count = cursor.fetchone()[0]
            
            if long_count > 0:
                print(f"ℹ️ Found {long_count} embeddings for very long texts (>32K chars)")
                # Check if these were truncated
                cursor.execute("""
                    SELECT d.id, length(r.translated_text) as text_len
                    FROM docs d 
                    JOIN raw_docs r ON d.id = r.id
                    WHERE length(r.translated_text) > 32000
                    LIMIT 3;
                """)
                for row in cursor.fetchall():
                    print(f"  - Document {row[0]}: {row[1]} chars")
            else:
                print("ℹ️ No embeddings for very long texts (>32K chars)")

        print("\nAll embedder edge case tests completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_edge_cases()