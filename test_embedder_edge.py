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
        
        # Edge Case 1: Check for empty or null embeddings
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM docs WHERE embedding IS NULL;")
            empty_count = cursor.fetchone()[0]
            
            if empty_count > 0:
                print(f"❌ Found {empty_count} NULL embeddings")
            else:
                print("✅ No NULL embeddings found")
        
        # Edge Case 2: Check dimension consistency
        with conn.cursor() as cursor:
            # First get the dimension of the pgvector extension
            cursor.execute("SELECT COUNT(*) FROM docs;")
            doc_count = cursor.fetchone()[0]
            print(f"ℹ️ Total document count: {doc_count}")
            
            if doc_count > 0:
                # Get a sample embedding to check dimensions
                cursor.execute("SELECT embedding FROM docs LIMIT 1;")
                sample = cursor.fetchone()[0]
                dims = len(sample)
                print(f"ℹ️ Vector dimensions: {dims}")
            else:
                print("ℹ️ No documents found to check dimensions")
        
        # Edge Case 3: Check for documents with embeddings but no translated text
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
        
        # Edge Case 4: Check if any embeddings were generated for very long texts (near token limit)
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
        
        # Edge Case 5: Check for truncated text with [Truncated] marker
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE translated_text LIKE '%[Truncated due to token limit]%';
            """)
            truncated_count = cursor.fetchone()[0]
            
            if truncated_count > 0:
                print(f"ℹ️ Found {truncated_count} texts with truncation markers")
                # Look at embeddings for truncated texts
                cursor.execute("""
                    SELECT d.id 
                    FROM docs d
                    JOIN raw_docs r ON d.id = r.id
                    WHERE r.translated_text LIKE '%[Truncated due to token limit]%'
                    LIMIT 3;
                """)
                truncated_ids = [row[0] for row in cursor.fetchall()]
                if truncated_ids:
                    print(f"  - Truncated document IDs: {', '.join(truncated_ids)}")
            else:
                print("ℹ️ No texts with truncation markers found")
        
        # Edge Case 6: Check metadata integrity
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM docs
                WHERE metadata->>'url' IS NULL 
                   OR metadata->>'fetched_at' IS NULL
                   OR metadata->>'text_preview' IS NULL;
            """)
            invalid_meta_count = cursor.fetchone()[0]
            
            if invalid_meta_count > 0:
                print(f"❌ Found {invalid_meta_count} embeddings with incomplete metadata")
            else:
                print("✅ All embeddings have complete metadata")

        # Edge Case 7: Check if there are orphaned embeddings (no matching raw doc)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM docs d
                LEFT JOIN raw_docs r ON d.id = r.id
                WHERE r.id IS NULL;
            """)
            orphaned_count = cursor.fetchone()[0]
            
            if orphaned_count > 0:
                print(f"❌ Found {orphaned_count} orphaned embeddings (no matching raw document)")
            else:
                print("✅ No orphaned embeddings found")
                
        # Edge Case 8: Check translation to embedding coverage
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE translated_text IS NOT NULL AND processed = TRUE;
            """)
            translated_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM docs;")
            embedding_count = cursor.fetchone()[0]
            
            if translated_count > embedding_count:
                print(f"⚠️ Not all translated documents have embeddings: {embedding_count}/{translated_count}")
                
                # Check for translations without embeddings
                cursor.execute("""
                    SELECT r.id, r.url
                    FROM raw_docs r
                    LEFT JOIN docs d ON r.id = d.id
                    WHERE r.translated_text IS NOT NULL 
                    AND r.processed = TRUE
                    AND d.id IS NULL
                    LIMIT 5;
                """)
                missing = cursor.fetchall()
                if missing:
                    print("  Missing embeddings for these translated documents:")
                    for doc in missing:
                        print(f"  - {doc[0]}: {doc[1]}")
            elif translated_count < embedding_count:
                print(f"⚠️ There are more embeddings than translated documents: {embedding_count}/{translated_count}")
            else:
                print(f"✅ All {translated_count} translated documents have embeddings")

        print("\nAll embedder edge case tests completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_edge_cases()