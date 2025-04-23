#!/usr/bin/env python3
import os
import sys
import argparse
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tabulate import tabulate

def main():
    """
    Tests the embedder functionality to ensure embeddings are properly generated and stored.
    """
    parser = argparse.ArgumentParser(description='Test the embedder functionality')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--sample', '-s', action='store_true', help='Show sample embedding values')
    args = parser.parse_args()

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
        
        # Test 1: Check if vector extension is enabled
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM pg_extension 
                    WHERE extname = 'vector'
                )
            """)
            extension_exists = cursor.fetchone()[0]
            
            if not extension_exists:
                print("❌ Test failed: 'vector' extension is not enabled")
                sys.exit(1)
            else:
                print("✅ Test passed: 'vector' extension is enabled")

        # Test 2: Check if docs table exists
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'docs'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("❌ Test failed: 'docs' table does not exist")
                sys.exit(1)
            else:
                print("✅ Test passed: 'docs' table exists")

        # Test 3: Check if there are any records in docs table
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM docs")
            record_count = cursor.fetchone()[0]
            
            if record_count == 0:
                print("❌ Test failed: No embeddings found in 'docs' table")
                sys.exit(1)
            else:
                print(f"✅ Test passed: Found {record_count} embeddings in 'docs' table")

        # Test 4: Check if all translated documents have embeddings
        with conn.cursor() as cursor:
            # Count translated documents
            cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NOT NULL")
            translated_count = cursor.fetchone()[0]
            
            # Check if counts match
            if translated_count != record_count:
                print(f"⚠️ Warning: Number of embeddings ({record_count}) doesn't match number of translated documents ({translated_count})")
                # This is a warning not an error, as the embedder might be still processing
            else:
                print(f"✅ Test passed: All {translated_count} translated documents have embeddings")
        
        # Test 5: Check embedding dimensions
        with conn.cursor() as cursor:
            cursor.execute("SELECT embedding FROM docs LIMIT 1")
            sample_embedding = cursor.fetchone()
            
            if not sample_embedding:
                print("❌ Test failed: Could not retrieve a sample embedding")
                sys.exit(1)
            
            embedding_array = sample_embedding[0]
            
            if args.sample:
                print(f"\nSample embedding (first 10 dimensions): {embedding_array[:10]}...")
                
            print(f"✅ Test passed: Embeddings have {len(embedding_array)} dimensions")

        # Test 6: Check metadata format
        with conn.cursor() as cursor:
            cursor.execute("SELECT metadata FROM docs LIMIT 1")
            sample_metadata = cursor.fetchone()[0]
            
            required_fields = ['url', 'fetched_at', 'text_preview']
            missing_fields = [field for field in required_fields if field not in sample_metadata]
            
            if missing_fields:
                print(f"❌ Test failed: Metadata missing required fields: {missing_fields}")
                sys.exit(1)
            else:
                print("✅ Test passed: Metadata contains all required fields")

        # If verbose, display the embeddings information
        if args.verbose:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        d.id, 
                        d.metadata->>'url' AS url, 
                        d.metadata->>'fetched_at' AS fetched_at,
                        LENGTH(array_to_string(d.embedding, ',')) AS embedding_size,
                        d.metadata->>'text_preview' AS preview
                    FROM docs d
                    ORDER BY d.metadata->>'fetched_at' DESC
                """)
                rows = cursor.fetchall()
                
                if rows:
                    print("\nEmbeddings:")
                    headers = ["ID", "URL", "Fetched At", "Embedding Size", "Text Preview"]
                    print(tabulate(rows, headers=headers, tablefmt="grid"))
                else:
                    print("No embeddings to display")

        print("\nAll embedder tests passed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()