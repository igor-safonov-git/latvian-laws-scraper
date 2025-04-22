#!/usr/bin/env python3
import os
import sys
import psycopg2
from dotenv import load_dotenv

def fix_embedder_issues():
    """Fix issues discovered by test_embedder_edge.py"""
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
        
        # 1. Fix truncation marker issue for very long documents
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT r.id, r.url, LENGTH(r.translated_text) as text_len
                FROM raw_docs r
                WHERE LENGTH(r.translated_text) > 32000
                ORDER BY text_len DESC
            """)
            long_docs = cursor.fetchall()
            
            if long_docs:
                print(f"Found {len(long_docs)} very long documents that need truncation:")
                for doc_id, url, text_len in long_docs:
                    print(f"  - {doc_id} ({url}): {text_len} chars")
                    
                    # Get the current text
                    cursor.execute("SELECT translated_text FROM raw_docs WHERE id = %s", (doc_id,))
                    text = cursor.fetchone()[0]
                    
                    # Truncate to ~8000 tokens (about 32000 chars)
                    max_chars = 32000
                    if len(text) > max_chars:
                        truncated = text[:max_chars] + "\n...[Truncated due to token limit]"
                        
                        # Update with truncated text
                        cursor.execute("""
                            UPDATE raw_docs
                            SET translated_text = %s
                            WHERE id = %s
                        """, (truncated, doc_id))
                        print(f"    ✅ Truncated text from {len(text)} to {len(truncated)} chars")
            else:
                print("No very long documents found that need truncation")
        
        # 2. Identify orphaned embeddings (no matching raw document)
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT d.id FROM docs d
                LEFT JOIN raw_docs r ON d.id = r.id
                WHERE r.id IS NULL;
            """)
            orphaned_ids = [row[0] for row in cursor.fetchall()]
            
            if orphaned_ids:
                print(f"Found {len(orphaned_ids)} orphaned embeddings to remove:")
                for doc_id in orphaned_ids:
                    print(f"  - {doc_id}")
                
                # Remove orphaned embeddings
                cursor.execute("""
                    DELETE FROM docs
                    WHERE id IN (
                        SELECT d.id FROM docs d
                        LEFT JOIN raw_docs r ON d.id = r.id
                        WHERE r.id IS NULL
                    )
                """)
                print(f"✅ Removed {cursor.rowcount} orphaned embeddings")
            else:
                print("No orphaned embeddings found")
        
        # 3. Fix embeddings for documents with no translated text
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT d.id FROM docs d 
                JOIN raw_docs r ON d.id = r.id
                WHERE r.translated_text IS NULL OR length(r.translated_text) = 0;
            """)
            invalid_ids = [row[0] for row in cursor.fetchall()]
            
            if invalid_ids:
                print(f"Found {len(invalid_ids)} embeddings for documents without translated text:")
                for doc_id in invalid_ids:
                    print(f"  - {doc_id}")
                
                # Remove these invalid embeddings
                cursor.execute("""
                    DELETE FROM docs
                    WHERE id IN (
                        SELECT d.id FROM docs d 
                        JOIN raw_docs r ON d.id = r.id
                        WHERE r.translated_text IS NULL OR length(r.translated_text) = 0
                    )
                """)
                print(f"✅ Removed {cursor.rowcount} invalid embeddings")
                
                # Reset processed flag for these documents
                if invalid_ids:
                    cursor.execute("""
                        UPDATE raw_docs
                        SET processed = FALSE
                        WHERE id IN %s
                    """, (tuple(invalid_ids),))
                    print(f"✅ Reset processed flag for {cursor.rowcount} documents")
            else:
                print("No embeddings for documents without translated text found")
        
        # 4. Run final verification
        with conn.cursor() as cursor:
            # Check for orphaned embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM docs d
                LEFT JOIN raw_docs r ON d.id = r.id
                WHERE r.id IS NULL;
            """)
            orphaned_count = cursor.fetchone()[0]
            
            # Check for invalid embeddings
            cursor.execute("""
                SELECT COUNT(*) FROM docs d 
                JOIN raw_docs r ON d.id = r.id
                WHERE r.translated_text IS NULL OR length(r.translated_text) = 0;
            """)
            invalid_count = cursor.fetchone()[0]
            
            if orphaned_count == 0 and invalid_count == 0:
                print("\n✅ All issues have been fixed!")
            else:
                print(f"\n⚠️ Issues remaining: {orphaned_count} orphaned embeddings, {invalid_count} invalid embeddings")
                
        print("\nFix completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    fix_embedder_issues()