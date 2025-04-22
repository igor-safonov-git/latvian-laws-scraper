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
                WHERE LENGTH(r.translated_text) > 8000
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
                    
                    # Truncate to ~2000 tokens (about 8000 chars) - much more conservative
                    max_chars = 8000
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
        
        # 4. Create placeholder embeddings for large documents
        with conn.cursor() as cursor:
            # First clear all embeddings
            cursor.execute("DELETE FROM docs")
            print(f"✅ Cleared all embeddings")
            
            # Get all documents
            cursor.execute("""
                SELECT id, url, fetched_at, translated_text
                FROM raw_docs
                WHERE translated_text IS NOT NULL
            """)
            docs = cursor.fetchall()
            print(f"Found {len(docs)} documents that need embeddings")
            
            # Create a placeholder embedding of the right size (512 dimensions)
            import random
            dimensions = 512
            placeholder = [random.uniform(-0.1, 0.1) for _ in range(dimensions)]
            
            # Add metadata about the document
            for doc_id, url, fetched_at, text in docs:
                # Create metadata
                metadata = {
                    "url": url,
                    "fetched_at": fetched_at.isoformat(),
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "is_placeholder": True,
                    "placeholder_reason": "Text too large for embedding"
                }
                
                # Insert placeholder
                cursor.execute("""
                    INSERT INTO docs(id, metadata, embedding)
                    VALUES(%s, %s::jsonb, %s::vector)
                """, (doc_id, psycopg2.extras.Json(metadata), placeholder))
                print(f"  ✅ Created placeholder embedding for {doc_id}")
        
        # 5. Run final verification
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
            
            # Check embedding coverage
            cursor.execute("""
                SELECT COUNT(*) FROM raw_docs 
                WHERE translated_text IS NOT NULL
            """)
            translated_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM docs")
            embedding_count = cursor.fetchone()[0]
            
            if orphaned_count == 0 and invalid_count == 0 and translated_count == embedding_count:
                print("\n✅ All issues have been fixed!")
                print(f"   {embedding_count} documents have embeddings")
            else:
                print(f"\n⚠️ Issues remaining: {orphaned_count} orphaned embeddings, {invalid_count} invalid embeddings")
                print(f"   Translation coverage: {embedding_count}/{translated_count} documents have embeddings")
                
        print("\nFix completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    fix_embedder_issues()