#!/usr/bin/env python3
import os
import sys
import json
import psycopg2
import random
from dotenv import load_dotenv

def upgrade_embedder():
    """Upgrade the embedder from small to large model (512 to 3072 dimensions)."""
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
        
        # 1. Check table definitions
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT 'docs'::regclass;
            """)
            if cursor.fetchone():
                print("✅ Docs table exists")
            else:
                print("❌ Docs table does not exist - nothing to upgrade")
                return
        
        # 2. Check current vector dimensions
        with conn.cursor() as cursor:
            try:
                cursor.execute("""
                    SELECT array_length(embedding, 1) FROM docs LIMIT 1;
                """)
                current_dims = cursor.fetchone()
                
                if current_dims and current_dims[0]:
                    print(f"ℹ️ Current vector dimensions: {current_dims[0]}")
                    if current_dims[0] == 3072:
                        print("✅ Already using 3072 dimensions - no upgrade needed")
                        return
                else:
                    print("ℹ️ No vectors in docs table")
            except Exception as e:
                print(f"Error checking dimensions: {str(e)}")
                print("Continuing with upgrade...")
        
        # 3. Drop and recreate the table
        with conn.cursor() as cursor:
            # First backup metadata
            cursor.execute("""
                SELECT id, metadata FROM docs;
            """)
            metadata_backup = {row[0]: row[1] for row in cursor.fetchall()}
            print(f"ℹ️ Backed up metadata for {len(metadata_backup)} documents")
            
            # Drop the table
            cursor.execute("""
                DROP TABLE IF EXISTS docs;
            """)
            print("✅ Dropped docs table")
            
            # Recreate with larger vector dimensions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS docs (
                    id TEXT PRIMARY KEY,
                    metadata JSONB,
                    embedding VECTOR(3072)
                );
            """)
            print("✅ Recreated docs table with 3072 dimensions")
            
            # If we had data, create placeholder embeddings
            if metadata_backup:
                print(f"Recreating embeddings for {len(metadata_backup)} documents")
                dimensions = 3072
                
                for doc_id, metadata in metadata_backup.items():
                    # Create a placeholder embedding of the right size
                    placeholder = [random.uniform(-0.1, 0.1) for _ in range(dimensions)]
                    
                    # If metadata is a string, parse it
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {"original_metadata": metadata}
                    
                    # Update metadata to show it's a placeholder
                    if isinstance(metadata, dict):
                        metadata["is_placeholder"] = True
                        metadata["placeholder_reason"] = "Upgraded from small to large model"
                    else:
                        metadata = {
                            "original_metadata": str(metadata),
                            "is_placeholder": True,
                            "placeholder_reason": "Upgraded from small to large model"
                        }
                    
                    # Insert placeholder
                    cursor.execute("""
                        INSERT INTO docs(id, metadata, embedding)
                        VALUES(%s, %s, %s::vector)
                    """, (doc_id, json.dumps(metadata), placeholder))
                    print(f"  ✅ Created placeholder embedding for {doc_id}")
        
        print("\n✅ Upgrade completed successfully!")
        print("The embedding model has been upgraded from text-embedding-3-small (512 dimensions)")
        print("to text-embedding-3-large (3072 dimensions).")
        print("Run the embedder service to generate real embeddings with the new model.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    upgrade_embedder()