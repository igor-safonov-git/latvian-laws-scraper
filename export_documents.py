#!/usr/bin/env python3
"""
Export all scraped documents from the database to individual text files.
These files will be placed in a directory accessible via the web app.

Usage:
    python export_documents.py [--output-dir DIR] [--limit N]

Options:
    --output-dir DIR    Directory to write files to (default: ./static/export)
    --limit N           Limit to N most recent documents (default: all documents)
"""
import os
import sys
import argparse
import logging
import asyncio
import psycopg2
import psycopg2.extras
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("export_documents")

def safe_filename(url: str, doc_id: str) -> str:
    """Generate a safe filename from a URL and document ID."""
    # Extract domain and path
    url = url.replace("https://", "").replace("http://", "")
    parts = url.split("/")
    domain = parts[0]
    
    # Use just the document ID if the domain is too long
    if len(domain) > 30:
        return f"{doc_id[:10]}.txt"
    
    # Create a more descriptive filename
    if len(parts) > 1:
        path = "_".join(parts[1:])
        if len(path) > 50:
            path = path[:50]
        return f"{domain}_{path}_{doc_id[:6]}.txt"
    else:
        return f"{domain}_{doc_id[:10]}.txt"

def export_documents(output_dir: str, limit: int = None) -> bool:
    """
    Export all documents from the database to text files.
    
    Args:
        output_dir: Directory to write files to
        limit: Limit to N most recent documents, or None for all
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load environment variables
    load_dotenv()
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return False
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Index file to list all exported documents
    index_path = output_path / "index.html"
    
    try:
        # Connect to the database
        conn = psycopg2.connect(database_url)
        conn.autocommit = True
        
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            # Get document count
            cursor.execute("SELECT COUNT(*) FROM raw_docs WHERE translated_text IS NOT NULL")
            total_docs = cursor.fetchone()[0]
            logger.info(f"Found {total_docs} documents with translations")
            
            # Get documents with limit
            limit_clause = f"LIMIT {limit}" if limit else ""
            cursor.execute(f"""
                SELECT id, url, fetched_at, translated_text 
                FROM raw_docs 
                WHERE translated_text IS NOT NULL
                ORDER BY fetched_at DESC
                {limit_clause}
            """)
            
            docs = cursor.fetchall()
            logger.info(f"Exporting {len(docs)} documents")
            
            # Write index file header
            with open(index_path, "w") as index_file:
                index_file.write("""<!DOCTYPE html>
<html>
<head>
    <title>Exported Latvian Law Documents</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Exported Latvian Law Documents</h1>
    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    <table>
        <tr>
            <th>Document</th>
            <th>URL</th>
            <th>Fetched At</th>
            <th>File Size</th>
        </tr>
""")
            
            # Export each document
            for doc in docs:
                doc_id = doc["id"]
                url = doc["url"]
                fetched_at = doc["fetched_at"]
                translated_text = doc["translated_text"]
                
                # Create a safe filename
                filename = safe_filename(url, doc_id)
                file_path = output_path / filename
                
                # Write document to file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"URL: {url}\n")
                    f.write(f"Fetched at: {fetched_at}\n")
                    f.write(f"Document ID: {doc_id}\n")
                    f.write("\n" + "="*80 + "\n\n")
                    f.write(translated_text)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                file_size_str = f"{file_size/1024:.1f} KB" if file_size > 1024 else f"{file_size} bytes"
                
                # Add to index
                with open(index_path, "a") as index_file:
                    index_file.write(f"""
        <tr>
            <td><a href="{filename}">{filename}</a></td>
            <td><a href="{url}" target="_blank">{url}</a></td>
            <td class="timestamp">{fetched_at}</td>
            <td>{file_size_str}</td>
        </tr>""")
                
                logger.info(f"Exported document: {filename} ({file_size_str})")
            
            # Close HTML file
            with open(index_path, "a") as index_file:
                index_file.write("""
    </table>
</body>
</html>""")
            
            logger.info(f"Export complete. Created index file: {index_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error exporting documents: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Run the document export script."""
    parser = argparse.ArgumentParser(description="Export documents from database to text files")
    # Default to /tmp/export on Heroku, or ./static/export locally
    default_dir = "/tmp/export" if "DYNO" in os.environ else "./static/export"
    parser.add_argument("--output-dir", default=default_dir, 
                        help=f"Directory to write files to (default: {default_dir})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N most recent documents (default: all)")
    
    args = parser.parse_args()
    
    success = export_documents(args.output_dir, args.limit)
    
    if success:
        logger.info("Export completed successfully")
    else:
        logger.error("Export failed")
        sys.exit(1)

if __name__ == "__main__":
    main()