#!/usr/bin/env python3
"""
Latvian text formatter using EuroLLM-9B-Instruct model via Hugging Face endpoint.
This script transforms raw Latvian texts into structured formats like bullet points or sections.
"""
import os
import gc
import json
import psutil
import logging
import time
import psycopg2
import psycopg2.extras
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("latvian_formatter")

# Ensure logs directory exists
if not os.path.exists("./logs"):
    os.makedirs("./logs")

# Setup file handler for logs
file_handler = logging.FileHandler("./logs/latvian_formatter.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Configuration from environment
DATABASE_URL = os.getenv("DATABASE_URL")
BATCH_DELAY_MS = int(os.getenv("BATCH_DELAY_MS", "100"))
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "4000"))
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://uyba45g29g72pbos.us-east4.gcp.endpoints.huggingface.cloud/v1/")
HF_API_KEY = os.getenv("HF_API_KEY")

class MemoryGuard:
    """Monitors memory usage and prevents exceeding limits."""
    
    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    
    @staticmethod
    def check_memory(threshold_mb: float = MEMORY_LIMIT_MB) -> bool:
        """
        Check if memory usage is below threshold.
        Returns True if memory usage is acceptable, False if it's too high.
        """
        usage = MemoryGuard.get_memory_usage_mb()
        if usage > threshold_mb:
            logger.warning(f"Memory usage too high: {usage:.2f}MB > {threshold_mb}MB threshold")
            # Trigger garbage collection
            gc.collect()
            
            # Check again after GC
            usage = MemoryGuard.get_memory_usage_mb()
            if usage > threshold_mb:
                logger.error(f"Memory usage still high after GC: {usage:.2f}MB")
                return False
        return True

class DatabaseConnector:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str):
        """Initialize database connector with connection string."""
        self.database_url = database_url
        self.conn = None
        self.cursor = None
    
    def connect(self) -> bool:
        """Create connection to database."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def fetch_unprocessed_latvian_texts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch unprocessed Latvian texts from the database."""
        if not MemoryGuard.check_memory():
            logger.error("Memory usage too high to fetch texts")
            return []
        
        try:
            self.cursor.execute("""
                SELECT id, url, fetched_at, raw_text
                FROM raw_docs
                WHERE raw_text IS NOT NULL 
                AND structured_format IS NULL
                LIMIT %s
            """, (limit,))
            
            results = []
            for row in self.cursor:
                results.append({
                    "id": row["id"],
                    "url": row["url"],
                    "fetched_at": row["fetched_at"],
                    "raw_text": row["raw_text"]
                })
            
            logger.info(f"Fetched {len(results)} unprocessed Latvian texts")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch Latvian texts: {str(e)}")
            return []
    
    def update_structured_format(self, doc_id: str, structured_text: str) -> bool:
        """Update the structured format for a document."""
        try:
            self.cursor.execute("""
                UPDATE raw_docs
                SET structured_format = %s,
                    formatted_at = NOW()
                WHERE id = %s
            """, (structured_text, doc_id))
            
            return True
        except Exception as e:
            logger.error(f"Failed to update structured format for {doc_id}: {str(e)}")
            return False

class LatvianFormatter:
    """Formats Latvian texts into structured formats using EuroLLM model via HF endpoint."""
    
    def __init__(self):
        """Initialize the formatter service."""
        self.db = DatabaseConnector(DATABASE_URL)
        self.client = None
        
    def connect_api(self) -> bool:
        """Connect to the Hugging Face API endpoint."""
        try:
            if not HF_API_KEY:
                logger.error("HF_API_KEY not found in environment variables")
                return False
                
            self.client = OpenAI(
                base_url=HF_ENDPOINT,
                api_key=HF_API_KEY
            )
            
            logger.info(f"Connected to Hugging Face endpoint at {HF_ENDPOINT}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Hugging Face API: {str(e)}")
            return False
    
    def format_text(self, latvian_text: str, format_type: str = "bullet_points") -> str:
        """
        Format Latvian text into structured format using EuroLLM via HF endpoint.
        For large texts, processes them in chunks and combines the results.
        
        Args:
            latvian_text: Raw Latvian text to format
            format_type: Type of formatting to apply (bullet_points, sections, etc.)
            
        Returns:
            Formatted text
        """
        if not MemoryGuard.check_memory():
            logger.error("Memory usage too high to format text")
            return ""
        
        # Maximum chunk size that can be safely processed
        MAX_CHUNK_SIZE = 6000
        
        # If text is small enough, process it directly
        if len(latvian_text) <= MAX_CHUNK_SIZE:
            return self._format_chunk(latvian_text, format_type)
        
        # For large texts, split into chunks and format each chunk
        logger.info(f"Text is too large ({len(latvian_text)} chars), processing in chunks")
        
        # Split the text into logical chunks (paragraphs or sections)
        chunks = self._split_into_logical_chunks(latvian_text, MAX_CHUNK_SIZE)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Format each chunk
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Formatting chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Add context to non-first chunks to maintain continuity
            if i > 0:
                chunk_instruction = "Turpini formatēt šo juridisko teksta daļu. Saglabā iepriekšējo formatējuma stilu."
                if format_type == "bullet_points":
                    chunk_instruction += " Turpini ar aizzīmētiem punktiem."
                elif format_type == "sections":
                    chunk_instruction += " Turpini ar sekcijām un apakšsekcijām."
            else:
                # Use regular instruction for first chunk
                if format_type == "bullet_points":
                    chunk_instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
                elif format_type == "sections":
                    chunk_instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
                else:
                    chunk_instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            
            # Format this chunk
            formatted_chunk = self._format_chunk_with_instruction(chunk, chunk_instruction)
            if formatted_chunk:
                formatted_chunks.append(formatted_chunk)
            
            # Apply delay between chunks to avoid rate limiting
            delay_time = max(BATCH_DELAY_MS / 1000, 0.5)
            logger.info(f"Delaying {delay_time:.2f}s before next chunk")
            time.sleep(delay_time)
        
        # Combine the formatted chunks
        if formatted_chunks:
            combined_text = "\n\n".join(formatted_chunks)
            logger.info(f"Successfully formatted entire text in chunks ({len(combined_text)} chars)")
            return combined_text
        else:
            logger.error("Failed to format any chunks")
            return ""
    
    def _split_into_logical_chunks(self, text: str, max_size: int) -> List[str]:
        """
        Split text into logical chunks respecting paragraph or section boundaries.
        
        Args:
            text: Text to split
            max_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            para_size = len(paragraph)
            
            # If a single paragraph is too large, split it by sentences
            if para_size > max_size:
                # Split by sentence endings (period, question mark, etc. followed by space)
                sentences = [s.strip() + "." for s in paragraph.replace(". ", ".|").split("|") if s.strip()]
                
                for sentence in sentences:
                    if current_size + len(sentence) + 1 <= max_size:
                        current_chunk.append(sentence)
                        current_size += len(sentence) + 1  # +1 for newline
                    else:
                        # Finish current chunk and start a new one
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
            
            # If paragraph fits in the current chunk
            elif current_size + para_size + 2 <= max_size:  # +2 for double newline
                current_chunk.append(paragraph)
                current_size += para_size + 2
            else:
                # Finish current chunk and start a new one
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _format_chunk_with_instruction(self, chunk_text: str, instruction: str) -> str:
        """Format a chunk with a specific instruction."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Tu esi EuroLLM — AI asistents, kas specializējas Eiropas valodās, īpaši latviešu valodā. Tu palīdzi pārveidot tekstus skaidrā, strukturētā formātā."
                },
                {
                    "role": "user", 
                    "content": f"{instruction}\n\nTEKSTS:\n{chunk_text}"
                }
            ]
            
            # Call the API
            response = self.client.chat.completions.create(
                model="tgi",
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9
            )
            
            # Extract the response
            formatted_text = response.choices[0].message.content
            return formatted_text
            
        except Exception as e:
            logger.error(f"Error formatting chunk: {str(e)}")
            return ""
            
    def _format_chunk(self, latvian_text: str, format_type: str) -> str:
        """Format a single chunk with the standard instructions."""
        try:
            # Prepare prompt based on format type
            if format_type == "bullet_points":
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            elif format_type == "sections":
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            else:
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            
            return self._format_chunk_with_instruction(latvian_text, instruction)
            
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return ""
    
    def process_batch(self, batch_size: int = 10, format_type: str = "bullet_points") -> None:
        """Process a batch of Latvian texts."""
        logger.info(f"Starting batch processing with format type: {format_type}")
        
        # Connect to database
        if not self.db.connect():
            logger.error("Failed to connect to database, aborting batch")
            return
        
        # Connect to API
        if not self.connect_api():
            logger.error("Failed to connect to Hugging Face API, aborting batch")
            self.db.disconnect()
            return
        
        try:
            # Fetch unprocessed texts
            texts = self.db.fetch_unprocessed_latvian_texts(limit=batch_size)
            
            if not texts:
                logger.info("No unprocessed texts found")
                return
                
            logger.info(f"Processing {len(texts)} texts")
            
            # Process each text
            for doc in texts:
                doc_id = doc["id"]
                raw_text = doc["raw_text"]
                
                logger.info(f"Formatting document {doc_id} ({len(raw_text)} chars)")
                
                # Format the text
                formatted_text = self.format_text(raw_text, format_type)
                
                if formatted_text:
                    # Update in database
                    success = self.db.update_structured_format(doc_id, formatted_text)
                    
                    if success:
                        logger.info(f"Successfully updated document {doc_id}")
                    else:
                        logger.warning(f"Failed to update document {doc_id}")
                
                # Apply delay between documents
                delay_time = max(BATCH_DELAY_MS / 1000, 0.1)
                logger.info(f"Delaying {delay_time:.2f}s before next document")
                
                # Sleep between documents
                time.sleep(delay_time)
                
                # Help garbage collection
                raw_text = None
                formatted_text = None
                gc.collect()
            
            logger.info(f"Batch complete: processed {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
        
        finally:
            # Close database connection
            self.db.disconnect()

def main():
    """Main entry point for the script."""
    global MEMORY_LIMIT_MB
    
    parser = argparse.ArgumentParser(description="Format Latvian texts using EuroLLM")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process in one batch")
    parser.add_argument("--format-type", type=str, default="bullet_points", 
                        choices=["bullet_points", "sections", "general"],
                        help="Type of formatting to apply")
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_MB, 
                        help=f"Memory limit in MB (default: {MEMORY_LIMIT_MB})")
    
    args = parser.parse_args()
    
    # Apply command line arguments
    MEMORY_LIMIT_MB = args.memory_limit
    
    logger.info(f"Using memory limit: {MEMORY_LIMIT_MB}MB, format type: {args.format_type}, batch size: {args.batch_size}")
    
    # Run the formatter
    formatter = LatvianFormatter()
    formatter.process_batch(batch_size=args.batch_size, format_type=args.format_type)

if __name__ == "__main__":
    main() 