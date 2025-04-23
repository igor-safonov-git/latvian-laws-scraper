#!/usr/bin/env python3
"""
Latvian text formatter using EuroLLM-9B-Instruct model.
This script transforms raw Latvian texts into structured formats like bullet points or sections.
"""
import os
import gc
import json
import psutil
import logging
import asyncio
import psycopg2
import psycopg2.extras
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
MODEL_ID = os.getenv("FORMATTER_MODEL_ID", "utter-project/EuroLLM-9B-Instruct")

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
    """Formats Latvian texts into structured formats using EuroLLM model."""
    
    def __init__(self):
        """Initialize the formatter service."""
        self.db = DatabaseConnector(DATABASE_URL)
        self.tokenizer = None
        self.model = None
        
    def load_model(self) -> bool:
        """Load the EuroLLM model and tokenizer."""
        try:
            logger.info(f"Loading EuroLLM model: {MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            
            # Check for GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model with memory efficiency options
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def format_text(self, latvian_text: str, format_type: str = "bullet_points") -> str:
        """
        Format Latvian text into structured format using EuroLLM.
        
        Args:
            latvian_text: Raw Latvian text to format
            format_type: Type of formatting to apply (bullet_points, sections, etc.)
            
        Returns:
            Formatted text
        """
        if not MemoryGuard.check_memory():
            logger.error("Memory usage too high to format text")
            return ""
        
        # Truncate text if too long (model context limit)
        max_chars = 6000  # Safe limit for context window
        if len(latvian_text) > max_chars:
            logger.warning(f"Truncating text from {len(latvian_text)} to {max_chars} chars")
            latvian_text = latvian_text[:max_chars]
        
        try:
            # Prepare prompt based on format type
            if format_type == "bullet_points":
                instruction = "Pārveido šo tekstu uz strukturētu formātu ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju."
            elif format_type == "sections":
                instruction = "Pārveido šo tekstu uz strukturētu formātu ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju."
            else:
                instruction = "Pārveido šo tekstu uz strukturētu, viegli lasāmu formātu. Saglabā visu būtisko informāciju."
            
            messages = [
                {
                    "role": "system",
                    "content": "Tu esi EuroLLM — AI asistents, kas specializējas Eiropas valodās, īpaši latviešu valodā. Tu palīdzi pārveidot tekstus skaidrā, strukturētā formātā."
                },
                {
                    "role": "user", 
                    "content": f"{instruction}\n\nTEKSTS:\n{latvian_text}"
                }
            ]
            
            # Tokenize and generate
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # Generate with appropriate parameters
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            # Decode output
            formatted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            # Find where the assistant's response starts
            assist_marker = "Assistant: "
            if assist_marker in formatted_text:
                formatted_text = formatted_text.split(assist_marker, 1)[1]
            
            logger.info(f"Successfully formatted text ({len(formatted_text)} chars)")
            return formatted_text
            
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
        
        # Load model
        if not self.load_model():
            logger.error("Failed to load model, aborting batch")
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
                import time
                time.sleep(delay_time)
                
                # Help garbage collection
                raw_text = None
                formatted_text = None
                gc.collect()
            
            logger.info(f"Batch complete: processed {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
        
        finally:
            # Release GPU memory
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            # Close database connection
            self.db.disconnect()

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Format Latvian texts using EuroLLM")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process in one batch")
    parser.add_argument("--format-type", type=str, default="bullet_points", 
                        choices=["bullet_points", "sections", "general"],
                        help="Type of formatting to apply")
    parser.add_argument("--memory-limit", type=int, default=MEMORY_LIMIT_MB, 
                        help=f"Memory limit in MB (default: {MEMORY_LIMIT_MB})")
    
    args = parser.parse_args()
    
    # Apply command line arguments
    global MEMORY_LIMIT_MB
    MEMORY_LIMIT_MB = args.memory_limit
    
    logger.info(f"Using memory limit: {MEMORY_LIMIT_MB}MB, format type: {args.format_type}, batch size: {args.batch_size}")
    
    # Run the formatter
    formatter = LatvianFormatter()
    formatter.process_batch(batch_size=args.batch_size, format_type=args.format_type)

if __name__ == "__main__":
    main() 