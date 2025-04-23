#!/usr/bin/env python3
"""
Test script for Latvian text formatting using EuroLLM-9B-Instruct via Hugging Face endpoint.
This script allows testing formatting on a single text file without database dependency.
"""
import os
import gc
import time
import argparse
import logging
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_formatter")

# Get environment variables
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://uyba45g29g72pbos.us-east4.gcp.endpoints.huggingface.cloud/v1/")
HF_API_KEY = os.getenv("HF_API_KEY")
BATCH_DELAY_MS = int(os.getenv("BATCH_DELAY_MS", "500"))  # Default to 500ms between chunks

def split_into_logical_chunks(text: str, max_size: int = 6000) -> List[str]:
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

def format_chunk(client, chunk_text: str, instruction: str) -> str:
    """Format a single chunk with the given instruction."""
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
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the response
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error formatting chunk: {str(e)}")
        return ""

def format_latvian_text(input_file, output_file, format_type="bullet_points"):
    """Format Latvian text using EuroLLM via Hugging Face endpoint with chunking for large documents."""
    try:
        # Check API key
        if not HF_API_KEY:
            logger.error("HF_API_KEY not found in environment variables")
            return False
        
        # Initialize OpenAI client with Hugging Face endpoint
        logger.info(f"Connecting to Hugging Face endpoint at {HF_ENDPOINT}")
        client = OpenAI(
            base_url=HF_ENDPOINT,
            api_key=HF_API_KEY
        )
        
        # Read input file
        logger.info(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            latvian_text = f.read()
        
        # Maximum chunk size that can be safely processed
        MAX_CHUNK_SIZE = 6000
        
        # If text is small enough, process it directly
        if len(latvian_text) <= MAX_CHUNK_SIZE:
            logger.info(f"Processing text directly ({len(latvian_text)} chars)")
            
            # Prepare prompt based on format type
            if format_type == "bullet_points":
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            elif format_type == "sections":
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            else:
                instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
            
            formatted_text = format_chunk(client, latvian_text, instruction)
        else:
            # For large texts, split into chunks and format each chunk
            logger.info(f"Text is too large ({len(latvian_text)} chars), processing in chunks")
            
            # Split the text into logical chunks
            chunks = split_into_logical_chunks(latvian_text, MAX_CHUNK_SIZE)
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
                formatted_chunk = format_chunk(client, chunk, chunk_instruction)
                if formatted_chunk:
                    formatted_chunks.append(formatted_chunk)
                
                # Apply delay between chunks to avoid rate limiting
                if i < len(chunks) - 1:
                    delay_time = max(BATCH_DELAY_MS / 1000, 0.5)
                    logger.info(f"Delaying {delay_time:.2f}s before next chunk")
                    time.sleep(delay_time)
            
            # Combine the formatted chunks
            if formatted_chunks:
                formatted_text = "\n\n".join(formatted_chunks)
                logger.info(f"Successfully formatted entire text in chunks ({len(formatted_text)} chars)")
            else:
                logger.error("Failed to format any chunks")
                return False
        
        # Write to output file
        logger.info(f"Writing formatted text to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        logger.info("Formatting complete!")
        
        # Clean up
        gc.collect()
        
        return True
        
    except Exception as e:
        logger.error(f"Error formatting text: {str(e)}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test Latvian text formatting with EuroLLM")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input text file in Latvian")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output file for formatted text")
    parser.add_argument("--format-type", "-t", type=str, default="bullet_points", 
                        choices=["bullet_points", "sections", "general"],
                        help="Type of formatting to apply")
    parser.add_argument("--chunk-delay", "-d", type=int, default=BATCH_DELAY_MS,
                       help="Delay between processing chunks in milliseconds")
    
    args = parser.parse_args()
    
    # Apply command line arguments
    global BATCH_DELAY_MS
    BATCH_DELAY_MS = args.chunk_delay
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Format the text
    success = format_latvian_text(args.input, args.output, args.format_type)
    
    if success:
        logger.info(f"Successfully formatted text from {args.input} to {args.output}")
    else:
        logger.error("Formatting failed")

if __name__ == "__main__":
    main() 