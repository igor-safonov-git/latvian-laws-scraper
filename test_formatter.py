#!/usr/bin/env python3
"""
Test script for Latvian text formatting using EuroLLM-9B-Instruct via Hugging Face endpoint.
This script allows testing formatting on a single text file without database dependency.
"""
import os
import gc
import argparse
import logging
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

def format_latvian_text(input_file, output_file, format_type="bullet_points"):
    """Format Latvian text using EuroLLM via Hugging Face endpoint."""
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
        
        # Truncate if needed
        max_chars = 6000
        if len(latvian_text) > max_chars:
            logger.warning(f"Truncating text from {len(latvian_text)} to {max_chars} chars")
            latvian_text = latvian_text[:max_chars]
        
        # Prepare prompt based on format type
        if format_type == "bullet_points":
            instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
        elif format_type == "sections":
            instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
        else:
            instruction = "Organizē šo juridisko tekstu skaidrā, strukturētā formātā. Saglabā visu būtisko informāciju, izcel svarīgākās tēmas un saglabā oriģinālo nozīmi, vienlaikus uzlabojot lasāmību."
        
        # Create messages for chat API
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
        
        # Call the API
        logger.info("Sending request to Hugging Face endpoint...")
        response = client.chat.completions.create(
            model="tgi",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the response
        formatted_text = response.choices[0].message.content
        
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
    
    args = parser.parse_args()
    
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