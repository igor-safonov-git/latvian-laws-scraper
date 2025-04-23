#!/usr/bin/env python3
"""
Test script for Latvian text formatting using EuroLLM-9B-Instruct.
This script allows testing formatting on a single text file without database dependency.
"""
import os
import gc
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_formatter")

def format_latvian_text(input_file, output_file, format_type="bullet_points"):
    """Format Latvian text using EuroLLM model."""
    try:
        # Load the model
        model_id = "utter-project/EuroLLM-9B-Instruct"
        logger.info(f"Loading model: {model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
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
            instruction = "Pārveido šo tekstu uz strukturētu formātu ar aizzīmētiem punktiem. Saglabā visu būtisko informāciju."
        elif format_type == "sections":
            instruction = "Pārveido šo tekstu uz strukturētu formātu ar sekcijām un apakšsekcijām. Saglabā visu būtisko informāciju."
        else:
            instruction = "Pārveido šo tekstu uz strukturētu, viegli lasāmu formātu. Saglabā visu būtisko informāciju."
        
        # Create messages for chat template
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
        logger.info("Formatting text...")
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate with appropriate parameters
        outputs = model.generate(
            inputs, 
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode output
        formatted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assist_marker = "Assistant: "
        if assist_marker in formatted_text:
            formatted_text = formatted_text.split(assist_marker, 1)[1]
        
        # Write to output file
        logger.info(f"Writing formatted text to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
        
        logger.info("Formatting complete!")
        
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
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