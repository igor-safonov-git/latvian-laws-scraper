#!/usr/bin/env python3
"""
Simple LLM Client module that provides an interface to OpenAI APIs.
"""
import os
import logging
import asyncio
from typing import List
import aiohttp
from dotenv import load_dotenv

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("llm_client")

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

# Add file handler
file_handler = logging.FileHandler("./logs/llm_client.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4")

# Web search tool definition
web_search_tool = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information that might help answer the question",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }
    }
}

async def answer(question: str, context: List[str]) -> str:
    """
    Query the OpenAI API with a question and provided context.
    
    Args:
        question: The question to ask
        context: List of context passages to include in the prompt
    
    Returns:
        The answer to the question based on the context
    """
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not found in environment variables")
        raise ValueError("OPENAI_API_KEY not provided. Please set it in the .env file.")

    # Prepare context by joining with newlines and double newlines between items
    joined_context = "\n\n".join(context)
    
    # Construct the complete prompt
    prompt = f"Answer the question using these excerpts:\n\n{joined_context}\n\nQ: {question}\nA:"
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the context provided."},
            {"role": "user", "content": prompt}
        ],
        "tools": [web_search_tool],
        "temperature": 0.0
    }
    
    # Log basic info
    logger.info(f"Question: {question}")
    logger.info(f"Context: {len(context)} excerpts, {len(joined_context)} characters")
    
    try:
        # Make the API request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API Error: {response.status}, {error_text}")
                    raise Exception(f"API Error: {response.status}, {error_text}")
                
                result = await response.json()
        
        # Extract the response text from the best choice
        response_message = result["choices"][0]["message"]
        answer_text = response_message["content"]
        
        # Log success
        logger.info(f"Successfully generated answer ({len(answer_text)} chars)")
        
        return answer_text
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        raise

# If run directly, demonstrate usage
if __name__ == "__main__":
    async def test():
        question = "What is the capital of France?"
        context = ["Paris is the capital and most populous city of France.", 
                  "France is a country in Western Europe."]
        
        result = await answer(question, context)
        print(f"Question: {question}")
        print(f"Answer: {result}")
    
    asyncio.run(test())