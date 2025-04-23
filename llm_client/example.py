#!/usr/bin/env python3
"""
Example usage of the LLM client module.
"""
import asyncio
from llm_client import answer

async def main():
    """Demonstrate the LLM client with a simple example."""
    # Example question
    question = "What is the capital of France?"
    
    # Example context
    context = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe with several overseas territories.",
        "The area of metropolitan France is 551,695 square kilometers."
    ]
    
    print(f"Question: {question}")
    print(f"Context: {len(context)} passages provided")
    
    # Call the answer function
    try:
        result = await answer(question, context)
        print("\nAnswer:")
        print(result)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())