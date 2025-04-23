#!/usr/bin/env python3
"""
Simple script to retrieve context from RAG and pass it to the LLM for answering questions.
"""
import asyncio
import sys
from rag import retrieve
from llm_client import answer

async def rag_with_llm(question: str):
    """
    Retrieve context from RAG and pass it to the LLM for answering.
    
    Args:
        question: The user's question
    """
    print(f"Question: {question}")
    print("Retrieving context...")
    
    # Retrieve context from RAG
    contexts = await retrieve(question)
    
    if not contexts:
        print("Sorry, no relevant context was found for your question.")
        return
    
    print(f"Found {len(contexts)} relevant passages")
    
    # Extract text from context results
    context_texts = [ctx["text"] for ctx in contexts]
    
    # Pass the question and context to the LLM
    print("Generating answer...")
    
    try:
        response = await answer(question, context_texts)
        
        print("\nAnswer:")
        print(response)
        
        print("\nSources:")
        for i, ctx in enumerate(contexts):
            print(f"- {ctx['url']}")
    
    except Exception as e:
        print(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    # Get the question from command line arguments or prompt the user
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")
    
    asyncio.run(rag_with_llm(question))