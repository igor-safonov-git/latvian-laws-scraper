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
    
    # Extract text from context results and filter out empty or None values
    context_texts = []
    for ctx in contexts:
        if ctx.get("text") and ctx["text"].strip():
            # Add some content around the text snippets to make them more usable
            formatted_text = f"From {ctx['url']}:\n{ctx['text']}"
            context_texts.append(formatted_text)
    
    if not context_texts:
        print("Sorry, the retrieved context didn't contain usable text.")
        return
    
    # Pass the question and context to the LLM
    print("Generating answer...")
    
    response = await answer(question, context_texts)
    
    print("\nAnswer:")
    print(response)
    
    print("\nSources:")
    urls_seen = set()
    for ctx in contexts:
        url = ctx.get('url', '')
        if url and url not in urls_seen:
            print(f"- {url}")
            urls_seen.add(url)

if __name__ == "__main__":
    # Get the question from command line arguments or prompt the user
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ")
    
    asyncio.run(rag_with_llm(question))