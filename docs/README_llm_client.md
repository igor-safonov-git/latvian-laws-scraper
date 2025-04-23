# LLM Client Module

An asyncio-based Python module for interacting with the OpenAI API to answer questions using provided context.

## Features

- Asynchronous API for efficient handling of multiple requests
- Structured context-based prompting
- Web search tool integration
- Comprehensive token usage and cost logging
- Error handling and detailed logging
- Proper environment variable management

## Installation

1. Ensure you have Python 3.7+ installed
2. Install the required dependencies:

```bash
pip install aiohttp python-dotenv
```

3. Set up your environment variables in a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key
MODEL=gpt-4
```

## Usage

```python
import asyncio
from llm_client import answer

async def main():
    question = "What is the capital of France?"
    context = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe."
    ]
    
    result = await answer(question, context)
    print(f"Answer: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## API Reference

### `answer(question: str, context: List[str]) -> str`

Queries the OpenAI API with a question and provided context.

**Parameters:**
- `question` (str): The question to ask
- `context` (List[str]): List of context passages to include in the prompt

**Returns:**
- `str`: The answer to the question based on the context

**Raises:**
- `ValueError`: If the OpenAI API key is not provided
- `Exception`: If there's an error in the API call

## Logging

The module logs token usage and cost information to `./logs/llm_client.log` in JSON format:

```json
{
    "timestamp": "2025-04-23T15:30:45.123456",
    "model": "gpt-4",
    "prompt_tokens": 250,
    "completion_tokens": 50,
    "total_tokens": 300,
    "cost_usd": 0.01095,
    "duration_seconds": 1.25
}
```

## Testing

Run the test suite to verify the functionality:

```bash
python test_llm_client.py
```