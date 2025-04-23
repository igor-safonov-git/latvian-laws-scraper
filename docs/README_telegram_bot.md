# Telegram Bot for Latvian Laws

A Telegram bot interface for the Latvian Laws RAG-LLM system that allows users to ask questions about Latvian laws and get answers in Telegram.

## Features

- Simple Telegram interface to the RAG-LLM system
- Direct integration with the retrieval and answer generation components
- User-friendly messages and command handling
- Prevents multiple simultaneous queries from the same user
- Provides sources for the information

## Setup

1. Create a Telegram bot using [BotFather](https://t.me/botfather) and get your bot token
2. Add the token to your `.env` file:
   ```
   TELEGRAM_API_KEY=your_telegram_bot_token
   OPENAI_API_KEY=your_openai_api_key
   MODEL=gpt-4
   DATABASE_URL=your_postgres_db_url
   ```
3. Install required dependencies:
   ```
   pip install python-telegram-bot==20.8
   ```

## Usage

Run the bot:

```bash
python telegram_llm_bot.py
```

On Heroku (to run as a service):

```bash
heroku ps:scale bot=1 --app latvian-laws
```

## Bot Commands

- `/start` - Start the bot with a welcome message
- `/help` - Show help information and example questions
- `/cancel` - Cancel any running operations

## How It Works

1. Users send questions to the Telegram bot
2. The bot passes the question to the RAG system to retrieve relevant context
3. The retrieved context is sent to the LLM for answer generation
4. The answer and sources are sent back to the user in Telegram

## Example Questions

- "What are the VAT tax rates in Latvia?"
- "How are digital signatures regulated in Latvia?"
- "What are the requirements for personal income tax in Latvia?"

## Adding to Procfile

To run the bot as a service on Heroku, add this line to your Procfile:

```
bot: python telegram_llm_bot.py
```