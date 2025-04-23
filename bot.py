#!/usr/bin/env python3
"""
A simpler Telegram bot for the Latvian Laws RAG-LLM system.
This version directly interacts with the RAG and LLM systems.
"""
import os
import asyncio
import io
import logging
import sys
from contextlib import redirect_stdout
from typing import Dict, Any, Optional, List, Tuple

import aiohttp  # for webhook mode if needed
from dotenv import load_dotenv
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from rag import retrieve
from llm_client import answer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("telegram_bot")

# Telegram token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_API_KEY")
if not TELEGRAM_TOKEN:
    logger.error("No TELEGRAM_API_KEY found in environment variables")
    raise ValueError("TELEGRAM_API_KEY not set in environment variables")

# Track ongoing conversations
active_users: Dict[int, bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start command is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! I'm your Latvian Law Assistant. Ask me questions about Latvian laws, and I'll try to find relevant information for you."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message when /help command is issued."""
    help_text = (
        "How to use the Latvian Law Assistant:\n\n"
        "Simply send me a question about Latvian laws, and I'll search through the database to find relevant information.\n\n"
        "Example questions:\n"
        "- What are the VAT tax rates in Latvia?\n"
        "- How are digital signatures regulated in Latvia?\n"
        "- What are the requirements for personal income tax in Latvia?\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/cancel - Cancel the current operation"
    )
    await update.message.reply_text(help_text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel current operation."""
    user_id = update.effective_user.id
    if user_id in active_users:
        del active_users[user_id]
        await update.message.reply_text("Operation cancelled.")
    else:
        await update.message.reply_text("No active operation to cancel.")

async def process_question(question: str) -> Tuple[str, List[str]]:
    """
    Process a question using the RAG and LLM systems.
    
    Args:
        question: The user's question
        
    Returns:
        Tuple of (answer_text, source_urls)
    """
    # Retrieve relevant context from RAG
    contexts = await retrieve(question)
    
    if not contexts:
        return "I couldn't find any relevant information about that in my database.", []
    
    # Extract text from context and filter empty values
    context_texts = []
    source_urls = []
    
    for ctx in contexts:
        if ctx.get("text") and ctx["text"].strip():
            # Add the source info to each context item
            formatted_text = f"From {ctx['url']}:\n{ctx['text']}"
            context_texts.append(formatted_text)
            
            # Track unique URLs
            url = ctx.get('url', '')
            if url and url not in source_urls:
                source_urls.append(url)
    
    if not context_texts:
        return "I found some matches, but couldn't extract usable text.", []
    
    # Get answer from LLM
    response = await answer(question, context_texts)
    
    return response, source_urls

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and respond with answers."""
    user_id = update.effective_user.id
    
    # Check if user has an active query
    if user_id in active_users and active_users[user_id]:
        await update.message.reply_text(
            "I'm still processing your previous question. Please wait or type /cancel to start over."
        )
        return
    
    # Set user as active
    active_users[user_id] = True
    
    # Get user's question
    question = update.message.text
    
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Indicate processing has started
        status_message = await update.message.reply_text("Searching for information...")
        
        # Process the question
        answer_text, source_urls = await process_question(question)
        
        # Format response with sources
        if source_urls:
            sources_text = "\n".join([f"- {url}" for url in source_urls])
            response = f"*Answer:*\n{answer_text}\n\n*Sources:*\n{sources_text}"
        else:
            response = answer_text
        
        # Send response
        await status_message.edit_text(response, parse_mode="Markdown", disable_web_page_preview=True)
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your question. Please try again later."
        )
    finally:
        # Clear user's active status
        if user_id in active_users:
            del active_users[user_id]

def main() -> None:
    """Start the bot."""
    # Create the application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()