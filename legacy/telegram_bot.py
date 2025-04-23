#!/usr/bin/env python3
"""
Telegram bot that interfaces with the RAG-LLM system for Latvian laws.
"""
import os
import logging
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from rag_with_llm import rag_with_llm

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("telegram_bot")

# Get Telegram token from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("No TELEGRAM_TOKEN found in environment variables")
    raise ValueError("TELEGRAM_TOKEN not set in environment variables")

# Track ongoing conversations to prevent multiple simultaneous queries
active_users: Dict[int, bool] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}! I'm your Latvian Law Assistant. Ask me questions about Latvian laws, and I'll try to find relevant information for you."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
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
    """Cancel current operation and reset user state."""
    user_id = update.effective_user.id
    if user_id in active_users:
        del active_users[user_id]
        await update.message.reply_text("Operation cancelled.")
    else:
        await update.message.reply_text("No active operation to cancel.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user questions and get answers from the RAG-LLM system."""
    user_id = update.effective_user.id
    
    # Check if user already has an active query
    if user_id in active_users and active_users[user_id]:
        await update.message.reply_text(
            "I'm still processing your previous question. Please wait or type /cancel to start over."
        )
        return
    
    # Set user as active
    active_users[user_id] = True
    
    # Get the user's question
    question = update.message.text
    
    try:
        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Indicate that processing has started
        status_message = await update.message.reply_text("Searching for information...")
        
        # Process the question through RAG-LLM
        # Create a string buffer to capture print outputs
        import io
        import sys
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            await rag_with_llm(question)
        
        # Parse the output to extract answer and sources
        output = output_buffer.getvalue()
        
        # Process and send the response
        if "Sorry, no relevant context was found for your question" in output:
            await status_message.edit_text("I couldn't find any relevant information about that topic in my database.")
        else:
            # Split the output to separate answer and sources
            try:
                answer_section = output.split("Answer:")[1].split("Sources:")[0].strip()
                sources_section = output.split("Sources:")[1].strip()
                
                # Format the response
                response = f"*Answer:*\n{answer_section}\n\n*Sources:*\n{sources_section}"
                
                await status_message.edit_text(response, parse_mode="Markdown")
            except IndexError:
                # Fallback if parsing fails
                await status_message.edit_text(output)
        
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
    # Create the Application
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