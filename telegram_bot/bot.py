"""
Telegram bot for the RAG pipeline: accepts questions and returns answers from the Synthara Chronicles knowledge base.
Run from project root: python -m telegram_bot.bot (or from telegram_bot: python bot.py).
"""

import asyncio
import sys
from pathlib import Path

# Добавляем родительскую директорию в путь для импорта rag_pipeline
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message

from telegram_bot import config

# Импорт после настройки path
from rag_pipeline import ask

dp = Dispatcher()
MAX_MESSAGE_LENGTH = 4096


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Hi! I'm a bot for the Synthara Chronicles universe. "
        "Ask me any question — I'll answer from the knowledge base.\n\n"
        "Examples: «What is the Aetherian Order?», «Who is Commander Thrane?»"
    )


async def on_text(message: Message) -> None:
    query = (message.text or "").strip()
    if not query:
        return
    chat_id = message.chat.id
    bot = message.bot
    try:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        result = await asyncio.to_thread(ask, query, model=config.OLLAMA_MODEL)
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources") or []
        if not answer:
            answer = "Could not get an answer."
        if sources:
            answer += "\n\n📎 Sources: " + ", ".join(sources)
        if len(answer) > MAX_MESSAGE_LENGTH:
            answer = answer[: MAX_MESSAGE_LENGTH - 50] + "\n\n… (truncated)"
        await message.answer(answer)
    except FileNotFoundError:
        await message.answer(
            "Error: knowledge base index not found. From project root run: python build_index.py"
        )
    except Exception as e:
        err = str(e).strip() or type(e).__name__
        await message.answer(f"Error processing request: {err}")


def main() -> None:
    if not config.BOT_TOKEN:
        print("Set BOT_TOKEN in .env (copy from .env.example)")
        sys.exit(1)
    bot = Bot(token=config.BOT_TOKEN)
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(on_text)
    asyncio.run(dp.start_polling(bot))


if __name__ == "__main__":
    main()
