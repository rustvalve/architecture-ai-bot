"""
Загрузка конфигурации Telegram бота из переменных окружения (.env).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_BOT_DIR = Path(__file__).resolve().parent
load_dotenv(_BOT_DIR / ".env")

BOT_TOKEN: str = os.environ.get("BOT_TOKEN", "").strip()
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
