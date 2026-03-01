# Telegram бот для RAG pipeline

Интерфейс в виде Telegram бота для вопросов по базе знаний Synthara Chronicles. Бот вызывает `rag_pipeline.ask()` и отправляет ответ с источниками.

## Требования

- Python 3.11+
- Собранный индекс: в корне проекта выполните `python build_index.py` (один раз).
- Запущенный Ollama с моделью: в отдельном терминале `ollama serve`, затем `ollama pull llama3.2`.
- Токен бота: получите у [@BotFather](https://t.me/BotFather), создайте бота и скопируйте токен.

## Установка

1. Из корня проекта `architecture-ai-bot` установите зависимости основного RAG pipeline и бота:

   ```bash
   pip install -r requirements.txt
   pip install -r telegram_bot/requirements.txt
   ```

2. Создайте файл `.env` в папке `telegram_bot/`:

   ```bash
   cp telegram_bot/.env.example telegram_bot/.env
   ```

3. Откройте `telegram_bot/.env` и укажите свой токен:

   ```
   BOT_TOKEN=123456:ABC-DEF...
   OLLAMA_MODEL=llama3.2
   ```

## Запуск

Из корня проекта:

```bash
python -m telegram_bot.bot
```

Либо из папки `telegram_bot/`:

```bash
cd telegram_bot && python bot.py
```

После запуска напишите боту в Telegram любой вопрос — он ответит на основе базы знаний и пришлёт список источников.
