FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
COPY telegram_bot/requirements.txt ./telegram_bot/
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r telegram_bot/requirements.txt

COPY . .

CMD ["python", "-m", "telegram_bot.bot"]
