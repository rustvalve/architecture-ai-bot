"""
Модуль логирования запросов к RAG-боту.

Обёртка вокруг rag_pipeline.ask(), которая сохраняет каждый запрос в JSONL-лог
с полями: timestamp, query, chunks_found, answer_length, sources, success, success_reason.

Логика определения успешности:
- success = False, если ответ содержит "I don't know" / "not enough information" / длина < 50
- success = True в остальных случаях

Использование:
    from query_logger import ask_and_log
    result = ask_and_log("What is the Aetherian Order?")
    print(result["answer"])
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import rag_pipeline

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "query_log.jsonl"


def _evaluate_success(answer: str) -> tuple[bool, str]:
    """
    Оценивает успешность ответа.
    
    Возвращает (success: bool, reason: str).
    """
    answer_lower = answer.lower()
    
    # Проверка на "не знаю"
    if "i don't know" in answer_lower:
        return False, "Contains 'I don't know'"
    
    if "not enough information" in answer_lower:
        return False, "Contains 'not enough information'"
    
    if "i cannot answer" in answer_lower:
        return False, "Contains 'I cannot answer'"
    
    # Проверка на слишком короткий ответ
    if len(answer.strip()) < 50:
        return False, f"Answer too short ({len(answer)} chars)"
    
    # Успешный ответ
    return True, "Valid answer"


def ask_and_log(
    query: str,
    log_path: str | Path | None = None,
    model: str | None = None,
    n_results: int = 5,
) -> dict:
    """
    Выполняет запрос к RAG-пайплайну и логирует результат в JSONL.
    
    Args:
        query: Текст запроса
        log_path: Путь к JSONL-логу (по умолчанию logs/query_log.jsonl)
        model: Ollama модель (по умолчанию из rag_pipeline)
        n_results: Количество чанков для поиска
    
    Returns:
        dict с ключами:
            - answer: str — ответ LLM
            - sources: list[str] — список источников
            - source_documents: list[Document] — документы из FAISS
            - log: dict — лог-запись
    """
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    else:
        log_path = Path(log_path)
    
    # Убедиться, что директория logs существует
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Вызов RAG-пайплайна
    kwargs = {"query": query, "n_results": n_results}
    if model is not None:
        kwargs["model"] = model
    
    result = rag_pipeline.ask(**kwargs)
    
    # Оценка успешности
    success, success_reason = _evaluate_success(result["answer"])
    
    # Формирование лог-записи
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "chunks_found": len(result.get("source_documents", [])),
        "answer_length": len(result["answer"]),
        "sources": result["sources"],
        "success": success,
        "success_reason": success_reason,
    }
    
    # Запись в JSONL
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    # Возврат результата с добавлением лога
    return {**result, "log": log_entry}


def clear_log(log_path: str | Path | None = None) -> None:
    """Очищает лог (удаляет файл)."""
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    else:
        log_path = Path(log_path)
    
    if log_path.exists():
        log_path.unlink()


if __name__ == "__main__":
    # Пример использования
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = ask_and_log(query)
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:", ", ".join(result["sources"]))
        print("\nLog entry:", result["log"])
    else:
        print("Usage: python query_logger.py <query>")
        print("Example: python query_logger.py 'What is the Aetherian Order?'")
