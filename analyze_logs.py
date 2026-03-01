"""
Скрипт анализа логов запросов к RAG-боту.

Читает logs/query_log.jsonl и выводит статистику:
- Общее число запросов, % успешных/неуспешных
- Средняя длина ответа (успешных vs неуспешных)
- Топ источников (какие файлы чаще цитируются)
- Вопросы без ответа — список тем, по которым бот «слеп»
- Рекомендации по расширению базы знаний

Использование:
    python analyze_logs.py
    python analyze_logs.py --log logs/query_log.jsonl --top 10
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "query_log.jsonl"


def load_log(log_path: Path) -> list[dict]:
    """Загружает JSONL-лог в список dict."""
    if not log_path.exists():
        return []
    
    logs = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return logs


def analyze_logs(log_path: Path = DEFAULT_LOG_PATH, top_n: int = 10) -> dict:
    """
    Анализирует лог запросов и возвращает статистику.
    
    Args:
        log_path: Путь к JSONL-логу
        top_n: Количество топ-источников для вывода
    
    Returns:
        dict со статистикой
    """
    logs = load_log(log_path)
    
    if not logs:
        return {"error": "Лог пуст или не найден"}
    
    # Базовая статистика
    total = len(logs)
    successful = [log for log in logs if log["success"]]
    failed = [log for log in logs if not log["success"]]
    
    success_rate = len(successful) / total * 100 if total > 0 else 0
    
    # Средняя длина ответа
    avg_length_success = (
        sum(log["answer_length"] for log in successful) / len(successful)
        if successful else 0
    )
    avg_length_failed = (
        sum(log["answer_length"] for log in failed) / len(failed)
        if failed else 0
    )
    
    # Средние чанки
    avg_chunks_success = (
        sum(log["chunks_found"] for log in successful) / len(successful)
        if successful else 0
    )
    avg_chunks_failed = (
        sum(log["chunks_found"] for log in failed) / len(failed)
        if failed else 0
    )
    
    # Топ источников
    all_sources = []
    for log in logs:
        all_sources.extend(log.get("sources", []))
    source_counts = Counter(all_sources)
    top_sources = source_counts.most_common(top_n)
    
    # Причины неудач
    failure_reasons = Counter(log["success_reason"] for log in failed)
    
    # Вопросы без ответа
    failed_queries = [log["query"] for log in failed]
    
    # Запросы с нулевыми чанками
    no_chunks = [log["query"] for log in logs if log["chunks_found"] == 0]
    
    return {
        "total_queries": total,
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": success_rate,
        "avg_length_success": avg_length_success,
        "avg_length_failed": avg_length_failed,
        "avg_chunks_success": avg_chunks_success,
        "avg_chunks_failed": avg_chunks_failed,
        "top_sources": top_sources,
        "failure_reasons": failure_reasons.most_common(),
        "failed_queries": failed_queries,
        "no_chunks_queries": no_chunks,
    }


def print_analysis(stats: dict) -> None:
    """Выводит анализ логов в консоль."""
    if "error" in stats:
        print(f"Ошибка: {stats['error']}")
        return
    
    print("=" * 80)
    print("АНАЛИЗ ЛОГОВ ЗАПРОСОВ К RAG-БОТУ")
    print("=" * 80)
    
    # Базовая статистика
    print(f"\n📊 Общая статистика:")
    print(f"   Всего запросов: {stats['total_queries']}")
    print(f"   Успешных: {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"   Неудачных: {stats['failed']} ({100 - stats['success_rate']:.1f}%)")
    
    # Длина ответов
    print(f"\n📏 Средняя длина ответа:")
    print(f"   Успешные: {stats['avg_length_success']:.0f} символов")
    print(f"   Неудачные: {stats['avg_length_failed']:.0f} символов")
    
    # Чанки
    print(f"\n📦 Среднее число найденных чанков:")
    print(f"   Успешные: {stats['avg_chunks_success']:.1f}")
    print(f"   Неудачные: {stats['avg_chunks_failed']:.1f}")
    
    # Топ источников
    if stats['top_sources']:
        print(f"\n📚 Топ источников (чаще всего цитируются):")
        for i, (source, count) in enumerate(stats['top_sources'], 1):
            print(f"   {i:2d}. {source:<40} ({count} раз)")
    
    # Причины неудач
    if stats['failure_reasons']:
        print(f"\n❌ Причины неудач:")
        for reason, count in stats['failure_reasons']:
            print(f"   - {reason}: {count} раз")
    
    # Вопросы без ответа
    if stats['failed_queries']:
        print(f"\n🚫 Вопросы без ответа ({len(stats['failed_queries'])}):")
        for i, query in enumerate(stats['failed_queries'][:10], 1):
            print(f"   {i:2d}. {query}")
        if len(stats['failed_queries']) > 10:
            print(f"   ... и ещё {len(stats['failed_queries']) - 10}")
    
    # Запросы с нулевыми чанками
    if stats['no_chunks_queries']:
        print(f"\n🔍 Запросы с нулевыми чанками ({len(stats['no_chunks_queries'])}):")
        for i, query in enumerate(stats['no_chunks_queries'][:5], 1):
            print(f"   {i:2d}. {query}")
        if len(stats['no_chunks_queries']) > 5:
            print(f"   ... и ещё {len(stats['no_chunks_queries']) - 5}")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    print("-" * 80)
    
    if stats['failed'] > stats['successful']:
        print("⚠️  Критическая проблема: неудачных запросов больше, чем успешных!")
        print("   Рекомендации:")
        print("   1. Расширить базу знаний по темам из 'Вопросы без ответа'")
        print("   2. Проверить качество чанкирования (размер чанков, overlap)")
        print("   3. Улучшить промпт для LLM")
    
    if stats['no_chunks_queries']:
        print("\n⚠️  Найдены запросы с нулевыми чанками:")
        print("   Рекомендации:")
        print("   1. Добавить документы по этим темам в базу знаний")
        print("   2. Проверить релевантность эмбеддингов (возможно, нужна другая модель)")
    
    if stats['avg_chunks_failed'] < 2 and stats['failed'] > 0:
        print("\n⚠️  У неудачных запросов мало найденных чанков:")
        print("   Рекомендации:")
        print("   1. Увеличить n_results в поиске (например, с 5 до 10)")
        print("   2. Улучшить качество индекса (пересобрать с другим чанкированием)")
    
    if not stats['failure_reasons']:
        print("\n✅ Отличная работа! Все запросы успешны.")
    
    # Темы для расширения базы знаний
    if stats['failed_queries']:
        print(f"\n📝 Темы для расширения базы знаний:")
        # Простой анализ: извлечение ключевых слов из неудачных запросов
        failed_keywords = set()
        for query in stats['failed_queries']:
            words = [w.strip("?,.:;!") for w in query.lower().split()]
            # Фильтруем короткие и служебные слова
            keywords = [
                w for w in words 
                if len(w) > 3 and w not in {"what", "who", "where", "when", "how", "which", "describe", "explain", "the", "and", "are", "from"}
            ]
            failed_keywords.update(keywords)
        
        if failed_keywords:
            print("   Возможные топики для добавления:")
            for keyword in sorted(failed_keywords)[:15]:
                print(f"   - {keyword}")
    
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ логов запросов к RAG-боту")
    parser.add_argument("--log", type=str, default=None, help="Путь к JSONL-логу")
    parser.add_argument("--top", type=int, default=10, help="Количество топ-источников")
    parser.add_argument("--json", action="store_true", help="Вывести результат в JSON")
    
    args = parser.parse_args()
    
    log_path = Path(args.log) if args.log else DEFAULT_LOG_PATH
    
    # Анализ
    stats = analyze_logs(log_path, top_n=args.top)
    
    # Вывод
    if args.json:
        # JSON-вывод для программной обработки
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
    else:
        # Человекочитаемый вывод
        print_analysis(stats)


if __name__ == "__main__":
    main()
