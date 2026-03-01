"""
Скрипт автоматического тестирования RAG-бота на золотом наборе вопросов.

Загружает golden_questions.json, для каждого вопроса вызывает ask_and_log(),
сравнивает результат с ожидаемым (expected_success) и формирует отчёт.

Использование:
    python evaluate.py
    python evaluate.py --model llama3.2 --n-results 10
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from query_logger import ask_and_log

_PROJECT_ROOT = Path(__file__).resolve().parent
GOLDEN_QUESTIONS_PATH = _PROJECT_ROOT / "golden_questions.json"
EVALUATION_REPORT_PATH = _PROJECT_ROOT / "evaluation_report.json"


def load_golden_questions(path: Path = GOLDEN_QUESTIONS_PATH) -> list[dict]:
    """Загружает золотой набор вопросов из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_keywords(answer: str, keywords: list[str]) -> dict:
    """
    Проверяет наличие ключевых слов в ответе.
    
    Возвращает {"found": int, "total": int, "missing": list[str]}.
    """
    answer_lower = answer.lower()
    missing = []
    found = 0
    
    for keyword in keywords:
        if keyword.lower() in answer_lower:
            found += 1
        else:
            missing.append(keyword)
    
    return {
        "found": found,
        "total": len(keywords),
        "missing": missing,
    }


def evaluate(
    model: str | None = None,
    n_results: int = 5,
    golden_path: Path = GOLDEN_QUESTIONS_PATH,
) -> list[dict]:
    """
    Выполняет оценку качества RAG-бота на золотом наборе вопросов.
    
    Args:
        model: Ollama модель (по умолчанию из rag_pipeline)
        n_results: Количество чанков для поиска
        golden_path: Путь к golden_questions.json
    
    Returns:
        list[dict] — результаты тестирования по каждому вопросу
    """
    questions = load_golden_questions(golden_path)
    results = []
    
    print(f"Запуск оценки на {len(questions)} вопросах...")
    print(f"Модель: {model or 'default (llama3.2)'}, n_results: {n_results}")
    print("-" * 80)
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question']}")
        print(f"   Ожидается: {'✓ успех' if q['expected_success'] else '✗ неудача'}")
        
        # Вызов RAG-бота
        kwargs = {"query": q["question"], "n_results": n_results}
        if model is not None:
            kwargs["model"] = model
        
        result = ask_and_log(**kwargs)
        
        # Сравнение с ожидаемым
        actual_success = result["log"]["success"]
        match = (actual_success == q["expected_success"])
        
        # Проверка ключевых слов (только для успешных вопросов)
        keywords_check = check_keywords(result["answer"], q.get("expected_keywords", []))
        
        # Проверка источника
        expected_source = q.get("expected_source", "")
        source_match = (expected_source in result["sources"]) if expected_source else None
        
        # Результат по вопросу
        result_entry = {
            "id": q["id"],
            "question": q["question"],
            "topic": q["topic"],
            "expected_success": q["expected_success"],
            "actual_success": actual_success,
            "success_reason": result["log"]["success_reason"],
            "match": match,
            "keywords_check": keywords_check,
            "source_match": source_match,
            "answer_length": result["log"]["answer_length"],
            "chunks_found": result["log"]["chunks_found"],
            "sources": result["sources"],
            "answer_preview": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
        }
        results.append(result_entry)
        
        # Вывод результата
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"   Фактически: {'✓ успех' if actual_success else '✗ неудача'} — {status}")
        
        if q["expected_success"] and actual_success:
            print(f"   Ключевые слова: {keywords_check['found']}/{keywords_check['total']}")
            if keywords_check['missing']:
                print(f"   Отсутствуют: {', '.join(keywords_check['missing'])}")
            if source_match is not None:
                print(f"   Источник: {'✓' if source_match else '✗'} {expected_source}")
    
    return results


def print_report(results: list[dict]) -> None:
    """Выводит отчёт по результатам тестирования."""
    total = len(results)
    matches = sum(1 for r in results if r["match"])
    accuracy = matches / total * 100 if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("ОТЧЁТ ПО ОЦЕНКЕ КАЧЕСТВА RAG-БОТА")
    print("=" * 80)
    print(f"\nОбщая точность: {matches}/{total} ({accuracy:.1f}%)")
    
    # Разбивка по успешным/неуспешным
    expected_success = [r for r in results if r["expected_success"]]
    expected_failure = [r for r in results if not r["expected_success"]]
    
    success_matches = sum(1 for r in expected_success if r["match"])
    failure_matches = sum(1 for r in expected_failure if r["match"])
    
    print(f"\nВопросы с ожидаемым успехом: {success_matches}/{len(expected_success)} корректно")
    print(f"Вопросы с ожидаемой неудачей: {failure_matches}/{len(expected_failure)} корректно")
    
    # Список неудач
    failed = [r for r in results if not r["match"]]
    if failed:
        print(f"\n{'─' * 80}")
        print(f"ОШИБКИ ({len(failed)}):")
        print(f"{'─' * 80}")
        for r in failed:
            print(f"\nID {r['id']}: {r['question']}")
            print(f"   Топик: {r['topic']}")
            print(f"   Ожидалось: {'успех' if r['expected_success'] else 'неудача'}")
            print(f"   Фактически: {'успех' if r['actual_success'] else 'неудача'}")
            print(f"   Причина: {r['success_reason']}")
            print(f"   Чанков найдено: {r['chunks_found']}")
            print(f"   Источники: {', '.join(r['sources']) if r['sources'] else 'нет'}")
    
    # Темы без ответа
    no_answer_topics = [r["topic"] for r in results if r["expected_success"] and not r["actual_success"]]
    if no_answer_topics:
        print(f"\n{'─' * 80}")
        print("ТЕМЫ БЕЗ ОТВЕТА:")
        print(f"{'─' * 80}")
        for topic in no_answer_topics:
            print(f"  - {topic}")
    
    # Нерелевантные источники
    wrong_sources = [
        r for r in expected_success 
        if r["source_match"] is False and r["actual_success"]
    ]
    if wrong_sources:
        print(f"\n{'─' * 80}")
        print("НЕРЕЛЕВАНТНЫЕ ИСТОЧНИКИ:")
        print(f"{'─' * 80}")
        for r in wrong_sources:
            print(f"\nID {r['id']}: {r['question']}")
            print(f"   Ожидался: {r.get('expected_source', 'N/A')}")
            print(f"   Получены: {', '.join(r['sources'])}")
    
    print("\n" + "=" * 80)


def save_results(results: list[dict], output_path: Path = EVALUATION_REPORT_PATH) -> None:
    """Сохраняет результаты тестирования в JSON."""
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_questions": len(results),
        "matches": sum(1 for r in results if r["match"]),
        "accuracy": sum(1 for r in results if r["match"]) / len(results) * 100 if results else 0,
        "results": results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчёт сохранён в {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества RAG-бота на золотом наборе")
    parser.add_argument("--model", type=str, default=None, help="Ollama модель (по умолчанию llama3.2)")
    parser.add_argument("--n-results", type=int, default=5, help="Количество чанков для поиска")
    parser.add_argument("--golden", type=str, default=None, help="Путь к golden_questions.json")
    parser.add_argument("--output", type=str, default=None, help="Путь для сохранения отчёта")
    
    args = parser.parse_args()
    
    golden_path = Path(args.golden) if args.golden else GOLDEN_QUESTIONS_PATH
    output_path = Path(args.output) if args.output else EVALUATION_REPORT_PATH
    
    # Запуск оценки
    results = evaluate(model=args.model, n_results=args.n_results, golden_path=golden_path)
    
    # Вывод отчёта
    print_report(results)
    
    # Сохранение результатов
    save_results(results, output_path)


if __name__ == "__main__":
    main()
