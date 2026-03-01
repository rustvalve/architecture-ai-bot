"""
RAG-пайплайн на базе LangChain RetrievalQA + Ollama c Few-shot prompting.

Пайплайн:
  1. FaissKnowledgeRetriever — оборачивает faiss_index.search() в BaseRetriever LangChain.
  2. _get_few_shot_examples() — извлекает 2 реальных примера из FAISS по фиксированным
     «showcase» вопросам и формирует секцию Examples в промпте.
  3. RetrievalQA (stuff chain) — кастомный PromptTemplate с few-shot секцией,
     контекстом и вопросом пользователя, отправляется в LLM.
  4. OllamaLLM — локальная модель (llama3.2 по умолчанию), без API-ключей.

Few-shot стратегия:
  Два вопроса-«якоря» (FEWSHOT_SEED_QUERIES) ищутся в FAISS один раз при старте.
  Топ-1 чанк каждого вопроса служит «контекстом» для эталонного ответа.
  Примеры вставляются в промпт до основного {context}, показывая модели нужный стиль.

Перед запуском:
  brew install ollama
  ollama serve          # в отдельном терминале
  ollama pull llama3.2  # ~2 GB, один раз

Запуск:
  python rag_pipeline.py "What is the Aetherian Order?"  # разовый запрос
  python rag_pipeline.py                                  # интерактивный режим
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

_PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_PROJECT_ROOT / ".cache" / "transformers"))

OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.2")
N_RESULTS: int = 5

# Фиксированные вопросы-«якоря» для few-shot примеров.
# Топ-1 чанк каждого вопроса используется как эталонный контекст/ответ.
FEWSHOT_SEED_QUERIES: list[tuple[str, str]] = [
    (
        "What is the Aetherian Order?",
        "The Aetherian Order was a noble monastic and nontheistic religious order "
        "devoted to the radiant path of synth flux. Its members served as mediators, "
        "warriors, teachers, and explorers, with a history stretching back thousands "
        "of years before the rise of the Dominion of Krath.",
    ),
    (
        "What caused the schism between the Aetherian and the Umbral Order?",
        "The schism was triggered when a rogue Aetherian led a group of followers "
        "into exile for exploring the forbidden umbral veil of synth flux. "
        "This divided the Order between those who remained faithful to the light "
        "and the vornegades who became the Umbral Order. The two factions warred "
        "for centuries afterwards.",
    ),
]

_FEWSHOT_TEMPLATE = """\
System: You are a helpful assistant who thinks step by step before answering. \
Always write out your reasoning steps explicitly, then provide the final answer.

You are a knowledgeable guide to the Synthara Chronicles universe.
Answer questions using ONLY the provided context.
If the context does not contain enough information, say "I don't know."

Here are two examples of well-formed answers drawn from the knowledge base:

Example 1:
Question: {ex1_question}
Context: {ex1_context}
Reasoning:
- The context describes the Aetherian Order as a monastic, nontheistic religious order.
- Their purpose includes serving as mediators, warriors, teachers, and explorers.
- Their history predates the rise of the Dominion of Krath by thousands of years.
Answer: {ex1_answer}

Example 2:
Question: {ex2_question}
Context: {ex2_context}
Reasoning:
- The context explains that a rogue Aetherian explored the forbidden umbral veil.
- This act caused a split: loyal members stayed, the vornegades followed the rogue into exile.
- The two resulting factions — Aetherian and Umbral — then warred for centuries.
Answer: {ex2_answer}

Now answer the following question using the context below.
First write your Reasoning steps, then write the Answer.

Context:
{context}

Question: {question}
Reasoning:"""


@lru_cache(maxsize=1)
def _get_few_shot_examples() -> tuple[dict, dict]:
    """
    Извлекает по одному реальному чанку из FAISS для каждого seed-вопроса.
    Результат кэшируется — поиск выполняется только один раз за сессию.
    """
    from faiss_index import search

    examples = []
    for question, answer in FEWSHOT_SEED_QUERIES:
        hits = search(question, n_results=1)
        context = hits[0]["document"][:600] if hits else "(no context found)"
        examples.append({"question": question, "context": context, "answer": answer})
    return examples[0], examples[1]


class FaissKnowledgeRetriever(BaseRetriever):
    """LangChain-совместимый ретривер поверх FAISS-индекса базы знаний."""

    n_results: int = N_RESULTS

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        from faiss_index import search
        hits = search(query, n_results=self.n_results)
        return [
            Document(page_content=h["document"], metadata=h["metadata"])
            for h in hits
        ]


def _build_prompt() -> PromptTemplate:
    """Строит PromptTemplate с few-shot секцией, заполняя примеры из FAISS."""
    ex1, ex2 = _get_few_shot_examples()
    filled = _FEWSHOT_TEMPLATE.format(
        ex1_question=ex1["question"],
        ex1_context=ex1["context"],
        ex1_answer=ex1["answer"],
        ex2_question=ex2["question"],
        ex2_context=ex2["context"],
        ex2_answer=ex2["answer"],
        context="{context}",
        question="{question}",
    )
    return PromptTemplate(template=filled, input_variables=["context", "question"])


def _build_chain(model: str = OLLAMA_MODEL, n_results: int = N_RESULTS):
    """Создаёт RetrievalQA chain с few-shot промптом, Ollama LLM и FAISS-ретривером."""
    from langchain_classic.chains import RetrievalQA
    from langchain_ollama import OllamaLLM

    prompt = _build_prompt()
    llm = OllamaLLM(model=model)
    retriever = FaissKnowledgeRetriever(n_results=n_results)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def ask(
    query: str,
    model: str = OLLAMA_MODEL,
    n_results: int = N_RESULTS,
) -> dict:
    """
    Полный RAG-пайплайн: запрос → FAISS → few-shot RetrievalQA → Ollama → ответ.

    Возвращает dict с ключами:
      answer  — строка с ответом LLM
      sources — список имён файлов-источников
      source_documents — список Document объектов (для логирования)
    """
    chain = _build_chain(model=model, n_results=n_results)
    result = chain.invoke({"query": query})
    sources = list(dict.fromkeys(
        d.metadata.get("source", "") for d in result["source_documents"]
    ))
    return {
        "answer": result["result"],
        "sources": sources,
        "source_documents": result["source_documents"],
    }


def _print_result(result: dict) -> None:
    print("\nAnswer:")
    print(result["answer"])
    if result["sources"]:
        print("\nSources:", ", ".join(result["sources"]))
    print()


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        _print_result(ask(query))
        return

    print(f"RAG pipeline ready (model: {OLLAMA_MODEL}). Type 'quit' to exit.\n")
    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        _print_result(ask(query))


if __name__ == "__main__":
    main()
