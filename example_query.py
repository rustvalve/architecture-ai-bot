"""
Пример запроса к индексу: поисковый запрос + найденные чанки.

Использует FAISS (faiss_index.search).
Запуск: python example_query.py
"""

from __future__ import annotations

from faiss_index import search

QUERY = "What is the Aetherian Order and who are the Aetherian Knights?"
N_RESULTS = 3
SNIPPET_LEN = 300


def main() -> None:
    print("Search query:", QUERY)
    print("Number of results:", N_RESULTS)
    print("-" * 60)

    results = search(QUERY, n_results=N_RESULTS)

    for i, hit in enumerate(results, 1):
        meta = hit["metadata"]
        doc = hit["document"]
        snippet = (doc[:SNIPPET_LEN] + "...") if len(doc) > SNIPPET_LEN else doc
        print(f"\n--- Chunk {i} ---")
        print(f"  id:       {hit['id']}")
        print(f"  source:   {meta.get('source', '')}")
        print(f"  title:    {meta.get('title', '')}")
        print(f"  distance: {hit['distance']:.4f} (cosine distance, 0 = best)")
        print(f"  snippet:\n    {snippet.strip()}")
    print("\n" + "-" * 60)
    print(f"Total chunks returned: {len(results)}")


if __name__ == "__main__":
    main()
