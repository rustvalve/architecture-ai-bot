"""
Полный пайплайн индексации базы знаний.

1. Разбиение документов на чанки (RecursiveCharacterTextSplitter)
2. Генерация эмбеддингов (BAAI/bge-base-en-v1.5) и сохранение .npy, metadata.json, FAISS-индекс

Результат: knowledge_base_faiss.index, chunks.json, embeddings.npy, metadata.json.
Время каждого шага и общее время записываются в index_info.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_INFO_PATH = PROJECT_ROOT / "index_info.json"


def main() -> dict:
    timings = {}
    total_start = time.perf_counter()

    # 1. Чанки
    t0 = time.perf_counter()
    from chunk_knowledge_base import run as run_chunk
    chunks = run_chunk()
    timings["chunk_seconds"] = round(time.perf_counter() - t0, 2)
    timings["n_chunks"] = len(chunks)

    # 2. Эмбеддинги + FAISS
    t0 = time.perf_counter()
    from embed_chunks import run as run_embed
    run_embed(save_faiss=True)
    timings["embed_seconds"] = round(time.perf_counter() - t0, 2)

    timings["total_seconds"] = round(time.perf_counter() - total_start, 2)
    timings["model"] = "BAAI/bge-base-en-v1.5"
    timings["knowledge_base"] = str(PROJECT_ROOT / "knowledge_base")
    timings["embedding_dim"] = 768

    INDEX_INFO_PATH.write_text(json.dumps(timings, indent=2, ensure_ascii=False))
    print(f"\nTiming saved to {INDEX_INFO_PATH}")
    print(json.dumps(timings, indent=2))
    return timings


if __name__ == "__main__":
    main()
