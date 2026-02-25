"""
Поиск по индексу FAISS базы знаний.

Загружает индекс FAISS, метаданные и чанки; эмбеддинг запроса — модель BGE (как при построении индекса).
Возвращает те же структуры, что и раньше Chroma: id, document, metadata, distance (cosine distance, 0 = best).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_PROJECT_ROOT / ".cache" / "transformers"))

CHUNKS_PATH = _PROJECT_ROOT / "knowledge_base_chunks.json"
METADATA_JSON_PATH = _PROJECT_ROOT / "knowledge_base_metadata.json"
FAISS_INDEX_PATH = _PROJECT_ROOT / "knowledge_base_faiss.index"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def _load_index_data():
    """Загружает чанки, метаданные и индекс FAISS."""
    import faiss

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(METADATA_JSON_PATH, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    if len(chunks) != len(metadata_list) or index.ntotal != len(chunks):
        raise ValueError(
            f"Mismatch: chunks={len(chunks)}, metadata={len(metadata_list)}, index.ntotal={index.ntotal}"
        )
    return chunks, metadata_list, index


def search(
    query: str,
    n_results: int = 5,
    model_name: str = BGE_MODEL_NAME,
) -> list[dict]:
    """
    Поиск по текстовому запросу: эмбеддинг запроса через BGE, затем k-NN в FAISS.
    Возвращает список dict с ключами: id, document, metadata, distance (cosine distance, 0 = best).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks, metadata_list, index = _load_index_data()
    n_results = min(n_results, len(chunks))

    cache_dir = _PROJECT_ROOT / ".cache" / "sentence_transformers"
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32).reshape(1, -1)

    # IndexFlatIP: inner product (для нормализованных векторов = cosine similarity)
    scores, indices = index.search(query_embedding, n_results)
    # cosine distance = 1 - similarity, чтобы 0 = лучший результат (как в Chroma)
    distances = 1.0 - scores[0]

    out = []
    for i, idx in enumerate(indices[0]):
        idx = int(idx)
        meta = metadata_list[idx]
        doc = chunks[idx]["content"]
        out.append({
            "id": meta["chunk_id"],
            "document": doc,
            "metadata": meta,
            "distance": float(distances[i]),
        })
    return out


if __name__ == "__main__":
    for hit in search("Aetherian Order", n_results=2):
        print(f"  id={hit['id']}, distance={hit['distance']:.4f}, source={hit['metadata'].get('source')}")
