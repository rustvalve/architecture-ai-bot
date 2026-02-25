"""
Генерация эмбеддингов для чанков базы знаний.

Модель: BAAI/bge-base-en-v1.5 (Sentence Transformers).
Сохраняются: векторы (.npy), метаданные (JSON), индекс FAISS для поиска (faiss_index).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Кэш моделей в директории проекта (чтобы не требовать запись в ~/.cache)
_PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_PROJECT_ROOT / ".cache" / "transformers"))


CHUNKS_PATH = Path(__file__).resolve().parent / "knowledge_base_chunks.json"
EMBEDDINGS_NPY_PATH = Path(__file__).resolve().parent / "knowledge_base_embeddings.npy"
METADATA_JSON_PATH = Path(__file__).resolve().parent / "knowledge_base_metadata.json"
FAISS_INDEX_PATH = Path(__file__).resolve().parent / "knowledge_base_faiss.index"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32


def _source_to_title(source: str) -> str:
    """Преобразует имя файла в читаемый заголовок (aetherian_order.md -> Aetherian order)."""
    name = source.replace(".md", "").replace("_", " ").strip()
    return name.capitalize() if name else source


def _build_chunk_id(source: str, chunk_index: int) -> str:
    """Уникальный id чанка: source#index."""
    return f"{source}#{chunk_index}"


def load_chunks(path: Path) -> list[dict]:
    """Загружает чанки из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metadata_list(chunks: list[dict]) -> list[dict]:
    """Строит список метаданных для каждого чанка (путь, заголовок, id и т.д.)."""
    result = []
    for i, item in enumerate(chunks):
        meta = dict(item["metadata"])
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", i)
        result.append({
            "chunk_id": _build_chunk_id(source, chunk_index),
            "title": _source_to_title(source),
            "source": source,
            "source_path": meta.get("source_path", ""),
            "chunk_index": chunk_index,
            "total_chunks": meta.get("total_chunks"),
            "start_char": meta.get("start_char"),
            "end_char": meta.get("end_char"),
            "global_index": i,
        })
    return result


def run(
    chunks_path: Path | None = None,
    embeddings_path: Path | None = None,
    metadata_path: Path | None = None,
    faiss_path: Path | None = None,
    model_name: str = BGE_MODEL_NAME,
    batch_size: int = BATCH_SIZE,
    save_faiss: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Загружает чанки, считает эмбеддинги BGE, сохраняет векторы, метаданные и индекс FAISS.
    Возвращает (embeddings, metadata_list).
    """
    chunks_path = chunks_path or CHUNKS_PATH
    embeddings_path = embeddings_path or EMBEDDINGS_NPY_PATH
    metadata_path = metadata_path or METADATA_JSON_PATH
    faiss_path = faiss_path or FAISS_INDEX_PATH

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks in file.")

    texts = [item["content"] for item in chunks]
    metadata_list = build_metadata_list(chunks)

    cache_dir = _PROJECT_ROOT / ".cache" / "sentence_transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    print(f"Encoding {len(texts)} chunks (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Сохранение векторов
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings shape {embeddings.shape} to {embeddings_path}")

    # Сохранение метаданных (только метаданные, без content)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(metadata_list)} metadata records to {metadata_path}")

    # Индекс FAISS (cosine = inner product на нормализованных векторах)
    if save_faiss:
        try:
            import faiss
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            faiss.write_index(index, str(faiss_path))
            print(f"Saved FAISS index to {faiss_path}")
        except Exception as e:
            print(f"Warning: could not save FAISS index: {e}")

    return embeddings, metadata_list


if __name__ == "__main__":
    run()
