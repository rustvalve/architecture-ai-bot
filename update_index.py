"""
Инкрементальное обновление векторного индекса базы знаний.

- Первое сканирование / полная пересборка: чанки и индекс строятся из knowledge_base (оригинальная база).
- Обновления: сканируется docs/ (аналог S3), новые/изменённые/удалённые .md подхватываются и обновляют индекс.

Лог: logs/update_index.log (JSONL).

Запуск:
  python update_index.py           # инкрементальное обновление (только docs/)
  python update_index.py --full-rebuild   # полная пересборка из knowledge_base (как build_index.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".cache" / "transformers"))

DOCS_DIR = PROJECT_ROOT / "docs"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
INDEX_STATE_PATH = PROJECT_ROOT / ".index_state.json"
CHUNKS_PATH = PROJECT_ROOT / "knowledge_base_chunks.json"
EMBEDDINGS_PATH = PROJECT_ROOT / "knowledge_base_embeddings.npy"
METADATA_PATH = PROJECT_ROOT / "knowledge_base_metadata.json"
FAISS_INDEX_PATH = PROJECT_ROOT / "knowledge_base_faiss.index"
LOGS_DIR = PROJECT_ROOT / "logs"
UPDATE_LOG_PATH = LOGS_DIR / "update_index.log"

# --- State tracking ---


def _file_md5(path: Path) -> str:
    """Вычисляет MD5-хеш содержимого файла."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path: Path | None = None) -> dict:
    """Загружает состояние индекса из .index_state.json или возвращает пустое."""
    path = path or INDEX_STATE_PATH
    if not path.is_file():
        return {
            "files": {},
            "knowledge_base_files": {},
            "docs_files": {},
            "last_update": None,
            "total_chunks": 0,
            "index_size": 0,
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "files" not in data:
        data["files"] = {}
    if "docs_files" not in data:
        data["docs_files"] = {}
    if "knowledge_base_files" not in data:
        data["knowledge_base_files"] = data.get("files", {})
    return data


def save_state(state: dict, path: Path | None = None) -> None:
    """Сохраняет состояние в .index_state.json."""
    path = path or INDEX_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def scan_docs_dir(kb_dir: Path) -> dict[str, dict]:
    """Сканирует директорию с .md файлами, возвращает {filename: {md5, mtime}}."""
    if not kb_dir.is_dir():
        return {}
    result = {}
    for md_file in sorted(kb_dir.glob("*.md")):
        try:
            content_hash = _file_md5(md_file)
            stat = md_file.stat()
            result[md_file.name] = {
                "md5": content_hash,
                "mtime": stat.st_mtime,
                "path": str(md_file),
            }
        except Exception:
            continue
    return result


def scan_changes(
    docs_dir: Path,
    state: dict,
) -> tuple[list[str], list[str], list[str]]:
    """
    Сравнивает текущее состояние docs/ с сохранённым state["docs_files"].
    Возвращает (new_files, modified_files, deleted_files) — списки имён файлов из docs.
    """
    current = scan_docs_dir(docs_dir)
    saved_files = state.get("docs_files", {})

    new_files = []
    modified_files = []
    deleted_files = []

    for name, info in current.items():
        if name not in saved_files:
            new_files.append(name)
        elif saved_files[name].get("md5") != info["md5"]:
            modified_files.append(name)

    for name in saved_files:
        if name not in current:
            deleted_files.append(name)

    return new_files, modified_files, deleted_files


# --- Chunk/embedding manipulation ---


def load_existing_artifacts(
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
) -> tuple[list[dict], list[dict], "np.ndarray"]:
    """Загружает существующие чанки, метаданные и эмбеддинги (если есть)."""
    import numpy as np

    if not chunks_path.is_file() or not embeddings_path.is_file() or not metadata_path.is_file():
        return [], [], np.array([], dtype=np.float32).reshape(0, 768)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
    embeddings = np.load(embeddings_path)

    dim = 768
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, dim)
    elif embeddings.shape[1] != dim:
        raise ValueError(f"Unexpected embedding dim {embeddings.shape[1]}, expected {dim}")

    return chunks, metadata_list, np.asarray(embeddings, dtype=np.float32)


def remove_chunks_for_files(
    chunks: list[dict],
    metadata_list: list[dict],
    embeddings: "np.ndarray",
    filenames: list[str],
) -> tuple[list[dict], list[dict], "np.ndarray"]:
    """Удаляет чанки, принадлежащие указанным source-файлам. Сохраняет порядок."""
    names_set = set(filenames)
    keep_indices = []
    for i, meta in enumerate(metadata_list):
        source = meta.get("source", "")
        if source not in names_set:
            keep_indices.append(i)

    new_chunks = [chunks[i] for i in keep_indices]
    new_metadata = [metadata_list[i] for i in keep_indices]
    new_embeddings = embeddings[keep_indices]
    return new_chunks, new_metadata, new_embeddings


def add_chunks_for_files(
    file_paths: list[Path],
    chunks: list[dict],
    metadata_list: list[dict],
    embeddings: "np.ndarray",
    model_name: str,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[dict], list[dict], "np.ndarray"]:
    """Загружает документы из file_paths, чанкирует, эмбеддит и добавляет к существующим данным."""
    import numpy as np
    from langchain_core.documents import Document
    from sentence_transformers import SentenceTransformer

    from chunk_knowledge_base import split_into_chunks, chunks_to_serializable
    from embed_chunks import build_metadata_list

    if not file_paths:
        return chunks, metadata_list, embeddings

    documents = []
    for p in file_paths:
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": p.name, "source_path": str(p)},
            )
        )

    if not documents:
        return chunks, metadata_list, embeddings

    new_chunks_docs = split_into_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    new_chunks_serialized = chunks_to_serializable(new_chunks_docs)
    new_metadata = build_metadata_list(new_chunks_serialized)

    texts = [item["content"] for item in new_chunks_serialized]
    cache_dir = PROJECT_ROOT / ".cache" / "sentence_transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    new_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    new_embeddings = np.asarray(new_embeddings, dtype=np.float32)

    # Переиндексация global_index в новых метаданных
    base_idx = len(metadata_list)
    for i, m in enumerate(new_metadata):
        m["global_index"] = base_idx + i

    combined_chunks = chunks + new_chunks_serialized
    combined_metadata = metadata_list + new_metadata
    combined_embeddings = np.vstack([embeddings, new_embeddings])

    return combined_chunks, combined_metadata, combined_embeddings


def rebuild_faiss(embeddings: "np.ndarray"):
    """Создаёт FAISS IndexFlatIP из матрицы эмбеддингов."""
    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def save_all(
    chunks: list[dict],
    metadata_list: list[dict],
    embeddings: "np.ndarray",
    index,
    state: dict,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    faiss_path: Path,
) -> None:
    """Сохраняет все артефакты индекса и состояние."""
    import numpy as np
    import faiss

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    faiss.write_index(index, str(faiss_path))
    save_state(state)


# --- Logging ---


def ensure_log_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def write_log_entry(entry: dict) -> None:
    """Добавляет одну JSON-строку в logs/update_index.log."""
    ensure_log_dir()
    with open(UPDATE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def setup_stdout_logging() -> None:
    """Настраивает вывод в stdout для cron."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


# --- Main ---


def run_full_rebuild(logger: logging.Logger) -> dict:
    """Полная пересборка индекса из knowledge_base (как build_index.py)."""
    from chunk_knowledge_base import run as run_chunk
    from embed_chunks import run as run_embed

    run_chunk(knowledge_base_dir=KNOWLEDGE_BASE_DIR, output_path=CHUNKS_PATH)
    run_embed(
        chunks_path=CHUNKS_PATH,
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
        faiss_path=FAISS_INDEX_PATH,
        save_faiss=True,
    )

    kb_current = scan_docs_dir(KNOWLEDGE_BASE_DIR)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    file_chunk_counts = {}
    for item in chunks:
        src = item["metadata"].get("source", "")
        if src not in file_chunk_counts:
            file_chunk_counts[src] = 0
        file_chunk_counts[src] += 1

    state = {
        "files": {
            name: {
                "md5": info["md5"],
                "mtime": info["mtime"],
                "n_chunks": file_chunk_counts.get(name, 0),
            }
            for name, info in kb_current.items()
        },
        "knowledge_base_files": {
            name: {"md5": info["md5"], "mtime": info["mtime"], "n_chunks": file_chunk_counts.get(name, 0)}
            for name, info in kb_current.items()
        },
        "docs_files": {},
        "last_update": datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
        "index_size": len(chunks),
    }
    save_state(state)

    return {
        "status": "success",
        "full_rebuild": True,
        "files_added": len(kb_current),
        "files_modified": 0,
        "files_deleted": 0,
        "new_chunks": len(chunks),
        "removed_chunks": 0,
        "total_chunks": len(chunks),
        "index_size": len(chunks),
        "errors": [],
    }


def run_incremental_update(logger: logging.Logger) -> dict:
    """Инкрементальное обновление: diff -> remove -> add -> rebuild -> save."""
    from chunk_knowledge_base import CHUNK_SIZE, CHUNK_OVERLAP
    from embed_chunks import BGE_MODEL_NAME, BATCH_SIZE

    state = load_state()
    new_names, modified_names, deleted_names = scan_changes(DOCS_DIR, state)

    if not new_names and not modified_names and not deleted_names:
        logger.info("No changes detected.")
        return {
            "status": "no_changes",
            "total_chunks": state.get("total_chunks", 0),
            "index_size": state.get("index_size", 0),
            "errors": [],
        }

    logger.info("Changes: added=%s modified=%s deleted=%s", new_names, modified_names, deleted_names)

    chunks, metadata_list, embeddings = load_existing_artifacts(
        CHUNKS_PATH, EMBEDDINGS_PATH, METADATA_PATH
    )
    initial_count = len(chunks)

    # Удалить чанки удалённых и изменённых файлов
    to_remove = list(deleted_names) + list(modified_names)
    if to_remove:
        chunks, metadata_list, embeddings = remove_chunks_for_files(
            chunks, metadata_list, embeddings, to_remove
        )
    removed_count = initial_count - len(chunks)

    # Добавить чанки новых и изменённых файлов
    to_add_paths = []
    for name in new_names:
        to_add_paths.append(DOCS_DIR / name)
    for name in modified_names:
        to_add_paths.append(DOCS_DIR / name)

    if to_add_paths:
        chunks, metadata_list, embeddings = add_chunks_for_files(
            to_add_paths,
            chunks,
            metadata_list,
            embeddings,
            model_name=BGE_MODEL_NAME,
            batch_size=BATCH_SIZE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    new_count = len(chunks) - (initial_count - removed_count)

    # Пересобрать FAISS и сохранить
    index = rebuild_faiss(embeddings)

    # Обновить state: knowledge_base без изменений, docs — актуальный скан
    current_docs = scan_docs_dir(DOCS_DIR)
    kb_files = state.get("knowledge_base_files", {})
    file_chunk_counts = {}
    for meta in metadata_list:
        src = meta.get("source", "")
        if src not in file_chunk_counts:
            file_chunk_counts[src] = 0
        file_chunk_counts[src] += 1

    state["docs_files"] = {
        name: {
            "md5": info["md5"],
            "mtime": info["mtime"],
            "n_chunks": file_chunk_counts.get(name, 0),
        }
        for name, info in current_docs.items()
    }
    state["files"] = {**kb_files, **state["docs_files"]}
    state["last_update"] = datetime.now(timezone.utc).isoformat()
    state["total_chunks"] = len(chunks)
    state["index_size"] = len(chunks)

    save_all(
        chunks,
        metadata_list,
        embeddings,
        index,
        state,
        CHUNKS_PATH,
        EMBEDDINGS_PATH,
        METADATA_PATH,
        FAISS_INDEX_PATH,
    )

    return {
        "status": "success",
        "files_added": len(new_names),
        "files_modified": len(modified_names),
        "files_deleted": len(deleted_names),
        "new_chunks": new_count,
        "removed_chunks": removed_count,
        "total_chunks": len(chunks),
        "index_size": len(chunks),
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update knowledge base vector index (incremental or full rebuild)")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Full rebuild from scratch (like build_index.py)",
    )
    args = parser.parse_args()

    setup_stdout_logging()
    logger = logging.getLogger(__name__)

    start_time = time.perf_counter()
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
        "duration_seconds": 0,
        "errors": [],
    }

    try:
        if args.full_rebuild or not INDEX_STATE_PATH.is_file():
            if not KNOWLEDGE_BASE_DIR.is_dir():
                raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
            logger.info("Running full rebuild from knowledge_base.")
            result = run_full_rebuild(logger)
        else:
            if not DOCS_DIR.is_dir():
                raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")
            result = run_incremental_update(logger)

        duration = round(time.perf_counter() - start_time, 2)
        log_entry["duration_seconds"] = duration
        log_entry.update(result)
        if "duration_seconds" not in log_entry:
            log_entry["duration_seconds"] = duration

        write_log_entry(log_entry)

        if result.get("status") == "success":
            logger.info(
                "Index updated: total_chunks=%s, duration=%ss",
                result.get("total_chunks", 0),
                duration,
            )
        elif result.get("status") == "no_changes":
            logger.info("No changes. Index size=%s", result.get("index_size", 0))

        return 0

    except Exception as e:
        duration = round(time.perf_counter() - start_time, 2)
        log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_entry["status"] = "error"
        log_entry["error"] = f"{type(e).__name__}: {e}"
        log_entry["duration_seconds"] = duration
        write_log_entry(log_entry)
        logger.exception("Update failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
