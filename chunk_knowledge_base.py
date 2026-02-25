from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Директория с исходными .md и выходной файл чанков
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
CHUNKS_OUTPUT_PATH = Path(__file__).resolve().parent / "knowledge_base_chunks.json"

# ~250 слов ≈ 1500 символов; 500–1000 токенов ≈ 2000–4000 символов (≈4 символа/токен)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def _find_chunk_position(source_text: str, chunk_text: str) -> tuple[int, int] | None:
    """Находит позицию чанка в исходном тексте (start_char, end_char)."""
    # Нормализуем пробелы для надёжного поиска
    chunk_stripped = chunk_text.strip()
    if not chunk_stripped:
        return None
    # Ищем по началу чанка (первые 100 символов достаточно для уникальности)
    search_key = chunk_stripped[: min(100, len(chunk_stripped))]
    start = source_text.find(search_key)
    if start == -1:
        return None
    end = start + len(chunk_stripped)
    return (start, end)


def load_documents_from_dir(dir_path: Path) -> list[Document]:
    """Загружает все .md файлы из директории как LangChain Document с метаданными."""
    documents: list[Document] = []
    for md_file in sorted(dir_path.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: skip {md_file.name}: {e}")
            continue
        if not text.strip():
            continue
        doc = Document(
            page_content=text,
            metadata={
                "source": md_file.name,
                "source_path": str(md_file),
            },
        )
        documents.append(doc)
    return documents


def split_into_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Разбивает документы на чанки с помощью RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )
    all_chunks: list[Document] = []
    for doc in documents:
        chunks = splitter.split_documents([doc])
        source = doc.metadata.get("source", "unknown")
        source_path = doc.metadata.get("source_path", "")
        source_text = doc.page_content
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = source
            chunk.metadata["source_path"] = source_path
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            pos = _find_chunk_position(source_text, chunk.page_content)
            if pos:
                chunk.metadata["start_char"] = pos[0]
                chunk.metadata["end_char"] = pos[1]
            all_chunks.append(chunk)
    return all_chunks


def chunks_to_serializable(chunks: list[Document]) -> list[dict]:
    """Преобразует чанки в список словарей для JSON (int/str в metadata)."""
    out = []
    for c in chunks:
        meta = dict(c.metadata)
        for k, v in meta.items():
            if hasattr(v, "item"):  # numpy etc
                meta[k] = int(v) if isinstance(v, (int, float)) else str(v)
            elif not isinstance(v, (str, int, float, bool, type(None))):
                meta[k] = str(v)
        out.append({"content": c.page_content, "metadata": meta})
    return out


def run(
    knowledge_base_dir: Path | None = None,
    output_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Загружает документы, разбивает на чанки и сохраняет в JSON."""
    knowledge_base_dir = knowledge_base_dir or KNOWLEDGE_BASE_DIR
    output_path = output_path or CHUNKS_OUTPUT_PATH

    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {knowledge_base_dir}")

    documents = load_documents_from_dir(knowledge_base_dir)
    if not documents:
        print("No .md documents found.")
        return []

    chunks = split_into_chunks(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            chunks_to_serializable(chunks),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {len(chunks)} chunks to {output_path}")
    return chunks


if __name__ == "__main__":
    run()
