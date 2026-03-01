#!/usr/bin/env python3
"""
Replace all Star Wars terms in source files with fictional terms from terms_map.json.
Preserves original case when replacing.

Default mode: reads raw_pages/ -> writes knowledge_base/
Update mode (--update): reads docs/ -> writes docs/ in-place (no knowledge_base pipeline)
"""
import argparse
import json
import re
from pathlib import Path

TERMS_MAP_PATH = Path(__file__).resolve().parent / "terms_map.json"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_pages"
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def slugify(name: str) -> str:
    """Convert to filename-safe slug (lowercase, spaces -> underscores)."""
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s or "page"


def title_from_slug(slug: str) -> str:
    """Convert filename slug back to title (e.g. luke_skywalker -> Luke Skywalker)."""
    return slug.replace("_", " ").title()


def replace_preserve_case(text: str, old: str, new: str) -> tuple[str, int]:
    """Replace all occurrences of old with new, preserving case of first letter. Returns (new_text, count)."""
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        t = m.group(0)
        if len(t) > 0 and t[0].isupper():
            return new[:1].upper() + new[1:] if len(new) > 1 else new.upper()
        return new.lower() if new else ""

    new_text = pattern.sub(repl, text)
    return new_text, count


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace Star Wars terms with fictional ones.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Read docs/ and write back to docs/ in-place (skips knowledge_base pipeline).",
    )
    args = parser.parse_args()

    with open(TERMS_MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    terms = data.get("terms", {})
    if not terms:
        raise SystemExit("terms_map.json has no 'terms'.")

    # Sort by length descending so "Darth Vader" is replaced before "Darth"
    sorted_pairs = sorted(terms.items(), key=lambda x: -len(x[0]))

    if args.update:
        input_dir = DOCS_DIR
        output_dir = DOCS_DIR
    else:
        input_dir = RAW_DIR
        output_dir = KNOWLEDGE_BASE_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_files = list(input_dir.glob("*.md"))
    if not raw_files:
        raise SystemExit(f"No .md files in {input_dir}. Run 1_fetch_pages.py{'  --update' if args.update else ''} first.")

    print(f"Replacing terms in {len(raw_files)} files from {input_dir} -> {output_dir}")
    total_replacements = 0

    for path in sorted(raw_files):
        text = path.read_text(encoding="utf-8")
        file_replacements = 0
        for old_term, new_term in sorted_pairs:
            text, n = replace_preserve_case(text, old_term, new_term)
            file_replacements += n
        total_replacements += file_replacements

        if args.update:
            # In update mode: overwrite in-place, keep original filename
            out_path = output_dir / path.name
        else:
            # Default mode: rename file using term replacement map
            stem = path.stem
            original_title = title_from_slug(stem)
            new_name = terms.get(original_title, original_title)
            new_slug = slugify(new_name)
            out_path = output_dir / f"{new_slug}.md"

        out_path.write_text(text, encoding="utf-8")
        print(f"  {path.name} -> {out_path.name}  ({file_replacements} replacements)")

    print(f"Done. Total replacements: {total_replacements}")


if __name__ == "__main__":
    main()
