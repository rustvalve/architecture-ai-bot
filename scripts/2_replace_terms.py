#!/usr/bin/env python3
"""
Replace all Star Wars terms in raw_pages/ with fictional terms from terms_map.json
and save results to knowledge_base/. Preserves original case when replacing.
"""
import json
import re
from pathlib import Path

TERMS_MAP_PATH = Path(__file__).resolve().parent / "terms_map.json"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_pages"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"


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
    with open(TERMS_MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    terms = data.get("terms", {})
    if not terms:
        raise SystemExit("terms_map.json has no 'terms'.")

    # Sort by length descending so "Darth Vader" is replaced before "Darth"
    sorted_pairs = sorted(terms.items(), key=lambda x: -len(x[0]))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.md"))
    if not raw_files:
        raise SystemExit(f"No .md files in {RAW_DIR}. Run 1_fetch_pages.py first.")

    print(f"Replacing terms in {len(raw_files)} files -> {OUTPUT_DIR}")
    total_replacements = 0

    for path in sorted(raw_files):
        text = path.read_text(encoding="utf-8")
        file_replacements = 0
        for old_term, new_term in sorted_pairs:
            text, n = replace_preserve_case(text, old_term, new_term)
            file_replacements += n
        total_replacements += file_replacements

        # Output filename: use new name for this entity (original page title -> term replacement)
        stem = path.stem
        original_title = title_from_slug(stem)
        new_name = terms.get(original_title, original_title)
        new_slug = slugify(new_name)
        out_path = OUTPUT_DIR / f"{new_slug}.md"
        out_path.write_text(text, encoding="utf-8")
        print(f"  {path.name} -> {out_path.name}  ({file_replacements} replacements)")

    print(f"Done. Total replacements: {total_replacements}")


if __name__ == "__main__":
    main()
