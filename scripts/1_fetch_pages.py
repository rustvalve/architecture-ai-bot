#!/usr/bin/env python3
"""
Download pages from Wookieepedia (Star Wars Fandom wiki), strip wiki markup,
and save as plain text. One file per entity.

Default mode: fetches PAGES -> raw_pages/
Update mode (--update): fetches PAGES_UPDATE -> docs/  (skips knowledge_base pipeline)
"""
import argparse
import re
import time
from pathlib import Path

import requests

API_URL = "https://starwars.fandom.com/api.php"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw_pages"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DELAY_SECONDS = 1

# 35+ page titles to fetch (characters, planets, tech, organizations, races, concepts)
PAGES = [
    "Luke Skywalker",
    "Darth Vader",
    "Obi-Wan Kenobi",
    "Yoda",
    "Emperor Palpatine",
    "Han Solo",
    "Leia Organa",
    "Chewbacca",
    "Boba Fett",
    "Darth Maul",
    "Anakin Skywalker",
    "Mace Windu",
    "Count Dooku",
    "Padmé Amidala",
    "Ahsoka Tano",
    "Tatooine",
    "Coruscant",
    "Hoth",
    "Endor",
    "Naboo",
    "Mustafar",
    "Dagobah",
    "Alderaan",
    "Death Star",
    "Lightsaber",
    "Millennium Falcon",
    "X-wing",
    "TIE fighter",
    "Galactic Empire",
    "Rebel Alliance",
    "Jedi Order",
    "Sith",
    "Wookiee",
    "Ewok",
    "Twi'lek",
    "Hutt",
    "The Force",
    "Star Destroyer",
]

# Pages to fetch in update mode (--update). Results go to docs/, NOT raw_pages/.
# These pages are meant to be served as-is and do NOT enter the knowledge_base pipeline.
PAGES_UPDATE = [
    "Mandalorian",
]


def slugify(title: str) -> str:
    """Convert page title to filename-safe slug (lowercase, spaces -> underscores)."""
    s = title.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s)
    return s or "page"


def strip_wikitext(wikitext: str) -> str:
    """Simple wiki markup to plain text: templates, links, bold/italic, headers."""
    text = wikitext
    # Remove {{template}} blocks (multiline, non-greedy)
    for _ in range(15):
        prev = text
        text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
        if text == prev:
            break
    # Remove orphan {{ or }} and lines that are only template params (|key=value)
    text = re.sub(r"^\s*\{\{?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\}\}?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|[^=]*=.*$", "", text, flags=re.MULTILINE)
    # Remove any remaining template braces (orphan }} or {{ in the middle of text)
    text = text.replace("}}", "")
    text = text.replace("{{", "")
    # External links [http(s)://... optional text] -> remove or leave URL only
    text = re.sub(r"\[https?://[^\]\s]+(?:\s+[^\]]*)?\]", "", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)
    # [[Link|display]] -> display, [[Link]] -> Link
    text = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    # '''bold''' and ''italic''
    text = re.sub(r"'{2,}([^']*)'{2,}", r"\1", text)
    # = Header = -> Header
    text = re.sub(r"^=+\s*([^=]*)\s*=+", r"\1", text, flags=re.MULTILINE)
    # Remove HTML tags and refs
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Horizontal line, file/image, thumb captions (thumb|left|200px|Caption)
    text = re.sub(r"----+", "", text)
    text = re.sub(r"\[\[File:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Image:[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"thumb\|[^|]*\|[^|]*\|([^\]|]*)", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*thumb\|.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Standalone list markers that are template debris (* or ** at line start with nothing after)
    text = re.sub(r"^\s*\*+\s*$", "", text, flags=re.MULTILINE)
    # Normalize whitespace and drop empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


def fetch_page_plain_text(title: str) -> str | None:
    """Fetch page from Wookieepedia API and return plain text (or None on error)."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print(f"  API error for '{title}': {data['error'].get('info', data['error'])}")
            return None
        parse = data.get("parse", {})
        wikitext = parse.get("wikitext")
        if isinstance(wikitext, dict):
            wikitext = wikitext.get("*", "")
        if not wikitext:
            print(f"  No wikitext for '{title}'")
            return None
        return strip_wikitext(wikitext)
    except requests.RequestException as e:
        print(f"  Request error for '{title}': {e}")
        return None
    except Exception as e:
        print(f"  Error for '{title}': {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch pages from Wookieepedia.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Fetch PAGES_UPDATE and save to docs/ instead of raw_pages/.",
    )
    args = parser.parse_args()

    if args.update:
        pages = PAGES_UPDATE
        output_dir = DOCS_DIR
    else:
        pages = PAGES
        output_dir = RAW_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(pages)} pages from Wookieepedia -> {output_dir}")
    success = 0
    for i, title in enumerate(pages, 1):
        print(f"[{i}/{len(pages)}] {title}")
        text = fetch_page_plain_text(title)
        if text:
            path = output_dir / f"{slugify(title)}.md"
            path.write_text(text, encoding="utf-8")
            success += 1
        time.sleep(DELAY_SECONDS)
    print(f"Done. Saved {success}/{len(pages)} pages.")


if __name__ == "__main__":
    main()
