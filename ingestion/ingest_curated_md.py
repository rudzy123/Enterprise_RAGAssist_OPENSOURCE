import re
from pathlib import Path
from typing import Optional

from core.config import CHUNK_OVERLAP_WORDS, CHUNK_WORD_LIMIT, CURATED_DOCS_DIR

def split_text_into_chunks(text: str, max_words: int = CHUNK_WORD_LIMIT, overlap_words: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        chunks.append(chunk_text)
        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def split_markdown_sections(text: str):
    """
    Splits markdown text into sections.
    Prefers ### sections; falls back to ## sections.
    Returns a list of (section_title, section_text).
    
    Robust to:
    - Files with only ## headers
    - Files with only ### headers
    - Mixed ## and ### (prefers ###)
    - Whitespace variations
    """
    lines = text.splitlines()
    sections = []
    current_title = None
    current_body = []

    def flush_section():
        nonlocal current_title, current_body
        if current_title and current_body:
            sections.append(
                (current_title, "\n".join(current_body).strip())
            )
            current_title = None
            current_body = []

    # Detect header level: prefer ### if present, else ##
    has_h3 = any(re.match(r"^###\s+", line) for line in lines)
    has_h2 = any(re.match(r"^##\s+", line) for line in lines)

    if not has_h3 and not has_h2:
        # No headers found, treat entire file as one section
        return [("Content", text)]

    for line in lines:
        # Match headers with regex (more robust than startswith)
        if has_h3 and re.match(r"^###\s+", line):
            flush_section()
            current_title = re.sub(r"^###\s+", "", line).strip()
            current_body = []
        elif not has_h3 and re.match(r"^##\s+", line):
            flush_section()
            current_title = re.sub(r"^##\s+", "", line).strip()
            current_body = []
        else:
            if current_title is not None:
                current_body.append(line)

    flush_section()
    return sections


def ingest_curated_markdown(docs_dir: Optional[Path] = None):
    all_chunks = []
    curated_dir = docs_dir or CURATED_DOCS_DIR

    for md_file in curated_dir.glob("*.md"):
        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        sections = split_markdown_sections(text)

        for idx, (title, content) in enumerate(sections):
            chunks = split_text_into_chunks(content)
            for part_index, chunk_text in enumerate(chunks, start=1):
                section_title = title
                if len(chunks) > 1:
                    section_title = f"{title} (part {part_index}/{len(chunks)})"

                chunk = {
                    "source_file": md_file.name,
                    "section_title": section_title,
                    "text": chunk_text,
                    "chunk_id": f"{md_file.stem}_{idx}_{part_index}"
                }
                all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    chunks = ingest_curated_markdown()

    print(f"\nIngested {len(chunks)} chunks:\n")

    for chunk in chunks:
        print("=" * 80)
        print(f"Source:  {chunk['source_file']}")
        print(f"Section: {chunk['section_title']}")
        print("-" * 80)
        print(chunk["text"][:500])  # preview only
        print()
