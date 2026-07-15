#!/usr/bin/env python3
"""
Comprehensive markdown parsing debugger.
Inspects file contents, encoding, headers, and parsing logic.
"""

import re
from pathlib import Path

# ============================================================================
# 1. FILE DISCOVERY & BASIC INSPECTION
# ============================================================================

print("\n" + "=" * 80)
print("1. FILE DISCOVERY & BASIC INSPECTION")
print("=" * 80)

CURATED_DIR = Path("data/curated")
print(f"\nLooking for markdown files in: {CURATED_DIR.absolute()}")
print(f"Directory exists: {CURATED_DIR.exists()}")
print(f"Is directory: {CURATED_DIR.is_dir()}")

md_files = list(CURATED_DIR.glob("*.md"))
print(f"Found {len(md_files)} markdown files:")
for f in md_files:
    print(f"  - {f} ({f.stat().st_size} bytes)")

if not md_files:
    print("\n⚠️  WARNING: No markdown files found! Checking for markdown elsewhere...")
    for search_dir in [Path("docs"), Path("docs/curated"), Path(".")]:
        found = list(search_dir.glob("*.md")) if search_dir.exists() else []
        if found:
            print(f"\nFound markdown in {search_dir}:")
            for f in found[:5]:
                print(f"  - {f}")

# ============================================================================
# 2. FILE READING & ENCODING INSPECTION
# ============================================================================

print("\n" + "=" * 80)
print("2. FILE READING & ENCODING INSPECTION")
print("=" * 80)

for md_file in md_files:
    print(f"\n------- {md_file.name} -------")

    # Read raw bytes
    with open(md_file, "rb") as f:
        raw_bytes = f.read()

    print(f"File size: {len(raw_bytes)} bytes")
    print(f"First 50 bytes (hex): {raw_bytes[:50].hex()}")

    # Check for BOM
    if raw_bytes.startswith(b'\xef\xbb\xbf'):
        print("✓ UTF-8 BOM detected")
    elif raw_bytes.startswith(b'\xff\xfe'):
        print("⚠️  UTF-16-LE BOM detected")
    elif raw_bytes.startswith(b'\xfe\xff'):
        print("⚠️  UTF-16-BE BOM detected")
    else:
        print("✓ No BOM detected")

    # Try different encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "ascii"]
    text = None
    successful_encoding = None

    for enc in encodings_to_try:
        try:
            text = raw_bytes.decode(enc)
            successful_encoding = enc
            print(f"✓ Successfully decoded with: {enc}")
            break
        except UnicodeDecodeError as e:
            print(f"✗ Failed with {enc}: {e}")

    if not text:
        print("✗ Could not decode file with any encoding!")
        continue

    # ========================================================================
    # 3. HEADER DETECTION & CHARACTER INSPECTION
    # ========================================================================

    print("\n--- Character-level inspection ---")

    lines = text.splitlines()
    print(f"Total lines: {len(lines)}")
    print(f"Total characters: {len(text)}")
    print(f"Is empty: {len(text.strip()) == 0}")

    # Find all lines that might be headers
    print("\n--- Potential header lines ---")
    header_patterns = [
        (r"^#{1,6}\s+", "Markdown header (# to ######)"),
        (r"^#{1,6}(?!\s)", "Header without space after #"),
        (r"^\s+#{1,6}\s+", "Indented header"),
    ]

    for pattern_str, description in header_patterns:
        pattern = re.compile(pattern_str)
        matches = [(i, line) for i, line in enumerate(lines) if pattern.match(line)]
        if matches:
            print(f"\n{description}:")
            for line_num, line in matches:
                print(f"  Line {line_num}: {repr(line[:60])}")

    # Specific check for ## and ###
    print("\n--- Specific header searches ---")
    h2_lines = [line for line in lines if line.startswith("## ")]
    h3_lines = [line for line in lines if line.startswith("### ")]
    h2_no_space = [line for line in lines if line.startswith("##") and not line.startswith("## ")]
    h3_no_space = [line for line in lines if line.startswith("###") and not line.startswith("### ")]

    print(f"Lines starting with '## ' (with space): {len(h2_lines)}")
    print(f"Lines starting with '### ' (with space): {len(h3_lines)}")
    print(f"Lines starting with '##' (no space): {len(h2_no_space)}")
    print(f"Lines starting with '###' (no space): {len(h3_no_space)}")

    if h2_no_space:
        print(f"  Examples: {h2_no_space[:3]}")
    if h3_no_space:
        print(f"  Examples: {h3_no_space[:3]}")

    # Show first 20 lines
    print("\n--- First 20 lines (raw) ---")
    for i, line in enumerate(lines[:20]):
        print(f"{i:3d}: {repr(line)}")

# ============================================================================
# 4. PARSING LOGIC TEST
# ============================================================================

print("\n" + "=" * 80)
print("3. PARSING LOGIC TEST")
print("=" * 80)

def split_markdown_sections_original(text: str):
    """Original parsing function"""
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

    # First pass: look for ###
    has_h3 = any(line.startswith("### ") for line in lines)

    for line in lines:
        if has_h3 and line.startswith("### "):
            flush_section()
            current_title = line.replace("### ", "").strip()
            current_body = []
        elif not has_h3 and line.startswith("## "):
            flush_section()
            current_title = line.replace("## ", "").strip()
            current_body = []
        else:
            if current_title:
                current_body.append(line)

    flush_section()
    return sections

def split_markdown_sections_robust(text: str):
    """More robust parsing function"""
    lines = text.splitlines()
    sections = []
    current_title = None
    current_body = []

    # First pass: detect what headers we have
    h3_count = sum(1 for line in lines if re.match(r"^###\s+", line))

    use_h3 = h3_count > 0

    for line in lines:
        # Match headers at the target depth
        if use_h3:
            match = re.match(r"^###\s+(.+)$", line)
        else:
            match = re.match(r"^##\s+(.+)$", line)

        if match:
            # Flush previous section
            if current_title and current_body:
                sections.append(
                    (current_title, "\n".join(current_body).strip())
                )
            current_title = match.group(1)
            current_body = []
        else:
            if current_title is not None:
                current_body.append(line)

    # Don't forget the last section
    if current_title and current_body:
        sections.append(
            (current_title, "\n".join(current_body).strip())
        )

    return sections

# Test both functions
for md_file in md_files:
    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\n--- Testing {md_file.name} ---")

    sections_orig = split_markdown_sections_original(text)
    sections_robust = split_markdown_sections_robust(text)

    print(f"Original parser: {len(sections_orig)} sections")
    for title, content in sections_orig[:3]:
        print(f"  - {title[:50]}: {len(content)} chars")

    print(f"Robust parser: {len(sections_robust)} sections")
    for title, content in sections_robust[:3]:
        print(f"  - {title[:50]}: {len(content)} chars")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)
