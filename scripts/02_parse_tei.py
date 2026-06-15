#!/usr/bin/env python3
"""
scripts/02_parse_tei.py

Parse Perseus TEI XML source texts into clean, book-and-chapter-segmented
Latin text.  Returns an in-memory list of Chapter dataclass instances
ready for CSV export (Step 1.3).

Expected input:
    data/raw/perseus/caes_bg_lat.xml  — De Bello Gallico (Books I-VIII)
    data/raw/perseus/caes.bc_lat.xml  — De Bello Civili

Expected output (in-memory; summary printed to stdout):
    DBG: ~400 chapters across Books I-VIII (including Hirtius)
    DBC: ~243 chapters across 3 books

The two files use DIFFERENT TEI structures — confirmed from samples:

    DBG (caes_bg_lat.xml):
        <div1 type="Book" n="1">              ← book boundary (capital B)
            <head>COMMENTARIUS PRIMUS</head>
            <p>
                <milestone n="1" unit="chapter"/>  ← chapter boundary
                <milestone n="1" unit="section"/>
                Gallia est omnis divisa...
            </p>
            <p><milestone n="2" unit="chapter"/>... </p>
        </div1>
        → Each chapter = one <p> starting with <milestone unit="chapter">

    DBC (caes.bc_lat.xml):
        <div1 type="book" n="1">              ← book boundary (lowercase)
            <head>... Liber Primus</head>
            <div2 type="chapter" n="1">        ← chapter boundary
                <milestone n="1" unit="section"/>
                <p>
                    <gap /> Litteris
                    <del status="unremarkable">a Fabio</del>
                    C. Caesaris consulibus redditis...
                </p>
            </div2>
            <div2 type="chapter" n="2">...</div2>
        </div1>
        → Each chapter = one <div2 type="chapter">

Strip policy (revised — preserves all tail text):
    - <del> : TEXT CONTENT cleared (e.g., "a Fabio" removed),
              element + tail text KEPT (e.g., "C. Caesaris consulibus..."
              survives).  No node removal — removal loses tail text
              regardless of lxml.strip_elements() flags.
    - <note> : Same treatment — text content cleared, tail preserved.
    - <gap /> : LEFT IN TREE.  Self-closing, no text; itertext() skips it
                naturally; tail text (" Litteris ") survives unimpeded.
    - <milestone/> : LEFT IN TREE.  Same logic — no text, tail preserved.
    - <add> : KEPT in full (editorially reconstructed text).
    - <head> : SKIPPED (book title, not part of running text).
    - All other XML tags : tags stripped by itertext(), text kept.

Validation:
    - Prints per-book chapter counts
    - Warns if total differs from expected (400 DBG / 243 DBC)
    - Prints first 200 chars of first 3 DBG + first DBC chapter for inspection
"""

import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from lxml import etree


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chapter:
    """A single chapter of clean, segmented Latin text."""
    work: str          # "dbg" or "dbc"
    book: int          # book number (1-indexed)
    chapter: int       # chapter number within the book (1-indexed)
    segment_id: str    # e.g., "dbg_book01_ch001"
    text: str          # clean running Latin text (no XML)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_int(value: str) -> int:
    """
    Extract the first integer from an attribute string like '55' or '55(56)'.

    Perseus XML occasionally uses compound numbering like n="55(56)"
    when chapter numbering differs across editions.  We take the first
    integer as the canonical chapter number.
    """
    match = re.search(r'\d+', value)
    if match:
        return int(match.group())
    raise ValueError(
        f"Cannot extract an integer from attribute value: {value!r}"
    )


def clean_text(elem: etree._Element) -> str:
    """
    Extract clean running text from an lxml Element subtree.

    CRITICAL: We MUST NOT remove <del> or <note> elements from the tree.
    Removing them — even with lxml.strip_elements(..., with_tail=True) —
    can drop their tail text.  The tail of <del> is real Latin prose that
    follows the editorial deletion; losing it silently corrupts the text.

    Instead we clear the text content of <del> and <note> (removing the
    editorial material like "a Fabio") while leaving the empty element
    and its tail text in place.  itertext() then naturally skips the
    empty text field but finds the tail — exactly what we want.

    <gap/> and <milestone/> are self-closing empty elements; itertext()
    skips them automatically, and their tail text survives unimpeded.
    """
    elem_copy = deepcopy(elem)

    # Clear text content of editorial-suppression tags.
    # We empty the element's .text and remove all its children, but leave
    # the element itself (and its .tail) in the tree.
    for tag_name in ('del', 'note'):
        for bad in elem_copy.findall(f'.//{tag_name}'):
            bad.text = ''               # e.g., "a Fabio" → ""
            for child in list(bad):
                bad.remove(child)       # remove any nested children

    # itertext() walks the tree in document order: for each element it
    # yields element.text, then recursively iterates children, then yields
    # element.tail.  Because <del> and <note> now have .text == '' and no
    # children, they contribute nothing — but their .tail (real Latin text)
    # is still emitted when the iterator reaches their parent.
    text = ''.join(elem_copy.itertext())

    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Parser: De Bello Gallico (milestone-based chapters)
# ---------------------------------------------------------------------------

def parse_dbg(xml_path: str) -> List[Chapter]:
    """
    Parse De Bello Gallico from caes_bg_lat.xml.

    Structure confirmed from sample:
      - Books are <div1 type="Book" n="1"> (capital B).
      - Chapters are marked by <milestone unit="chapter" n="1"/> inside <p>.
      - Each chapter occupies exactly one <p> element; the chapter milestone
        is the first child.
      - Book VIII by Aulus Hirtius is included.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    chapters: List[Chapter] = []

    # DBG uses type="Book" (capital B)
    book_divs = root.xpath("//div1[@type='Book']")

    if not book_divs:
        # Fallback: some editions use lowercase
        book_divs = root.xpath("//div1[@type='book']")

    if not book_divs:
        print("ERROR: No <div1 type='Book'> or <div1 type='book'> elements "
              "found in DBG XML.  Cannot parse.")
        sys.exit(1)

    print(f"  Found {len(book_divs)} book <div1> elements in DBG.")

    for book_div in book_divs:
        book_num = safe_int(book_div.get('n', '0'))

        # Each chapter is a <p> whose first child is <milestone unit="chapter">
        for p in book_div.findall('.//p'):
            chapter_ms = p.find("milestone[@unit='chapter']")
            if chapter_ms is None:
                # Paragraph without a chapter milestone — very rare in DBG
                # (could be a continuation or editorial note).  Skip.
                continue

            chapter_num = safe_int(chapter_ms.get('n', '0'))
            text = clean_text(p)

            if not text:
                continue

            segment_id = f'dbg_book{book_num:02d}_ch{chapter_num:03d}'

            chapters.append(Chapter(
                work='dbg',
                book=book_num,
                chapter=chapter_num,
                segment_id=segment_id,
                text=text,
            ))

    return chapters


# ---------------------------------------------------------------------------
# Parser: De Bello Civili (div2-based chapters)
# ---------------------------------------------------------------------------

def parse_dbc(xml_path: str) -> List[Chapter]:
    """
    Parse De Bello Civili from caes.bc_lat.xml.

    Structure confirmed from sample:
      - Books are <div1 type="book" n="1"> (lowercase).
      - Chapters are <div2 type="chapter" n="1">.
      - Each <div2> wraps one or more <p> elements; milestones mark sections.
      - Some chapter n-attributes use compound numbering like n="55(56)"
        reflecting cross-edition discrepancies; safe_int() handles this.
    """
    tree = etree.parse(xml_path)
    root = tree.getroot()

    chapters: List[Chapter] = []

    # DBC uses type="book" (lowercase)
    book_divs = root.xpath("//div1[@type='book']")

    if not book_divs:
        # Fallback: some mirrored editions use capital
        book_divs = root.xpath("//div1[@type='Book']")

    if not book_divs:
        print("ERROR: No <div1 type='book'> or <div1 type='Book'> elements "
              "found in DBC XML.  Cannot parse.")
        sys.exit(1)

    print(f"  Found {len(book_divs)} book <div1> elements in DBC.")

    for book_div in book_divs:
        book_num = safe_int(book_div.get('n', '0'))

        for ch_div in book_div.findall("div2[@type='chapter']"):
            chapter_num = safe_int(ch_div.get('n', '0'))
            text = clean_text(ch_div)

            if not text:
                continue

            segment_id = f'dbc_book{book_num:02d}_ch{chapter_num:03d}'

            chapters.append(Chapter(
                work='dbc',
                book=book_num,
                chapter=chapter_num,
                segment_id=segment_id,
                text=text,
            ))

    return chapters


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

EXPECTED_DBG_CHAPTERS = 400   # Books I-VIII (incl. Hirtius Book VIII)
EXPECTED_DBC_CHAPTERS = 243   # Books I-III


def summarise(chapters: List[Chapter], label: str, expected: int) -> bool:
    """
    Print per-book chapter counts, compare total to expected, return
    True if the count is within a small tolerance.
    """
    actual = len(chapters)

    # Group by book
    book_counts: dict = {}
    for ch in chapters:
        book_counts[ch.book] = book_counts.get(ch.book, 0) + 1

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total chapters parsed: {actual}  (expected ~{expected})")
    print(f"  Books found: {len(book_counts)}")
    for bk in sorted(book_counts):
        print(f"    Book {bk}: {book_counts[bk]:4d} chapters")
    print()

    if actual == expected:
        print(f"  ✓  Count matches expected exactly.")
        return True
    elif abs(actual - expected) <= 5:
        print(f"  ⚠  Count is off by {actual - expected:+d} — likely an edition "
              f"or chapter-boundary difference.  Verify manually if needed.")
        return True   # close enough — warn but don't fail
    else:
        print(f"  ✗  Count differs by {actual - expected:+d}.  This is large "
              f"enough to warrant checking the XML structure and parsing logic.")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Resolve project root relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)   # one level up from scripts/

    dbg_path = os.path.join(project_root, 'data', 'raw', 'perseus',
                            'caes_bg_lat.xml')
    dbc_path = os.path.join(project_root, 'data', 'raw', 'perseus',
                            'caes.bc_lat.xml')

    # ------------------------------------------------------------------
    # De Bello Gallico
    # ------------------------------------------------------------------
    if not os.path.exists(dbg_path):
        print(f"ERROR: DBG XML not found at:\n  {dbg_path}")
        sys.exit(1)

    print(f"Parsing De Bello Gallico …")
    print(f"  Source: {dbg_path}")
    dbg_chapters = parse_dbg(dbg_path)
    dbg_ok = summarise(dbg_chapters,
                       "De Bello Gallico (Books I-VIII, incl. Hirtius)",
                       EXPECTED_DBG_CHAPTERS)

    # ------------------------------------------------------------------
    # De Bello Civili
    # ------------------------------------------------------------------
    if not os.path.exists(dbc_path):
        print(f"ERROR: DBC XML not found at:\n  {dbc_path}")
        sys.exit(1)

    print(f"Parsing De Bello Civili …")
    print(f"  Source: {dbc_path}")
    dbc_chapters = parse_dbc(dbc_path)
    dbc_ok = summarise(dbc_chapters,
                       "De Bello Civili (Books I-III)",
                       EXPECTED_DBC_CHAPTERS)

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    all_chapters = dbg_chapters + dbc_chapters
    print(f"{'='*60}")
    print(f"  GRAND TOTAL: {len(all_chapters)} chapters "
          f"({len(dbg_chapters)} DBG + {len(dbc_chapters)} DBC)")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Spot-check: print first paragraphs for manual inspection
    # ------------------------------------------------------------------
    print(f"\n--- Spot-check: first 200 characters of first chapters ---\n")
    for ch in all_chapters[:4]:
        preview = ch.text[:200].replace('\n', ' ')
        print(f"  [{ch.segment_id}]")
        print(f"  {preview}…\n")

    # Print first DBC chapter — should now read:
    # "Litteris C. Caesaris consulibus redditis aegre ab his impetratum est…"
    dbc_first = next((ch for ch in all_chapters if ch.work == 'dbc'), None)
    if dbc_first:
        preview = dbc_first.text[:200].replace('\n', ' ')
        print(f"  [{dbc_first.segment_id}]")
        print(f"  {preview}…\n")

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------
    if not (dbg_ok and dbc_ok):
        print("⚠  WARNING: One or both chapter counts differ from expected.")
        print("   This may be due to edition differences.  Inspect the")
        print("   per-book breakdown above before proceeding to Step 1.3.")
    else:
        print("✓  Both parses completed successfully.")

    print(f"\n{len(all_chapters)} Chapter objects are in memory.")
    print("Next step: scripts/03_export_corpus.py  (serialize to CSV)")

    return all_chapters


if __name__ == '__main__':
    main()
