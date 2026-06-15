#!/usr/bin/env python3
"""
scripts/03_build_corpus.py

Build CSV corpora from the parsed TEI-XML data produced by 02_parse_tei.py.

Produces two files in data/corpus/:

  (1) corpus_chapters.csv
      Columns: work, book, chapter, segment_id, text
      One row per chapter — ~644 rows (401 DBG + 243 DBC).

  (2) corpus_books.csv
      Columns: work, book, segment_id, text
      One row per book-level segment — 9 rows total:
        DBG Book I  … DBG Book VIII  (dbg, books 1–8)
        DBC Complete                  (dbc, book=0, segment_id='dbc_complete')

Text is preserved VERBATIM at this stage — no lowercasing, no j→i / v→u
normalisation, no lemmatisation.  Those steps happen in later scripts.

Validation:
  - Prints row counts and compares to expected.
  - Flags any chapter whose text is empty or whitespace-only.
  - Prints the first few segment_ids for manual spot-checking.
"""

import csv
import os
import re
import sys
from importlib.machinery import SourceFileLoader


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DBG_XML_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'perseus',
                             'caes_bg_lat.xml')
DBC_XML_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'perseus',
                             'caes.bc_lat.xml')

CHAPTERS_CSV = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                            'corpus_chapters.csv')
BOOKS_CSV    = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                            'corpus_books.csv')

PARSER_SCRIPT = os.path.join(SCRIPT_DIR, '02_parse_tei.py')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space; strip ends."""
    return re.sub(r'\s+', ' ', text).strip()


def load_parser():
    """
    Dynamically load the 02_parse_tei.py module (whose filename starts with a
    digit, so plain `import` won't work).  Returns the module object.
    """
    if not os.path.exists(PARSER_SCRIPT):
        print(f"ERROR: Parser script not found at:\n  {PARSER_SCRIPT}")
        sys.exit(1)

    try:
        loader = SourceFileLoader('parse_tei', PARSER_SCRIPT)
        module = loader.load_module()
        # Verify the expected symbols are available
        _ = module.Chapter
        _ = module.parse_dbg
        _ = module.parse_dbc
        return module
    except Exception as e:
        print(f"ERROR: Failed to load parser module:\n  {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Build chapter-level CSV
# ---------------------------------------------------------------------------

def write_chapters_csv(chapters, path: str) -> int:
    """
    Write corpus_chapters.csv.

    Columns: work, book, chapter, segment_id, text

    Returns the number of rows written (excluding header).
    """
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['work', 'book', 'chapter', 'segment_id', 'text'])

        for ch in chapters:
            writer.writerow([
                ch.work,
                ch.book,
                ch.chapter,
                ch.segment_id,
                ch.text,          # verbatim — no normalisation yet
            ])

    return len(chapters)


# ---------------------------------------------------------------------------
# Build book-level CSV
# ---------------------------------------------------------------------------

def write_books_csv(chapters, path: str) -> int:
    """
    Write corpus_books.csv.

    Columns: work, book, segment_id, text

    Groups:
      DBG Book I   … DBG Book VIII  (segment_id = dbg_book01 … dbg_book08)
      DBC Complete                  (segment_id = dbc_complete,  book=0)

    Chapter texts are concatenated with a space separator; whitespace is
    normalised per book.

    Returns the number of rows written (should be 9).
    """
    # Group by (work, book)
    book_map: dict[tuple, list] = {}
    for ch in chapters:
        key = (ch.work, ch.book)
        book_map.setdefault(key, []).append(ch.text)

    written = 0

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['work', 'book', 'segment_id', 'text'])

        # DBG Books I–VIII
        for bk in range(1, 9):
            key = ('dbg', bk)
            if key in book_map:
                full = normalise_whitespace(' '.join(book_map[key]))
                seg_id = f'dbg_book{bk:02d}'
                writer.writerow(['dbg', bk, seg_id, full])
                written += 1
            else:
                print(f"  WARNING: DBG Book {bk} has no chapters — skipping.")

        # DBC Complete (all three books concatenated)
        dbc_texts: list[str] = []
        for bk in range(1, 4):
            key = ('dbc', bk)
            if key in book_map:
                dbc_texts.extend(book_map[key])
        if dbc_texts:
            full = normalise_whitespace(' '.join(dbc_texts))
            writer.writerow(['dbc', 0, 'dbc_complete', full])
            written += 1
        else:
            print("  WARNING: DBC has no chapters — skipping DBC Complete row.")

    return written


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(chapters, n_chapter_rows: int, n_book_rows: int):
    """Print validation summary; exit non-zero if anomalies found."""
    expected_chapters = 644   # 401 DBG + 243 DBC
    expected_books    = 9
    ok = True

    # --- Chapter counts ---
    print(f"{'='*60}")
    print(f"  VALIDATION")
    print(f"{'='*60}")
    print(f"  corpus_chapters.csv rows: {n_chapter_rows:4d}  "
          f"(expected ~{expected_chapters})")
    print(f"  corpus_books.csv rows:    {n_book_rows:4d}  "
          f"(expected  {expected_books})")

    # Per-work breakdown
    dbg_count = sum(1 for ch in chapters if ch.work == 'dbg')
    dbc_count = sum(1 for ch in chapters if ch.work == 'dbc')
    print(f"    DBG chapters: {dbg_count}")
    print(f"    DBC chapters: {dbc_count}")

    if abs(n_chapter_rows - expected_chapters) > 5:
        print(f"  ✗  Chapter count off by {n_chapter_rows - expected_chapters:+d}")
        ok = False
    else:
        print(f"  ✓  Chapter count within tolerance")

    if n_book_rows != expected_books:
        print(f"  ✗  Book row count is {n_book_rows}, expected {expected_books}")
        ok = False
    else:
        print(f"  ✓  Book row count correct")

    # --- Empty segments ---
    empty = [ch for ch in chapters if not ch.text.strip()]
    if empty:
        print(f"\n  ✗  {len(empty)} EMPTY CHAPTER(S) FOUND:")
        for ch in empty:
            print(f"       {ch.segment_id}")
        ok = False
    else:
        print(f"\n  ✓  No empty chapters found")

    # --- Spot-check ---
    print(f"\n  Spot-check — first 5 chapter segment_ids:")
    for ch in chapters[:5]:
        preview = ch.text[:80].replace('\n', ' ')
        print(f"    {ch.segment_id:30s}  \"{preview}…\"")

    print(f"\n  Spot-check — book segment_ids:")
    for bk in range(1, 9):
        print(f"    dbg_book{bk:02d}")
    print(f"    dbc_complete")

    if not ok:
        print(f"\n  ⚠  Validation found issues — review output above.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load the parser and run it ----------------------------------------
    print("Loading parser module …")
    parse_tei = load_parser()
    Chapter = parse_tei.Chapter

    print("Parsing De Bello Gallico …")
    dbg_chapters = parse_tei.parse_dbg(DBG_XML_PATH)

    print("Parsing De Bello Civili …")
    dbc_chapters = parse_tei.parse_dbc(DBC_XML_PATH)

    all_chapters: list = dbg_chapters + dbc_chapters
    print(f"Total chapters in memory: {len(all_chapters)}\n")

    # --- Write CSVs --------------------------------------------------------
    print("Writing corpus_chapters.csv …")
    n_ch = write_chapters_csv(all_chapters, CHAPTERS_CSV)
    print(f"  → {n_ch} rows written to {CHAPTERS_CSV}")

    print("Writing corpus_books.csv …")
    n_bk = write_books_csv(all_chapters, BOOKS_CSV)
    print(f"  → {n_bk} rows written to {BOOKS_CSV}\n")

    # --- Validate ----------------------------------------------------------
    validate(all_chapters, n_ch, n_bk)

    print(f"\n{'='*60}")
    print("  ✓  CSV corpora built successfully.")
    print(f"{'='*60}")
    print(f"\nNext step: scripts/04_normalize_corpus.py (orthographic normalisation)")


if __name__ == '__main__':
    main()
