#!/usr/bin/env python3
"""
scripts/09_export_for_stylo.py

Export corpora as plain‑text files for the R `stylo` package.

`stylo()` expects a directory of UTF‑8 text files — one file per document —
NOT a CSV.  Each filename encodes authorship and work identity, and the
text inside is the clean, tokenised (or lemmatised) running prose.

CRITICAL: stylo lists ALL entries in the corpus directory and tries to
open each one as a text file.  Subdirectories cause it to fail.  Therefore
chapter‑level exports are placed in SEPARATE SIBLING directories, not
subdirectories of the book‑level corpus.

Reads:
    data/corpus/corpus_books_normalized_lemmatized.csv
    data/corpus/corpus_chapters_normalized_lemmatized.csv

Produces:
    data/stylo_corpus_tokens/
        Caesar_DBG-01.txt   … Caesar_DBG-07.txt
        Hirtius_DBG-08.txt
        Caesar_DBC.txt

    data/stylo_corpus_lemmas/
        (same filenames, but content built from the `lemmas` column)

    data/stylo_corpus_tokens_chapters/
        Caesar_DBG-01_ch001.txt … Caesar_DBC-03_ch112.txt

    data/stylo_corpus_lemmas_chapters/
        (same filenames, lemma content)

Filename scheme:
    Book:   <Author>_<Work>-<Book>.txt
    Chapter: <Author>_<Work>-<Book>_ch<Chapter>.txt

    Author:  Caesar  (DBG I–VII, DBC)
             Hirtius (DBG VIII)
    Book:    01–08 for DBG, 01–03 for DBC (zero‑padded for sorting)
"""

import csv
import os
import sys

# --- Raise CSV field size limit ---------------------------------------------
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# Configuration
# ===========================================================================

# Toggle chapter‑level export (set to False to skip)
DO_CHAPTERS = True


# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'corpus')
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

BOOKS_CSV    = os.path.join(CORPUS_DIR, 'corpus_books_normalized_lemmatized.csv')
CHAPTERS_CSV = os.path.join(CORPUS_DIR, 'corpus_chapters_normalized_lemmatized.csv')

# Book‑level: one directory per representation (no subdirectories!)
STYLO_TOKENS_DIR = os.path.join(DATA_DIR, 'stylo_corpus_tokens')
STYLO_LEMMAS_DIR = os.path.join(DATA_DIR, 'stylo_corpus_lemmas')

# Chapter‑level: SEPARATE sibling directories (stylo cannot handle
# subdirectories inside its corpus dir)
STYLO_TOKENS_CHAPTERS_DIR = os.path.join(DATA_DIR, 'stylo_corpus_tokens_chapters')
STYLO_LEMMAS_CHAPTERS_DIR = os.path.join(DATA_DIR, 'stylo_corpus_lemmas_chapters')


# ===========================================================================
# Filename builder
# ===========================================================================

def build_book_filename(row: dict) -> str:
    """
    Build a stylo‑compatible filename for a book‑level row.

    Rules:
        DBG Book 1–7  ->  Caesar_DBG-01.txt … Caesar_DBG-07.txt
        DBG Book 8    ->  Hirtius_DBG-08.txt
        DBC           ->  Caesar_DBC.txt
    """
    work  = row.get('work', '')
    book  = row.get('book', '')
    segid = row.get('segment_id', '')

    if segid == 'dbc_complete':
        return 'Caesar_DBC.txt'

    if work == 'dbg':
        book_int = int(book)
        author = 'Hirtius' if book_int == 8 else 'Caesar'
        return f'{author}_DBG-{book_int:02d}.txt'

    return f'Unknown_{work}_{book}.txt'


def build_chapter_filename(row: dict) -> str:
    """
    Build a stylo‑compatible filename for a chapter‑level row.

    Rules:
        DBG Book 1–7, ch 1–N   ->  Caesar_DBG-01_ch001.txt
        DBG Book 8,    ch 1–N   ->  Hirtius_DBG-08_ch001.txt
        DBC Book 1–3,  ch 1–N   ->  Caesar_DBC-01_ch001.txt
    """
    work    = row.get('work', '')
    book    = row.get('book', '')
    chapter = row.get('chapter', '')

    book_int    = int(book)
    chapter_int = int(chapter)

    if work == 'dbg':
        author = 'Hirtius' if book_int == 8 else 'Caesar'
        return f'{author}_DBG-{book_int:02d}_ch{chapter_int:03d}.txt'

    if work == 'dbc':
        return f'Caesar_DBC-{book_int:02d}_ch{chapter_int:03d}.txt'

    return f'Unknown_{work}_{book_int:02d}_ch{chapter_int:03d}.txt'


# ===========================================================================
# File writer
# ===========================================================================

def write_text_file(directory: str, filename: str, content: str):
    """
    Write a UTF‑8 plain‑text file.

    Strips trailing whitespace and ensures the file ends with exactly
    one newline (stylo's text parser is sensitive to this).
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    clean = content.strip()

    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(clean)
        fh.write('\n')


def clear_directory(directory: str):
    """
    Remove ALL files from a directory (stylo corpus dirs must be clean).
    Does NOT recurse into subdirectories.  Removes subdirectories too.
    """
    if not os.path.isdir(directory):
        return

    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)


def export_directory(row_list: list[dict],
                     directory: str,
                     column: str,
                     filename_fn,
                     label: str) -> list[str]:
    """
    Export a list of rows to a directory of plain‑text files.

    Args:
        row_list   — list of row dicts from a CSV
        directory  — output directory (created if needed)
        column     — which CSV column to use ('tokens' or 'lemmas')
        filename_fn — callable(row) -> str filename
        label      — human label for progress messages

    Returns a sorted list of filenames written.
    """
    os.makedirs(directory, exist_ok=True)

    written: list[str] = []

    for row in row_list:
        fname = filename_fn(row)
        text  = row.get(column, '')

        write_text_file(directory, fname, text)
        written.append(fname)

    written.sort()

    print(f"\n  [{label}]  {len(written)} files -> {directory}/")
    for fname in written:
        path = os.path.join(directory, fname)
        size = os.path.getsize(path)
        print(f"    {fname:48s}  {size:>8,} bytes")

    return written


# ===========================================================================
# Validation
# ===========================================================================

def validate_exports(tokens_files: list[str],
                     lemmas_files: list[str],
                     tokens_dir: str,
                     lemmas_dir: str):
    """
    Verify that the token and lemma directories contain the same filenames
    and that no files are empty.
    """
    print(f"\n{'-'*50}")
    print(f"  Validation")
    print(f"{'-'*50}")

    tokens_set = set(tokens_files)
    lemmas_set = set(lemmas_files)

    if tokens_set == lemmas_set:
        print(f"  OK  Token and lemma directories have identical filenames "
              f"({len(tokens_set)} files each)")
    else:
        only_tokens = tokens_set - lemmas_set
        only_lemmas = lemmas_set - tokens_set
        if only_tokens:
            print(f"  ERROR  Files only in tokens/: {sorted(only_tokens)}")
        if only_lemmas:
            print(f"  ERROR  Files only in lemmas/: {sorted(only_lemmas)}")

    # Quick content check: no file should be empty
    for directory, label in [(tokens_dir, 'tokens'),
                              (lemmas_dir, 'lemmas')]:
        empty_files = []
        for fname in os.listdir(directory):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(directory, fname)
            if os.path.getsize(path) == 0:
                empty_files.append(fname)
        if empty_files:
            print(f"  ERROR  {len(empty_files)} empty file(s) in {label}/:")
            for ef in empty_files:
                print(f"       {ef}")
        else:
            print(f"  OK  No empty files in {label}/")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("  Stylo Plain‑Text Corpus Export")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\nLoading corpora ...")
    books_rows    = load_csv(BOOKS_CSV)
    chapters_rows = load_csv(CHAPTERS_CSV)

    print(f"  Books:    {len(books_rows)} rows")
    print(f"  Chapters: {len(chapters_rows)} rows")

    # ------------------------------------------------------------------
    # 2. Clear previous exports (once, before any writes)
    # ------------------------------------------------------------------
    print("\nClearing previous exports ...")
    all_dirs = [STYLO_TOKENS_DIR, STYLO_LEMMAS_DIR]
    if DO_CHAPTERS:
        all_dirs.extend([STYLO_TOKENS_CHAPTERS_DIR, STYLO_LEMMAS_CHAPTERS_DIR])
    for d in all_dirs:
        clear_directory(d)
    # Also clean up old chapter subdirectories inside the book dirs
    # (leftover from the previous version of this script)
    for old_sub in ['chapters']:
        for parent in [STYLO_TOKENS_DIR, STYLO_LEMMAS_DIR]:
            old_path = os.path.join(parent, old_sub)
            if os.path.isdir(old_path):
                import shutil
                shutil.rmtree(old_path)
                print(f"  Removed stale directory: {old_path}")
    print("  Done.")

    # ------------------------------------------------------------------
    # 3. Export book‑level
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  BOOK‑LEVEL EXPORT")
    print(f"{'='*50}")

    book_tokens = export_directory(
        books_rows, STYLO_TOKENS_DIR, 'tokens',
        build_book_filename, 'tokens (books)'
    )

    book_lemmas = export_directory(
        books_rows, STYLO_LEMMAS_DIR, 'lemmas',
        build_book_filename, 'lemmas (books)'
    )

    validate_exports(book_tokens, book_lemmas,
                     STYLO_TOKENS_DIR, STYLO_LEMMAS_DIR)

    # ------------------------------------------------------------------
    # 4. Export chapter‑level (to SEPARATE sibling directories)
    # ------------------------------------------------------------------
    if DO_CHAPTERS:
        print(f"\n{'='*50}")
        print(f"  CHAPTER‑LEVEL EXPORT (separate directories)")
        print(f"{'='*50}")

        chap_tokens = export_directory(
            chapters_rows, STYLO_TOKENS_CHAPTERS_DIR, 'tokens',
            build_chapter_filename, 'tokens (chapters)'
        )

        chap_lemmas = export_directory(
            chapters_rows, STYLO_LEMMAS_CHAPTERS_DIR, 'lemmas',
            build_chapter_filename, 'lemmas (chapters)'
        )

        validate_exports(chap_tokens, chap_lemmas,
                         STYLO_TOKENS_CHAPTERS_DIR,
                         STYLO_LEMMAS_CHAPTERS_DIR)

        print(f"\n  Chapter files: {len(chap_tokens)} tokens, "
              f"{len(chap_lemmas)} lemmas")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Export complete.")
    print(f"{'='*60}")

    print(f"""
  Directories for stylo (these are clean — no subdirectories):

    Book‑level:
      {STYLO_TOKENS_DIR}/
      {STYLO_LEMMAS_DIR}/

    Chapter‑level:
      {STYLO_TOKENS_CHAPTERS_DIR}/
      {STYLO_LEMMAS_CHAPTERS_DIR}/

  To use in R:

    library(stylo)

    # Book‑level tokens:
    stylo(corpus.dir = "{STYLO_TOKENS_DIR}",
          analyzed.features = "w",
          mfw.min = 200, mfw.max = 200,
          distance.measure = "dist.delta")

    # Chapter‑level lemmas:
    stylo(corpus.dir = "{STYLO_LEMMAS_CHAPTERS_DIR}",
          analyzed.features = "w",
          mfw.min = 200, mfw.max = 200,
          distance.measure = "dist.delta")
""")


# ===========================================================================
# Data loader
# ===========================================================================

def load_csv(path: str) -> list[dict]:
    """Load a CSV file and return rows as a list of dicts."""
    if not os.path.exists(path):
        print(f"ERROR: File not found:\n  {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


if __name__ == '__main__':
    main()
