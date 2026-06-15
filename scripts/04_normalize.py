#!/usr/bin/env python3
"""
scripts/04_normalize.py

Orthographic normalisation of the raw CSV corpora for Classical Latin.

Reads:
    data/corpus/corpus_chapters.csv
    data/corpus/corpus_books.csv

Produces:
    data/corpus/corpus_chapters_normalized.csv
    data/corpus/corpus_books_normalized.csv

Normalisation rules (applied in order):
    1.  Uppercase J → I    (e.g., "Julius" → "Iulius")
    2.  Uppercase V → U    (e.g., "VENIT"   → "UENIT")
    3.  Lowercase j → i    (e.g., "jam"     → "iam")
    4.  Lowercase v → u    (e.g., "venit"   → "uenit")
    5.  Lowercase everything

The order matters: we handle uppercase J/V FIRST so they become I/U, then
lowercase j/v → i/u, then lowercase all.  If we lowered first, "Julius" →
"julius" and the j→i rule would still catch it, but it's cleaner to work
at the case level intended.

Python's csv module has a default field-size limit of 131 072 bytes (128 KB).
Book-level CSVs concatenate entire books (~30 000–100 000 words) into a single
'text' field, easily exceeding that limit.  We raise the limit to sys.maxsize
before any CSV I/O.

Original (raw) CSVs are never modified.
"""

import csv
import os
import sys


# --- Raise CSV field size limit ---------------------------------------------
# Book-level text fields can be huge (DBG Book 7 = 90 chapters concatenated).
# The default 128 KB limit is far too small for this corpus.
csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')

INPUTS = {
    'chapters': os.path.join(CORPUS_DIR, 'corpus_chapters.csv'),
    'books':    os.path.join(CORPUS_DIR, 'corpus_books.csv'),
}

OUTPUTS = {
    'chapters': os.path.join(CORPUS_DIR, 'corpus_chapters_normalized.csv'),
    'books':    os.path.join(CORPUS_DIR, 'corpus_books_normalized.csv'),
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Apply Classical Latin orthographic normalisation.

    Uses str.translate() with a character-mapping table — the fastest
    Python method for per-character replacement (C-level, Unicode-safe).

    Order of operations:
      1. J→I, V→U, j→i, v→u  (single translate pass)
      2. str.lower()
    """
    trans_map = {
        ord('J'): 'I',
        ord('V'): 'U',
        ord('j'): 'i',
        ord('v'): 'u',
    }

    result = text.translate(trans_map)
    result = result.lower()

    return result


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_csv(input_path: str, output_path: str, label: str) -> int:
    """
    Read a CSV, normalise the 'text' column, write to output_path.

    Returns the number of rows processed.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found:\n  {input_path}")
        sys.exit(1)

    rows = []
    with open(input_path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames

        if 'text' not in fieldnames:
            print(f"ERROR: No 'text' column found in {label} CSV. "
                  f"Columns: {fieldnames}")
            sys.exit(1)

        for row in reader:
            row['text'] = normalize(row['text'])
            rows.append(row)

    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Before/after examples
# ---------------------------------------------------------------------------

def print_examples(input_path: str, output_path: str, label: str,
                   n: int = 3):
    """
    Print side-by-side before/after snippets for manual verification.
    """
    # Read the first N rows from both files
    with open(input_path, 'r', encoding='utf-8') as fh:
        raw_rows = list(csv.DictReader(fh))[:n]

    with open(output_path, 'r', encoding='utf-8') as fh:
        norm_rows = list(csv.DictReader(fh))[:n]

    print(f"\n--- {label}: first {n} rows (before → after) ---\n")

    for i, (raw, norm) in enumerate(zip(raw_rows, norm_rows)):
        seg_id = raw.get('segment_id', f'row {i}')
        before = raw['text'][:120].replace('\n', ' ')
        after  = norm['text'][:120].replace('\n', ' ')
        print(f"  [{seg_id}]")
        print(f"    BEFORE:  {before}…")
        print(f"    AFTER:   {after}…")
        print()

    # Also show a few words with known J/V occurrences if present
    print(f"  Checking specific normalisations across all {label} rows …")
    with open(output_path, 'r', encoding='utf-8') as fh:
        all_text = ' '.join(row['text'] for row in csv.DictReader(fh))

    checks = {
        "Residual 'j'":   'j' in all_text,
        "Residual 'J'":   'J' in all_text,
        "Residual 'v'":   'v' in all_text,
        "Residual 'V'":   'V' in all_text,
    }
    for description, found in checks.items():
        if found:
            # Count occurrences to gauge severity
            count = all_text.count(description[-2])
            print(f"    ⚠  {description}: {count} occurrence(s) — investigate")
        else:
            print(f"    ✓  {description}: none found")

    # Confirm lowercase
    if all_text == all_text.lower():
        print(f"    ✓  All text is lowercase")
    else:
        upper_chars = set(ch for ch in all_text if ch.isupper())
        print(f"    ⚠  Uppercase characters remain: {sorted(upper_chars)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Orthographic Normalisation (j→i, v→u, lowercase)")
    print("=" * 60)

    # --- Process chapters CSV ---
    print("\nProcessing corpus_chapters.csv …")
    n_ch = process_csv(INPUTS['chapters'], OUTPUTS['chapters'], 'chapters')
    print(f"  → {n_ch} rows written to {OUTPUTS['chapters']}")

    # --- Process books CSV ---
    print("\nProcessing corpus_books.csv …")
    n_bk = process_csv(INPUTS['books'], OUTPUTS['books'], 'books')
    print(f"  → {n_bk} rows written to {OUTPUTS['books']}")

    # --- Verify original files untouched ---
    print("\nVerifying original files are intact …")
    for key in ('chapters', 'books'):
        with open(INPUTS[key], 'r', encoding='utf-8') as fh:
            first_line = fh.readline().strip()
        if 'work,book' in first_line:
            print(f"  ✓  {os.path.basename(INPUTS[key])} unchanged (header present)")
        else:
            print(f"  ✗  {os.path.basename(INPUTS[key])} appears modified!")

    # --- Before/after examples ---
    print_examples(INPUTS['chapters'], OUTPUTS['chapters'], 'corpus_chapters')
    print_examples(INPUTS['books'],    OUTPUTS['books'],    'corpus_books')

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  ✓  Normalisation complete.")
    print(f"{'='*60}")
    print(f"\n  Chapter-level: {n_ch} rows → {OUTPUTS['chapters']}")
    print(f"  Book-level:    {n_bk} rows → {OUTPUTS['books']}")
    print(f"\nNext step: scripts/05_lemmatize.py (CLTK lemmatisation)")


if __name__ == '__main__':
    main()
