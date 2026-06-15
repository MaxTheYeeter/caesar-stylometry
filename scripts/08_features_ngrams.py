#!/usr/bin/env python3
"""
scripts/08_features_ngrams.py

Character n‑gram feature matrices.

Reads:
    data/corpus/corpus_books_normalized_lemmatized.csv
    data/corpus/corpus_chapters_normalized_lemmatized.csv

Produces (in outputs/):
    Book‑level matrices:
        features_char2gram_books.csv
        features_char3gram_books.csv
        features_char4gram_books.csv

    Chapter‑level matrices:
        features_char2gram_chapters.csv
        features_char3gram_chapters.csv
        features_char4gram_chapters.csv

All values are RELATIVE frequencies (n‑gram count / total n‑grams in segment).
Each matrix includes metadata columns: segment_id, author_group, work, book,
total_ngrams.

Character n‑grams are robust to lemmatization error, capture sub‑word
morphological patterns (case endings, verb suffixes), and complement the
word‑based features from 07_features_words.py.  They are the primary feature
type used by the stylo R package and are standard in Latin stylometry.

Text is preprocessed for n‑gram extraction:
    – Already normalized (lowercase, j→i, v→u) from Step 2.1
    – Punctuation, digits, brackets removed → only [a‑z ] kept
    – Space (' ') preserved as word‑boundary signal
    – N‑grams extracted with a sliding window of width n

Parameters (configurable at top of script):
    NGRAM_SIZES  — character n‑gram sizes  (default: 2, 3, 4)
    TOP_K_MAP    — top‑K most‑frequent n‑grams per size  (default: 200/500/1000)
"""

import csv
import os
import re
import sys
from collections import Counter

# --- Raise CSV field size limit ---------------------------------------------
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# Configuration
# ===========================================================================

# N‑gram sizes to generate
NGRAM_SIZES = [2, 3, 4]

# Top‑K most‑frequent n‑grams to retain as features, per n‑gram size.
# Larger n → sparser feature space → larger K useful.
TOP_K_MAP = {
    2: 200,
    3: 500,
    4: 1000,
}

# Aggregation levels
AGGREGATION_LEVELS = ['books', 'chapters']


# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'corpus')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'outputs')

BOOKS_CSV    = os.path.join(CORPUS_DIR, 'corpus_books_normalized_lemmatized.csv')
CHAPTERS_CSV = os.path.join(CORPUS_DIR, 'corpus_chapters_normalized_lemmatized.csv')


# ===========================================================================
# Data loading
# ===========================================================================

def load_csv(path: str) -> list[dict]:
    """Load a CSV file; return list of row dicts."""
    if not os.path.exists(path):
        print(f"ERROR: File not found:\n  {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# ===========================================================================
# Author group labelling
# ===========================================================================

def get_author_group(row: dict) -> str:
    """
    Classify a row into its authorial group.

    Rules:
        DBG Books I–VII  → 'caesar'
        DBG Book VIII    → 'hirtius'
        DBC (any)        → 'caesar_dbc'
    """
    work = row.get('work', '')
    book = row.get('book', '')

    if work == 'dbc':
        return 'caesar_dbc'
    if work == 'dbg' and book == '8':
        return 'hirtius'
    if work == 'dbg':
        return 'caesar'
    return 'unknown'


# ===========================================================================
# Character n‑gram extraction
# ===========================================================================

# Characters to keep for n‑gram generation.
# Lowercase Latin letters (a‑z) plus space (word‑boundary signal).
CLEAN_RE = re.compile(r'[^a-z ]')


def clean_for_ngrams(text: str) -> str:
    """
    Normalise text for character n‑gram extraction.

    Keeps only lowercase letters a‑z and the space character.
    Removes: punctuation, digits, editorial brackets [ ], remaining
    uppercase letters, newlines, tabs.
    """
    # Remove everything except a‑z and space
    cleaned = CLEAN_RE.sub('', text)
    # Collapse multiple spaces to a single space
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.strip()


def extract_ngrams(text: str, n: int) -> list[str]:
    """
    Extract character n‑grams from pre‑cleaned text using a sliding
    window of width n.  Returns a list of n‑gram strings.
    """
    if len(text) < n:
        return []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


# ===========================================================================
# Feature selection
# ===========================================================================

def get_top_ngrams(all_texts: list[str], n: int, k: int) -> list[str]:
    """
    Compute the top‑K most‑frequent character n‑grams across a list
    of cleaned text strings.

    Returns a list of n‑gram strings in descending frequency order
    (most frequent first — conventional for stylometric features).
    """
    global_counter: Counter = Counter()

    for text in all_texts:
        ngrams = extract_ngrams(text, n)
        global_counter.update(ngrams)

    return [ng for ng, _count in global_counter.most_common(k)]


# ===========================================================================
# Relative frequency vectors
# ===========================================================================

def build_frequency_vector(cleaned_text: str, n: int,
                           feature_list: list[str]) -> dict[str, float]:
    """
    Build a relative‑frequency dict for one segment.

    Returns {ngram: proportion} for every n‑gram in feature_list.
    Proportion = count(ngram) / total_ngrams_in_segment.
    If the segment has zero n‑grams, all proportions are 0.0.
    """
    ngrams = extract_ngrams(cleaned_text, n)
    total = len(ngrams)

    if total == 0:
        return {f: 0.0 for f in feature_list}

    counts = Counter(ngrams)
    return {f: counts.get(f, 0) / total for f in feature_list}


# ===========================================================================
# Matrix builder
# ===========================================================================

def build_matrix(rows: list[dict], feature_list: list[str],
                 n: int) -> list[dict]:
    """
    Build a feature matrix as a list of row dicts.

    Each output row:
        segment_id, author_group, work, book, total_ngrams,
        <ngram_1>, <ngram_2>, … (relative frequencies)

    The `text` column is cleaned (punctuation stripped) before n‑gram
    extraction.  The original `text` column is NOT included in the output
    matrix (it's redundant with the n‑gram features).
    """
    matrix: list[dict] = []

    for row in rows:
        raw_text = row.get('text', '')
        cleaned = clean_for_ngrams(raw_text)
        ngrams = extract_ngrams(cleaned, n)

        vec = build_frequency_vector(cleaned, n, feature_list)

        out_row = {
            'segment_id':   row.get('segment_id', ''),
            'author_group': get_author_group(row),
            'work':         row.get('work', ''),
            'book':         row.get('book', ''),
            'total_ngrams': len(ngrams),
        }
        # Add feature columns in the defined order
        for feat in feature_list:
            out_row[feat] = round(vec.get(feat, 0.0), 10)

        matrix.append(out_row)

    return matrix


def write_matrix(matrix: list[dict], feature_list: list[str],
                 path: str):
    """Write a feature matrix to CSV."""
    if not matrix:
        print(f"  ⚠  Empty matrix — nothing written to {path}")
        return

    fieldnames = ['segment_id', 'author_group', 'work', 'book',
                  'total_ngrams'] + feature_list

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(matrix)

    print(f"  → {len(matrix)} rows × {len(feature_list)} features → {path}")


# ===========================================================================
# Validation helpers
# ===========================================================================

def validate_matrix(matrix: list[dict], feature_list: list[str],
                    label: str):
    """Print a compact validation summary for a matrix."""
    n_rows = len(matrix)
    n_feat = len(feature_list)
    author_counts = Counter(r['author_group'] for r in matrix)

    # Zero‑variance features
    zero_var = [f for f in feature_list
                if all(r[f] == 0.0 for r in matrix)]
    n_zero = len(zero_var)

    # Row‑sum range (should be ≤1.0; exactly 1.0 if the feature list
    # covers ALL possible n‑grams in the text, which it won't for n≥3)
    sum_min = min(sum(r[f] for f in feature_list) for r in matrix)
    sum_max = max(sum(r[f] for f in feature_list) for r in matrix)

    # Total n‑grams range
    ngram_min = min(r['total_ngrams'] for r in matrix)
    ngram_max = max(r['total_ngrams'] for r in matrix)

    print(f"    Rows: {n_rows} | Features: {n_feat}")
    print(f"    Author distribution: "
          f"caesar={author_counts['caesar']} "
          f"hirtius={author_counts['hirtius']} "
          f"caesar_dbc={author_counts['caesar_dbc']}")
    print(f"    N‑grams per segment: min={ngram_min:,}  max={ngram_max:,}")
    print(f"    Feature coverage (row sum): "
          f"min={sum_min:.3f}  max={sum_max:.3f}")

    if n_zero > 0:
        if n_zero <= 5:
            print(f"    ⚠  {n_zero} zero‑variance feature(s): "
                  f"{zero_var}")
        else:
            print(f"    ⚠  {n_zero} zero‑variance features "
                  f"(first 5: {zero_var[:5]})")
    else:
        print(f"    ✓  No zero‑variance features")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("  Character n‑Gram Feature Matrix Builder")
    print("=" * 60)

    print(f"\n  Parameters:")
    for n in NGRAM_SIZES:
        k = TOP_K_MAP.get(n, '???')
        print(f"    n={n}  →  top‑{k} n‑grams")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load corpora
    # ------------------------------------------------------------------
    print("Loading corpora …")
    books_rows    = load_csv(BOOKS_CSV)
    chapters_rows = load_csv(CHAPTERS_CSV)

    print(f"  Books:    {len(books_rows)} rows")
    print(f"  Chapters: {len(chapters_rows)} rows")

    # ------------------------------------------------------------------
    # 2. Pre‑clean all texts (do it once, not per n‑gram size)
    # ------------------------------------------------------------------
    print("\nCleaning texts for n‑gram extraction …")
    books_cleaned    = [clean_for_ngrams(r['text']) for r in books_rows]
    chapters_cleaned = [clean_for_ngrams(r['text']) for r in chapters_rows]

    # Quick stats on cleaned text
    book_chars  = sum(len(t) for t in books_cleaned)
    chap_chars  = sum(len(t) for t in chapters_cleaned)
    print(f"  Books:    {book_chars:,} characters (cleaned)")
    print(f"  Chapters: {chap_chars:,} characters (cleaned)")
    print(f"  Total:    {book_chars + chap_chars:,} characters")

    # Character‑set check
    all_chars = set()
    for t in books_cleaned + chapters_cleaned:
        all_chars.update(t)
    non_alpha = all_chars - set('abcdefghijklmnopqrstuvwxyz ')
    if non_alpha:
        print(f"  ⚠  Unexpected characters found: {sorted(non_alpha)!r}")
    else:
        print(f"  ✓  Character set clean: only [a‑z ]")

    # ------------------------------------------------------------------
    # 3. Build matrices for each n‑gram size
    # ------------------------------------------------------------------
    total_matrices = 0

    for n in NGRAM_SIZES:
        k = TOP_K_MAP.get(n)
        if k is None:
            print(f"\n  ⚠  No K defined for n={n} — skipping")
            continue

        print(f"\n{'='*50}")
        print(f"  Character {n}‑grams  (top‑{k})")
        print(f"{'='*50}")

        for agg in AGGREGATION_LEVELS:
            if agg == 'books':
                rows    = books_rows
                cleaned = books_cleaned
            else:
                rows    = chapters_rows
                cleaned = chapters_cleaned

            # --- Build feature list globally ---
            feature_list = get_top_ngrams(cleaned, n, k)

            if not feature_list:
                print(f"\n  ⚠  No n‑grams found for n={n}, {agg} — skipping")
                continue

            # Print a few example features
            examples = feature_list[:12]
            print(f"\n  [{agg}] Top features (first 12): "
                  f"{'  '.join(repr(ng) for ng in examples)}")

            # --- Build and write matrix ---
            matrix = build_matrix(rows, feature_list, n)

            fname = f'features_char{n}gram_{agg}.csv'
            fpath = os.path.join(OUTPUT_DIR, fname)
            write_matrix(matrix, feature_list, fpath)

            # --- Validate ---
            validate_matrix(matrix, feature_list, fname)

            total_matrices += 1

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ✓  {total_matrices} character n‑gram matrices written "
          f"to {OUTPUT_DIR}/")
    print(f"{'='*60}")

    print(f"\n  Matrices produced:")
    for n in NGRAM_SIZES:
        for agg in AGGREGATION_LEVELS:
            fname = f'features_char{n}gram_{agg}.csv'
            print(f"    {fname}")

    print(f"\n  Parameter summary:")
    print(f"    N‑gram sizes:  {NGRAM_SIZES}")
    print(f"    Top‑K map:     {TOP_K_MAP}")
    print(f"    Text source:   'text' column (normalized: lowercase, "
          f"j→i, v→u)")
    print(f"    Cleaning:      [^a‑z ] removed (punctuation, digits, "
          f"editorial brackets)")
    print(f"    Space:         preserved as word‑boundary signal")
    print(f"    Values:        relative frequency (count / total n‑grams "
          f"in segment)")

    print(f"\n  Next step: PCA / distance analysis, or "
          f"scripts/09_stylo_export.py")


if __name__ == '__main__':
    main()
