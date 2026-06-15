#!/usr/bin/env python3
"""
scripts/06_diagnostics.py

Corpus diagnostics and profiling.

Reads:
    data/corpus/corpus_chapters_normalized_lemmatized.csv
    data/corpus/corpus_books_normalized_lemmatized.csv

Produces:
    outputs/corpus_diagnostics.csv        — per-book statistics table
    outputs/chapter_lengths.csv           — per-chapter token counts
    figures/bar_per_book_tokens.png        — bar chart of tokens per book
    figures/hist_chapter_lengths.png       — histogram of chapter lengths

Purpose:
    Expose small-sample and unequal-length risks before feature extraction.
    The overview notes that individual books range from ~5k to ~15k words,
    and 7 Caesarian books is a tiny n for trend detection.  This script
    quantifies those risks and prints a diagnostic summary.
"""

import csv
import os
import sys

# --- Raise CSV field size limit ---------------------------------------------
# Book-level text/tokens fields can exceed the 128 KB default.
csv.field_size_limit(sys.maxsize)

import matplotlib
matplotlib.use('Agg')                 # headless backend — no GUI required
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CHAPTERS_CSV = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                            'corpus_chapters_normalized_lemmatized.csv')
BOOKS_CSV    = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                            'corpus_books_normalized_lemmatized.csv')

OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')

DIAGNOSTICS_OUT = os.path.join(OUTPUT_DIR, 'corpus_diagnostics.csv')
CHAPTER_LEN_OUT = os.path.join(OUTPUT_DIR, 'chapter_lengths.csv')

BAR_CHART    = os.path.join(FIGURES_DIR, 'bar_per_book_tokens.png')
HIST_CHART   = os.path.join(FIGURES_DIR, 'hist_chapter_lengths.png')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def token_count(text_or_tokens: str) -> int:
    """Count whitespace-delimited tokens in a tokens column."""
    if not text_or_tokens or not text_or_tokens.strip():
        return 0
    return len(text_or_tokens.strip().split())


def type_token_ratio(tokens_str: str) -> float:
    """Type-token ratio: unique tokens / total tokens."""
    tokens = tokens_str.strip().split()
    total = len(tokens)
    if total == 0:
        return 0.0
    return len(set(tokens)) / total


def median(values: list) -> float:
    """Simple median — no numpy dependency for a single stat."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return float(sorted_vals[n // 2])
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(path: str) -> list[dict]:
    """Load chapter-level CSV.  Returns list of row dicts."""
    if not os.path.exists(path):
        print(f"ERROR: Chapter CSV not found:\n  {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def load_books(path: str) -> list[dict]:
    """Load book-level CSV.  Returns list of row dicts."""
    if not os.path.exists(path):
        print(f"ERROR: Book CSV not found:\n  {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_chapters(chapters: list[dict]) -> list[dict]:
    """
    Compute per-chapter token counts.

    Returns a list of dicts: {segment_id, work, book, chapter, tokens}
    """
    rows = []
    for ch in chapters:
        tc = token_count(ch.get('tokens', ''))
        rows.append({
            'segment_id': ch.get('segment_id', ''),
            'work':       ch.get('work', ''),
            'book':       ch.get('book', ''),
            'chapter':    ch.get('chapter', ''),
            'tokens':     tc,
        })
    return rows


def analyze_books(books: list[dict],
                  chapter_lengths: list[dict]) -> list[dict]:
    """
    Compute per-book statistics.

    For DBG books (I–VIII), chapter stats are looked up by (work, book).

    For DBC Complete, chapter stats are aggregated from DBC Books 1–3
    because the 'dbc_complete' row spans all three books and has book=0,
    which does not match any individual chapter's book number.

    Returns a list of dicts, one per book row, with keys:
        work, book, segment_id, tokens, types, ttr,
        min_chap, median_chap, max_chap, n_chapters
    """
    # Index chapter lengths by (work, book)
    ch_by_book: dict[tuple, list[int]] = {}
    for cl in chapter_lengths:
        key = (cl['work'], cl['book'])
        ch_by_book.setdefault(key, []).append(cl['tokens'])

    rows = []
    for bk in books:
        work = bk.get('work', '')
        book = bk.get('book', '')
        seg_id = bk.get('segment_id', '')
        tokens_str = bk.get('tokens', '')

        tc = token_count(tokens_str)
        types_count = len(set(tokens_str.strip().split())) if tokens_str.strip() else 0
        ttr = type_token_ratio(tokens_str)

        # Collect chapter lengths
        if seg_id == 'dbc_complete':
            # Aggregate all DBC chapters (books 1–3)
            ch_lengths = []
            for b in (1, 2, 3):
                ch_lengths.extend(ch_by_book.get(('dbc', str(b)), []))
        else:
            ch_lengths = ch_by_book.get((work, book), [])

        rows.append({
            'work':        work,
            'book':        book,
            'segment_id':  seg_id,
            'tokens':      tc,
            'types':       types_count,
            'ttr':         round(ttr, 4),
            'n_chapters':  len(ch_lengths),
            'min_chap':    min(ch_lengths) if ch_lengths else 0,
            'median_chap': round(median(ch_lengths)) if ch_lengths else 0,
            'max_chap':    max(ch_lengths) if ch_lengths else 0,
        })

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_diagnostics(book_stats: list[dict], path: str):
    """Write per-book diagnostics CSV."""
    fieldnames = ['work', 'book', 'segment_id', 'tokens', 'types', 'ttr',
                  'n_chapters', 'min_chap', 'median_chap', 'max_chap']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(book_stats)
    print(f"  → {len(book_stats)} rows written to {path}")


def write_chapter_lengths(chapter_stats: list[dict], path: str):
    """Write per-chapter token counts CSV."""
    fieldnames = ['segment_id', 'work', 'book', 'chapter', 'tokens']
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in chapter_stats:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"  → {len(chapter_stats)} rows written to {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bar_per_book(book_stats: list[dict], path: str):
    """
    Bar chart of total tokens per book.

    DBG Books I-VIII in purple; DBC Complete in blue.
    Annotated with exact token counts.
    """
    labels = []
    values = []
    colors = []

    for row in book_stats:
        seg_id = row['segment_id']
        labels.append(seg_id)
        values.append(row['tokens'])
        colors.append('#2E86AB' if row['work'] == 'dbc' else '#A23B72')

    fig, ax = plt.subplots(figsize=(10, 5))

    x_positions = range(len(labels))
    bars = ax.bar(x_positions, values, color=colors, edgecolor='white',
                  linewidth=0.8)

    # FIXED: set tick positions before setting tick labels (suppresses
    # matplotlib UserWarning about set_ticklabels() without set_ticks()).
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)

    # Annotate each bar with the token count
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max(values) * 0.01,
                f'{val:,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Tokens (words)')
    ax.set_title('Tokens per Book — De Bello Gallico & De Bello Civili')
    ax.set_ylim(0, max(values) * 1.12)  # leave headroom for annotations

    # Legend block
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A23B72', label='DBG (Caesar + Hirtius Bk VIII)'),
        Patch(facecolor='#2E86AB', label='DBC (Caesar, later work)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    # Horizontal line at median (Caesar-only books: DBG I–VII)
    caesar_values = [v for r, v in zip(book_stats, values)
                     if r['work'] == 'dbg' and r['book'] != '8']
    if caesar_values:
        med = median(caesar_values)
        ax.axhline(y=med, color='gray', linestyle='--', linewidth=0.7)
        ax.text(len(values) - 0.5, med + max(values) * 0.01,
                f'Caesar median: {med:,.0f}', fontsize=8, color='gray')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {path}")


def plot_hist_chapter_lengths(chapter_stats: list[dict], path: str):
    """
    Histogram of chapter token counts, split by work (DBG vs DBC).

    Overlaid semi-transparent histograms so the distributions can be
    compared directly.
    """
    dbg_lengths = [c['tokens'] for c in chapter_stats if c['work'] == 'dbg']
    dbc_lengths = [c['tokens'] for c in chapter_stats if c['work'] == 'dbc']

    if not dbg_lengths and not dbc_lengths:
        print("  ⚠  No chapter length data to plot.")
        return

    combined_max = max(max(dbg_lengths) if dbg_lengths else 0,
                       max(dbc_lengths) if dbc_lengths else 0)

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(0, combined_max, 40)

    ax.hist(dbg_lengths, bins=bins, alpha=0.6, color='#A23B72',
            label=f'DBG (n={len(dbg_lengths)})', edgecolor='white')
    ax.hist(dbc_lengths, bins=bins, alpha=0.6, color='#2E86AB',
            label=f'DBC (n={len(dbc_lengths)})', edgecolor='white')

    ax.set_xlabel('Tokens per chapter')
    ax.set_ylabel('Number of chapters')
    ax.set_title('Chapter Length Distribution — DBG vs DBC')
    ax.legend(fontsize=10)

    # Mark "too short" threshold (100 words)
    ax.axvline(x=100, color='crimson', linestyle=':', linewidth=1.2)
    ax.text(105, ax.get_ylim()[1] * 0.95,
            '100 tokens\n(too short for stable\nfrequency estimates)',
            fontsize=7, color='crimson', va='top')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {path}")


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def print_summary(book_stats: list[dict], chapter_stats: list[dict]):
    """Print a human-readable diagnostic summary to stdout."""
    # Split by work
    dbg_books = [b for b in book_stats if b['work'] == 'dbg']
    dbc_book  = [b for b in book_stats if b['work'] == 'dbc']

    # Compute overall stats
    all_tokens_all_books = [b['tokens'] for b in book_stats]

    print(f"\n{'='*70}")
    print(f"  CORPUS DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")

    # --- Corpus-wide ---
    print(f"\n  ┌─ Corpus-wide ─────────────────────────────────────────────┐")
    total_db = sum(all_tokens_all_books)
    print(f"  │  Total tokens (all works):           {total_db:>10,}")
    print(f"  │  Chapters parsed:                     {len(chapter_stats):>10}")
    print(f"  │  Book-level rows:                     {len(book_stats):>10}")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # --- Per-book table ---
    print(f"\n  ┌─ Per-book breakdown ──────────────────────────────────────────────────┐")
    print(f"  │ {'Book':<18s} {'Tokens':>8s} {'Types':>8s} {'TTR':>7s}  "
          f"{'Chs':>5s}  {'MinCh':>6s} {'MedCh':>6s} {'MaxCh':>6s} │")
    print(f"  │ {'─'*18} {'─'*8} {'─'*8} {'─'*7}  "
          f"{'─'*5}  {'─'*6} {'─'*6} {'─'*6} │")
    for row in book_stats:
        name = (f"{row['work']}_{row['book']}"
                if row['work'] == 'dbg' else 'dbc_complete')
        print(f"  │ {name:<18s} {row['tokens']:>8,} {row['types']:>8,} "
              f"{row['ttr']:>7.3f}  "
              f"{row['n_chapters']:>5}  {row['min_chap']:>6} "
              f"{row['median_chap']:>6} {row['max_chap']:>6} │")
    print(f"  └──────────────────────────────────────────────────────────────────────┘")

    # --- Small-sample warnings ---
    print(f"\n  ┌─ WARNINGS — Small-sample & unequal-length risks ──────────┐")

    # Shortest DBG books
    dbg_sorted = sorted(dbg_books, key=lambda b: b['tokens'])
    shortest = dbg_sorted[:2]
    for b in shortest:
        print(f"  │  ⚠  {b['segment_id']} has only {b['tokens']:,} tokens — "
              f"frequency estimates will be noisiest here.")

    # Unequal length
    dbg_tokens_list = [b['tokens'] for b in dbg_books]
    if dbg_tokens_list:
        ratio = max(dbg_tokens_list) / min(dbg_tokens_list)
        print(f"  │  ⚠  DBG book length ratio (longest / shortest): {ratio:.1f}x")
        if ratio > 2.0:
            print(f"  │     → Strongly recommend normalising by text length "
                  f"(proportions / z-scores)")
        else:
            print(f"  │     → Moderate range; raw counts may be acceptable "
                  f"with caution")

    # Too-short chapters
    short_chapters = [c for c in chapter_stats if c['tokens'] < 100]
    if short_chapters:
        print(f"  │  ⚠  {len(short_chapters)} chapter(s) < 100 tokens — "
              f"too short for stable frequency estimates")
        # Show examples
        for sc in short_chapters[:3]:
            print(f"  │       {sc['segment_id']}: {sc['tokens']} tokens")
        if len(short_chapters) > 3:
            print(f"  │       … and {len(short_chapters) - 3} more")

    # n=7 warning
    print(f"  │")
    print(f"  │  ⚠  Only n=7 Caesarian books (8 with Hirtius Book VIII).")
    print(f"  │     Any apparent trend is a correlation over 7 points.")
    print(f"  │     ALL findings MUST be assessed with permutation / bootstrap")
    print(f"  │     tests — never rely on visual impression alone.")

    print(f"  └────────────────────────────────────────────────────────────────────┘")

    # --- Implications ---
    print(f"\n  ┌─ IMPLICATIONS FOR LATER STEPS ─────────────────────────────┐")
    print(f"  │  1. Feature extraction: prefer function words, most-frequent")
    print(f"  │     words, and character n-grams — these are content-independent")
    print(f"  │     and stable even at ~5,000 tokens.")
    print(f"  │  2. Distance metric: use a normalised measure (Burrows's Delta,")
    print(f"  │     cosine distance) — not raw frequency counts.")
    print(f"  │  3. Aggregation level: book-level comparisons are the primary")
    print(f"  │     unit of analysis (9 rows).  Chapter-level (~644 rows) can")
    print(f"  │     be used for cross-validation and sliding-window analyses.")
    print(f"  │  4. Controls: Book VIII (Hirtius) must separate from I–VII;")
    print(f"  │     DBC must serve as the later-Caesar anchor.")
    print(f"  └──────────────────────────────────────────────────────────┘")

    print(f"\n  Outputs saved:")
    print(f"    {DIAGNOSTICS_OUT}")
    print(f"    {CHAPTER_LEN_OUT}")
    print(f"    {BAR_CHART}")
    print(f"    {HIST_CHART}")
    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Corpus Diagnostics & Profiling")
    print("=" * 60)

    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Load data ---
    print("\nLoading data …")
    chapters = load_chapters(CHAPTERS_CSV)
    books    = load_books(BOOKS_CSV)
    print(f"  Chapters: {len(chapters)} rows")
    print(f"  Books:    {len(books)} rows")

    # --- Analyze ---
    print("\nComputing per-chapter token counts …")
    chapter_stats = analyze_chapters(chapters)

    print("Computing per-book statistics …")
    book_stats = analyze_books(books, chapter_stats)

    # --- Write outputs ---
    print("\nWriting outputs …")
    write_diagnostics(book_stats, DIAGNOSTICS_OUT)
    write_chapter_lengths(chapter_stats, CHAPTER_LEN_OUT)

    # --- Plot ---
    print("\nGenerating figures …")
    plot_bar_per_book(book_stats, BAR_CHART)
    plot_hist_chapter_lengths(chapter_stats, HIST_CHART)

    # --- Summary ---
    print_summary(book_stats, chapter_stats)


if __name__ == '__main__':
    main()
