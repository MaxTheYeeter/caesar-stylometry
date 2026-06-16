#!/usr/bin/env python3
"""
scripts/20b_fix_calibration.py

Post-processes the calibration CSVs to meet minimum quality thresholds.
Run AFTER scripts/20_calibration_corpus.py.

Fixes:
  1. Cicero yearly bins: merges years with < 2000 tokens into adjacent years
  2. DBC pseudo-books: regenerates to exactly 7 equal-token chunks
"""

import csv
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'corpus')
RAW_DIR      = os.path.join(PROJECT_ROOT, 'data', 'raw', 'perseus')

MIN_TOKENS = 2000  # minimum tokens per unit for reliable feature estimation


def fix_cicero_yearly():
    """
    Merge low-token Cicero years with adjacent years.
    Strategy: iteratively merge the smallest unit with its smaller neighbour
    until all units have >= MIN_TOKENS tokens.
    """
    input_path  = os.path.join(CORPUS_DIR, 'calib_cicero_atticum_yearly.csv')
    output_path = os.path.join(CORPUS_DIR,
                               'calib_cicero_atticum_yearly_merged.csv')

    if not os.path.exists(input_path):
        print(f"✗ Input not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"\nCicero yearly bins: {len(rows)} initial units")
    print(f"  Minimum token threshold: {MIN_TOKENS}")

    # Build initial state
    units = []
    for r in rows:
        tokens = len(r['text'].split())
        n_letters = int(r.get('n_letters', 1))
        units.append({
            'unit_id': r['unit_id'],
            'order_index': int(r['order_index']),
            'known_date': int(r['known_date']),
            'text': r['text'],
            'tokens': tokens,
            'n_letters': n_letters,
        })

    # Iteratively merge
    merged_any = True
    iteration = 0
    while merged_any:
        merged_any = False
        iteration += 1

        # Find smallest unit below threshold
        undersized = [(i, u) for i, u in enumerate(units)
                      if u['tokens'] < MIN_TOKENS]
        if not undersized:
            break

        # Merge the one with fewest tokens
        undersized.sort(key=lambda x: x[1]['tokens'])
        idx, unit = undersized[0]

        # Decide which neighbour to merge with
        if idx == 0:
            merge_with = idx + 1  # only option: next
        elif idx == len(units) - 1:
            merge_with = idx - 1  # only option: previous
        else:
            prev_tokens = units[idx - 1]['tokens']
            next_tokens = units[idx + 1]['tokens']
            merge_with = idx - 1 if prev_tokens <= next_tokens else idx + 1

        # Merge
        other = units[merge_with]
        merged_text = unit['text'] + '\n\n' + other['text']
        merged_tokens = len(merged_text.split())
        merged_letters = unit['n_letters'] + other['n_letters']
        merged_date = (unit['known_date'] + other['known_date']) // 2  # avg year
        merged_id = unit['unit_id'] + '+' + other['unit_id']

        # Replace
        new_unit = {
            'unit_id': merged_id,
            'order_index': min(unit['order_index'], other['order_index']),
            'known_date': merged_date,
            'text': merged_text,
            'tokens': merged_tokens,
            'n_letters': merged_letters,
        }

        # Remove both and insert merged at the lower index
        low_idx = min(idx, merge_with)
        high_idx = max(idx, merge_with)
        del units[high_idx]
        del units[low_idx]
        units.insert(low_idx, new_unit)

        merged_any = True
        print(f"  Iter {iteration}: merged {unit['unit_id']} ({unit['tokens']}tk) "
              f"+ {other['unit_id']} ({other['tokens']}tk) "
              f"→ {merged_id} ({merged_tokens}tk)")

    # Re-assign order_index sequentially
    for i, u in enumerate(units):
        u['order_index'] = i + 1

    # Write
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['work', 'unit_id', 'order_index', 'known_date',
                      'date_source', 'text', 'n_letters']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for u in units:
            # Build date_source from constituent years
            u['work'] = 'cicero_atticum_yearly'
            u['date_source'] = f'Merged {u["n_letters"]} letters'
            writer.writerow(u)

    print(f"\n  Final: {len(units)} units")
    print(f"  Token range: {min(u['tokens'] for u in units)} – "
          f"{max(u['tokens'] for u in units)}")
    still_undersized = [u for u in units if u['tokens'] < MIN_TOKENS]
    if still_undersized:
        print(f"  ⚠ Still undersized: {len(still_undersized)} units")
    else:
        print(f"  ✓ All units >= {MIN_TOKENS} tokens")
    print(f"  Saved: {output_path}")


def fix_dbc_pseudo_books():
    """Regenerate DBC pseudo-books to exactly 7."""
    input_path = os.path.join(CORPUS_DIR, 'calib_dbc_pseudo_books.csv')

    if not os.path.exists(input_path):
        print(f"✗ Input not found: {input_path}")
        return

    # Read existing text (full DBC concatenated)
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Reconstruct full text
    full_text = ' '.join(r['text'] for r in rows)
    tokens = full_text.split()
    total = len(tokens)
    N = 7
    per_book = total // N

    print(f"\nDBC pseudo-books: {total} tokens → {per_book} tokens × {N} books")

    new_books = []
    for i in range(N):
        start = i * per_book
        end = start + per_book if i < N - 1 else total
        book_tokens = tokens[start:end]

        new_books.append({
            'work': 'dbc_pseudo_books',
            'unit_id': f'dbc_pseudo_{i + 1:02d}',
            'order_index': i + 1,
            'known_date': '',
            'date_source': f'DBC narrative-order pseudo-book {i + 1}/7',
            'text': ' '.join(book_tokens),
        })

    output_path = input_path  # overwrite
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['work', 'unit_id', 'order_index', 'known_date',
                      'date_source', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in new_books:
            writer.writerow(b)

    print(f"  Saved: {output_path} ({len(new_books)} books, "
          f"{min(len(b['text'].split()) for b in new_books)}–"
          f"{max(len(b['text'].split()) for b in new_books)} tokens each)")


def main():
    print("=" * 60)
    print("  CALIBRATION CORPUS POST-PROCESSING")
    print("=" * 60)

    fix_cicero_yearly()
    fix_dbc_pseudo_books()

    print(f"\n{'=' * 60}")
    print(f"  FIXES COMPLETE")
    print(f"  Use: calib_cicero_atticum_yearly_merged.csv (Cicero)")
    print(f"  Use: calib_dbc_pseudo_books.csv (DBC, now 7 books)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
