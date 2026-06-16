#!/usr/bin/env python3
"""
scripts/20c_normalize_lemmatize_calib.py

Applies normalization and CLTK lemmatization to the post-processed
calibration CSVs (merged Cicero yearly, fixed 7-book DBC pseudo-books).

Run AFTER scripts/20b_fix_calibration.py.
"""

import csv
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'corpus')


def normalize_latin(text):
    text = text.lower()
    text = text.replace('j', 'i')
    text = text.replace('v', 'u')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_corpus(input_stem, corpus_name):
    """
    Read {input_stem}.csv, produce:
      {input_stem}_normalized.csv
      {input_stem}_normalized_lemmatized.csv
    """
    input_path = os.path.join(CORPUS_DIR, f'{input_stem}.csv')
    norm_path  = os.path.join(CORPUS_DIR, f'{input_stem}_normalized.csv')
    lemma_path = os.path.join(CORPUS_DIR,
                              f'{input_stem}_normalized_lemmatized.csv')

    if not os.path.exists(input_path):
        print(f"  ✗ Not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"\n  {corpus_name}: {len(rows)} units")

    # ── Normalize ────────────────────────────────────────────────
    for r in rows:
        r['text'] = normalize_latin(r['text'])

    with open(norm_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"    Normalized: {norm_path}")

    # ── Lemmatize ────────────────────────────────────────────────
    try:
        from cltk import NLP
        nlp = NLP(language_code="lat", suppress_banner=True)
    except Exception as e:
        print(f"    ✗ CLTK unavailable: {e}")
        print(f"    Skipping lemmatization.")
        return

    n_done = 0
    for i, r in enumerate(rows):
        text = r['text']
        if len(text) < 50:
            r['tokens'] = text
            r['lemmas'] = text
            continue

        try:
            doc = nlp.analyze(text=text[:8000])
            tokens = [w.string for w in doc.words]
            lemmas = [w.lemma if w.lemma else w.string for w in doc.words]
            r['tokens'] = ' '.join(tokens)
            r['lemmas'] = ' '.join(lemmas)
            n_done += 1
        except Exception as e:
            print(f"    ⚠ Lemmatization failed for {r['unit_id']}: {e}")
            r['tokens'] = text
            r['lemmas'] = text

        if (i + 1) % 3 == 0:
            print(f"      {i + 1}/{len(rows)} lemmatized...")

    # Write lemmatized CSV (add tokens and lemmas columns)
    out_fieldnames = list(fieldnames) + ['tokens', 'lemmas']

    with open(lemma_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"    Lemmatized ({n_done}/{len(rows)}): {lemma_path}")


def main():
    print("=" * 60)
    print("  CALIBRATION NORMALIZE + LEMMATIZE")
    print("=" * 60)

    process_corpus('calib_cicero_atticum_yearly_merged',
                   'Cicero Ad Atticum (merged yearly)')
    process_corpus('calib_dbc_pseudo_books',
                   'DBC Pseudo-Books (7 books)')

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
