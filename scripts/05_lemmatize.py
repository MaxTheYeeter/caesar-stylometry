#!/usr/bin/env python3
"""
scripts/05_lemmatize.py

Tokenize and lemmatize the normalized corpora using the CLTK NLP pipeline
(Stanza backend for Latin).

Reads:
    data/corpus/corpus_chapters_normalized.csv
    data/corpus/corpus_books_normalized.csv

Produces:
    data/corpus/corpus_chapters_normalized_lemmatized.csv
    data/corpus/corpus_books_normalized_lemmatized.csv
    logs/unrecognized_lemmas.csv

Each output CSV retains all original columns and ADDS:
    tokens  — whitespace-joined raw token strings
    lemmas  — whitespace-joined lemma strings

The original 'text' column (normalized running text) is preserved; both
representations are needed downstream (the project overview calls for
analysis on both token and lemma representations).

Unrecognized lemmas (None, empty, or lemma == token where the token looks
inflected) are logged to logs/unrecognized_lemmas.csv with segment_id,
token, and proposed_lemma.

This script is SLOW — the Stanza Latin pipeline processes ~644 chapters
and 9 whole books.  Expect 10–40 minutes depending on hardware.
Progress is printed every 20 segments.
"""

import csv
import os
import re
import sys
import time
from datetime import datetime

# --- Raise CSV field size limit ---------------------------------------------
# Book-level text fields can be enormous.  The default 128 KB limit is
# far too small for whole-book concatenations.
csv.field_size_limit(sys.maxsize)

from cltk import NLP


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR = os.path.join(PROJECT_ROOT, 'data', 'corpus')
LOG_DIR    = os.path.join(PROJECT_ROOT, 'logs')

INPUTS = {
    'chapters': os.path.join(CORPUS_DIR, 'corpus_chapters_normalized.csv'),
    'books':    os.path.join(CORPUS_DIR, 'corpus_books_normalized.csv'),
}

OUTPUTS = {
    'chapters': os.path.join(CORPUS_DIR,
                             'corpus_chapters_normalized_lemmatized.csv'),
    'books':    os.path.join(CORPUS_DIR,
                             'corpus_books_normalized_lemmatized.csv'),
}

UNRECOGNIZED_LOG = os.path.join(LOG_DIR, 'unrecognized_lemmas.csv')


# ---------------------------------------------------------------------------
# NLP pipeline (loaded once at module level for reuse)
# ---------------------------------------------------------------------------

print("Loading CLTK Latin NLP pipeline (Stanza backend) …")
print("  First run may download Stanza models (~300 MB) — please be patient.")
print()

nlp = NLP(language_code='lat')
print("Pipeline ready.\n")


# ---------------------------------------------------------------------------
# Lemma classification helpers
# ---------------------------------------------------------------------------

# Words that are almost never inflected in Latin — lemma == token is expected.
INDECLINABLE_SET = {
    # Prepositions
    'a', 'ab', 'ad', 'ante', 'apud', 'contra', 'cum', 'de', 'e', 'ex',
    'in', 'inter', 'ob', 'per', 'post', 'prae', 'pro', 'propter', 'sine',
    'sub', 'super', 'trans', 'circum', 'extra', 'intra', 'supra',
    'usque', 'uersus', 'erga', 'penes', 'citra',
    # Conjunctions
    'et', 'atque', 'ac', 'que', 'nec', 'neque', 'aut', 'uel', 'si',
    'nisi', 'quod', 'quia', 'quoniam', 'cum', 'ut', 'ne', 'dum', 'donec',
    'priusquam', 'antequam', 'postquam', 'ubi', 'simul', 'quasi',
    'tamquam', 'quam', 'an', 'siue', 'seu', 'sin', 'quin', 'quominus',
    'etsi', 'tametsi', 'quamquam', 'quamuis', 'licet',
    # Adverbs (indeclinable)
    'non', 'haud', 'tamen', 'enim', 'nam', 'igitur', 'itaque', 'ergo',
    'ideo', 'ita', 'sic', 'tam', 'adeo', 'fere', 'paene', 'uix',
    'prope', 'satis', 'bene', 'male', 'forte', 'iam', 'nunc', 'tunc',
    'hic', 'huc', 'hinc', 'illic', 'illuc', 'illinc', 'ibi', 'inde',
    'ubi', 'nusquam', 'semper', 'saepe', 'interim', 'subito', 'statim',
    'mox', 'sane', 'quidem', 'uero', 'certe', 'scilicet', 'uidelicet',
    'quoque', 'etiam', 'praeterea', 'denique', 'tum', 'postea', 'antea',
    'nuper', 'olim', 'quondam', 'adhuc', 'ultro', 'sponte', 'frustra',
    'magis', 'potius', 'nimis', 'parum', 'uero', 'uix', 'procul',
    # Interjections / other invariants
    'o', 'heu', 'eheu', 'ecce', 'en',
}

# Very common nouns/adjectives whose nominative singular is the dictionary
# form — lemma == token is expected for nominative forms.
COMMON_NOMINATIVE_SET = {
    'caesar', 'gallia', 'roma', 'pompeius', 'senatus', 'exercitus',
    'bellum', 'castra', 'flumen', 'pars', 'hostis', 'miles', 'legio',
    'consul', 'imperator', 'populus', 'ciuitas', 'res', 'dies',
}

# Morphological endings that suggest the token is NOT in its lemma form
# (a quick heuristic — not linguistically rigorous, but useful for logging)
INFLECTED_ENDINGS = re.compile(
    r'(orum|arum|ibus|ebus|bus|erunt|erant|erint|issent|isset|isse|'
    r'arumque|orumque|ibusque|busque|eruntque|erantque|issentque|'
    r'arumue|orumue|ibusue|busue|'
    r'i$|ae$|is$|os$|as$|es$|em$|am$|um$|us$|a$)', re.IGNORECASE
)


def lemma_is_suspicious(token: str, lemma: str) -> bool:
    """
    Return True if the lemma warrants logging.

    We log:
      - Lemma is None or empty (lemmatizer had no guess).
      - Lemma equals the token AND the token appears to be an inflected
        form (not a nominative singular or an indeclinable word).  This
        catches cases where the lemmatizer silently returned the token
        as a fallback.
    """
    if lemma is None or lemma.strip() == '':
        return True

    if token == lemma:
        token_lower = token.lower()
        # Known indeclinable or expected nominative → not suspicious
        if token_lower in INDECLINABLE_SET:
            return False
        if token_lower in COMMON_NOMINATIVE_SET:
            return False
        # Token looks inflected but lemma wasn't derived → suspicious
        if INFLECTED_ENDINGS.search(token_lower):
            return True
        # Otherwise, token == lemma may be correct (e.g., "senatus")
        return False

    return False


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def process_text(text: str) -> tuple[str, str, list[dict]]:
    """
    Run the CLTK NLP pipeline on a text segment.

    Returns:
        tokens_str  — whitespace-joined token strings
        lemmas_str  — whitespace-joined lemma strings
        log_entries — list of dicts: {token, proposed_lemma} for
                       suspicious/missing lemmas
    """
    if not text or not text.strip():
        return '', '', []

    doc = nlp.analyze(text=text)

    tokens: list[str] = []
    lemmas: list[str] = []
    log_entries: list[dict] = []

    for word in doc.words:
        token = word.string or ''
        lemma = word.lemma or ''

        tokens.append(token)
        lemmas.append(lemma)

        if lemma_is_suspicious(token, lemma):
            log_entries.append({
                'token': token,
                'proposed_lemma': lemma if lemma else '<EMPTY>',
            })

    tokens_str = ' '.join(tokens)
    lemmas_str = ' '.join(lemmas)

    return tokens_str, lemmas_str, log_entries


# ---------------------------------------------------------------------------
# CSV processing
# ---------------------------------------------------------------------------

def process_csv(input_path: str, output_path: str, label: str) -> int:
    """
    Read a normalized CSV, lemmatize each row, write augmented CSV.

    Returns the number of rows processed.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found:\n  {input_path}")
        sys.exit(1)

    rows_out: list[dict] = []
    total_log_entries: list[dict] = []

    with open(input_path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        fieldnames_in = list(reader.fieldnames)

        if 'text' not in fieldnames_in:
            print(f"ERROR: No 'text' column in {label} CSV.")
            sys.exit(1)

        all_rows = list(reader)

    total = len(all_rows)
    fieldnames_out = fieldnames_in + ['tokens', 'lemmas']

    print(f"\n{'='*60}")
    print(f"  Processing {label}: {total} rows")
    print(f"{'='*60}")

    start_time = time.time()

    for idx, row in enumerate(all_rows):
        seg_id = row.get('segment_id', f'row_{idx}')
        text   = row['text']

        tokens_str, lemmas_str, log_entries = process_text(text)

        row['tokens'] = tokens_str
        row['lemmas'] = lemmas_str
        rows_out.append(row)

        # Tag each log entry with the segment_id
        for entry in log_entries:
            entry['segment_id'] = seg_id
        total_log_entries.extend(log_entries)

        # Progress
        if (idx + 1) % 20 == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx + 1:4d}/{total}] "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                  f"({rate:.1f} row/s)")

    # Write augmented CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames_out)
        writer.writeheader()
        writer.writerows(rows_out)

    # Write lemma log for this file
    write_lemma_log(total_log_entries, label)

    elapsed_total = time.time() - start_time
    print(f"  → {total} rows written to {output_path}")
    print(f"  → {len(total_log_entries)} suspicious/missing lemmas logged")
    print(f"  → Total time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    return total


def write_lemma_log(entries: list[dict], label: str):
    """
    Append lemma-log entries to the shared log file.

    Writes header only if the file is new.  Uses the label in a comment
    line to separate batches.
    """
    file_exists = os.path.exists(UNRECOGNIZED_LOG)

    with open(UNRECOGNIZED_LOG, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh,
                                fieldnames=['segment_id', 'token',
                                            'proposed_lemma'])

        if not file_exists:
            writer.writeheader()

        # Add a separator comment (as a row with empty fields won't confuse
        # CSV readers — but a comment line starting with # is safer)
        fh.write(f"# --- {label}: {len(entries)} entries "
                 f"({datetime.now().isoformat()}) ---\n")
        writer.writerows(entries)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(input_path: str, output_path: str, label: str, expected_rows: int):
    """Quick sanity checks on the lemmatized output."""
    with open(output_path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames

    n = len(rows)
    ok = True

    print(f"\n  {label}:")
    print(f"    Rows: {n} (expected {expected_rows})")

    if n != expected_rows:
        print(f"    ✗  Row count mismatch!")
        ok = False
    else:
        print(f"    ✓  Row count matches")

    # Check required columns
    for col in ('tokens', 'lemmas', 'text'):
        if col not in fieldnames:
            print(f"    ✗  Missing column: {col}")
            ok = False
            continue
        present = sum(1 for r in rows if r.get(col, '').strip())
        empty  = n - present
        if empty > 0:
            print(f"    ⚠  {empty} row(s) have empty '{col}' column")
        else:
            print(f"    ✓  All rows have '{col}'")

    # Spot-check: first chapter
    if rows:
        first = rows[0]
        seg_id = first.get('segment_id', '?')
        text_preview = first.get('text', '')[:80]
        tok_preview  = first.get('tokens', '')[:80]
        lem_preview  = first.get('lemmas', '')[:80]
        print(f"    Spot-check [{seg_id}]:")
        print(f"      text:   {text_preview}…")
        print(f"      tokens: {tok_preview}…")
        print(f"      lemmas: {lem_preview}…")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Tokenization + Lemmatization (CLTK + Stanza Latin)")
    print("=" * 60)
    print(f"  Log: {UNRECOGNIZED_LOG}")
    print()

    # Remove old log if it exists (we're rebuilding)
    if os.path.exists(UNRECOGNIZED_LOG):
        os.remove(UNRECOGNIZED_LOG)
        print("  (Previous lemma log removed)")

    overall_ok = True

    # --- Chapters ---
    n_ch = process_csv(INPUTS['chapters'], OUTPUTS['chapters'], 'chapters')
    overall_ok &= validate(INPUTS['chapters'], OUTPUTS['chapters'],
                           'corpus_chapters_normalized_lemmatized', n_ch)

    # --- Books ---
    n_bk = process_csv(INPUTS['books'], OUTPUTS['books'], 'books')
    overall_ok &= validate(INPUTS['books'], OUTPUTS['books'],
                           'corpus_books_normalized_lemmatized', n_bk)

    # --- Summary ---
    print(f"\n{'='*60}")
    if overall_ok:
        print(f"  ✓  Lemmatization complete.")
    else:
        print(f"  ⚠  Lemmatization complete with warnings (see above).")
    print(f"{'='*60}")
    print(f"\n  Chapter-level: {n_ch} rows → {OUTPUTS['chapters']}")
    print(f"  Book-level:    {n_bk} rows → {OUTPUTS['books']}")
    print(f"  Lemma log:              → {UNRECOGNIZED_LOG}")
    print(f"\nNext step: scripts/06_stylo_export.py (plain-text export for R/stylo)")


if __name__ == '__main__':
    main()
