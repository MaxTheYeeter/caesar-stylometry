#!/usr/bin/env python3
"""
scripts/07_features_words.py

Build style feature matrices emphasising content‑independent signals.

Reads:
    data/corpus/corpus_books_normalized_lemmatized.csv
    data/corpus/corpus_chapters_normalized_lemmatized.csv

Produces (in outputs/):
    Book‑level MFW matrices (N = 100, 200, 300) — tokens & lemmas:
        features_mfw100_tokens_books.csv
        features_mfw200_tokens_books.csv
        features_mfw300_tokens_books.csv
        features_mfw100_lemmas_books.csv
        features_mfw200_lemmas_books.csv
        features_mfw300_lemmas_books.csv

    Chapter‑level MFW matrices:
        features_mfw100_tokens_chapters.csv
        features_mfw100_lemmas_chapters.csv

    Function‑word matrices:
        features_function_words_tokens_books.csv
        features_function_words_tokens_chapters.csv
        features_function_words_lemmas_books.csv
        features_function_words_lemmas_chapters.csv

All matrices use RELATIVE frequencies (per‑segment proportions) to control
for unequal book lengths (range 4,104–37,106 tokens; 3.1× ratio).

Each matrix includes a metadata column 'author_group':
    'caesar'    — DBG Books I–VII (Caesar, authentic)
    'hirtius'   — DBG Book VIII   (Aulus Hirtius, control)
    'caesar_dbc'— DBC Complete     (Caesar, later work, anchor)

Feature columns are the words/lemmas themselves.  Values are proportions
in [0, 1].
"""

import csv
import os
import sys
from collections import Counter

# --- Raise CSV field size limit ---------------------------------------------
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# Configuration
# ===========================================================================

MFW_SIZES = [100, 200, 300]          # top‑N most‑frequent‑word sizes to build

# All feature‑matrix permutations we will build
MATRIX_SPECS = {
    # --- MFW matrices ---
    **{f'mfw{n}_tokens':  {'representation': 'tokens',  'mfw_n': n}
       for n in MFW_SIZES},
    **{f'mfw{n}_lemmas':  {'representation': 'lemmas',  'mfw_n': n}
       for n in MFW_SIZES},
    # --- Function‑word matrices ---
    'function_words_tokens': {'representation': 'tokens',  'mfw_n': None},
    'function_words_lemmas': {'representation': 'lemmas',  'mfw_n': None},
}

# Aggregation levels to build
AGGREGATION_LEVELS = ['books', 'chapters']


# ===========================================================================
# Curated Latin function‑word list  (normalised: lowercase, j→i, v→u)
# ===========================================================================
# These are the content‑independent words that stylometry relies on:
# conjunctions, prepositions, pronouns, particles, forms of *esse*, and
# very common adverbs serving grammatical functions.

FUNCTION_WORDS: set[str] = {
    # ── Conjunctions ──────────────────────────────────────────────────────
    'et', 'que', 'atque', 'ac', 'nec', 'neque', 'aut', 'uel', 'si', 'nisi',
    'quod', 'quia', 'quoniam', 'cum', 'ut', 'ne', 'dum', 'donec',
    'priusquam', 'antequam', 'postquam', 'ubi', 'simul', 'quasi',
    'tamquam', 'quam', 'an', 'siue', 'seu', 'sin', 'quin', 'quominus',
    'etsi', 'tametsi', 'quamquam', 'quamuis', 'licet',
    'sed', 'uerum', 'autem', 'at', 'tamen', 'ceterum', 'atqui',
    'enim', 'nam', 'igitur', 'itaque', 'ergo', 'ideo', 'quare',
    'quamobrem', 'quocirca', 'quapropter', 'unde',

    # ── Prepositions ──────────────────────────────────────────────────────
    'a', 'ab', 'abs', 'ad', 'adversus', 'aduersus', 'ante', 'apud',
    'circum', 'circa', 'contra', 'coram', 'cum', 'de', 'e', 'ex',
    'erga', 'extra', 'in', 'infra', 'inter', 'intra', 'iuxta',
    'ob', 'penes', 'per', 'pone', 'post', 'prae', 'praeter',
    'pro', 'prope', 'propter', 'secundum', 'sine', 'sub', 'subter',
    'super', 'supra', 'trans', 'usque', 'uersus', 'ultra', 'cis',
    'citra', 'clam',

    # ── Personal / reflexive pronouns ─────────────────────────────────────
    'ego', 'me', 'mei', 'mihi', 'tu', 'te', 'tui', 'tibi',
    'nos', 'nobis', 'nosmet', 'uos', 'uobis', 'uestrum', 'uestri',
    'se', 'sui', 'sibi', 'secum', 'sese',

    # ── Demonstrative pronouns ────────────────────────────────────────────
    'hic', 'haec', 'hoc', 'hunc', 'hanc', 'huius', 'huic',
    'hi', 'hae', 'hos', 'has', 'horum', 'harum', 'his',
    'ille', 'illa', 'illud', 'illum', 'illam', 'illius', 'illi',
    'illi', 'illae', 'illos', 'illas', 'illorum', 'illarum', 'illis',
    'iste', 'ista', 'istud', 'istum', 'istam', 'istius', 'isti',
    'isti', 'istae', 'istos', 'istas', 'istorum', 'istarum', 'istis',
    'is', 'ea', 'id', 'eum', 'eam', 'eius', 'ei',
    'ei', 'eae', 'eos', 'eas', 'eorum', 'earum', 'eis', 'iis',

    # ── Relative / interrogative / indefinite pronouns ────────────────────
    'qui', 'quae', 'quod', 'quem', 'quam', 'cuius', 'cui',
    'qui', 'quae', 'quos', 'quas', 'quorum', 'quarum', 'quibus',
    'quis', 'quid', 'quem', 'quo', 'qua',
    'aliquis', 'aliqui', 'aliqua', 'aliquod', 'aliquid',
    'quisquam', 'quicquam', 'quidam', 'quaedam', 'quoddam',
    'quisque', 'quaeque', 'quodque', 'quicumque', 'quaecumque',
    'uter', 'utra', 'utrum', 'uterque', 'utraque', 'utrumque',
    'unusquisque',

    # ── Intensive ─────────────────────────────────────────────────────────
    'ipse', 'ipsa', 'ipsum', 'ipsius', 'ipsi',
    'ipsi', 'ipsae', 'ipsos', 'ipsas', 'ipsorum', 'ipsarum', 'ipsis',
    'idem', 'eadem', 'idem',

    # ── Particles & common adverbs ────────────────────────────────────────
    'non', 'haud', 'ne', 'num', 'utrum', 'enim', 'quidem',
    'quoque', 'etiam', 'etiamsi', 'uero', 'certe',
    'ita', 'sic', 'tam', 'adeo', 'fere', 'paene', 'uix',
    'prope', 'satis', 'bene', 'male', 'forte', 'fortasse',
    'iam', 'nunc', 'tunc', 'tum', 'hic', 'huc', 'hinc',
    'illic', 'illuc', 'illinc', 'ibi', 'inde', 'ubicumque',
    'nusquam', 'semper', 'saepe', 'interim', 'subito', 'statim',
    'mox', 'sane', 'scilicet', 'uidelicet', 'praeterea',
    'denique', 'postea', 'antea', 'nuper', 'olim', 'quondam',
    'adhuc', 'ultro', 'sponte', 'frustra',
    'magis', 'potius', 'nimis', 'parum', 'procul',

    # ── Forms of *esse* (to be) ───────────────────────────────────────────
    'sum', 'es', 'est', 'sumus', 'estis', 'sunt',
    'eram', 'eras', 'erat', 'eramus', 'eratis', 'erant',
    'ero', 'eris', 'erit', 'erimus', 'eritis', 'erunt',
    'fui', 'fuisti', 'fuit', 'fuimus', 'fuistis', 'fuerunt',
    'fueram', 'fueras', 'fuerat', 'fueramus', 'fueratis', 'fuerant',
    'fuero', 'fueris', 'fuerit', 'fuerimus', 'fueritis', 'fuerint',
    'sim', 'sis', 'sit', 'simus', 'sitis', 'sint',
    'essem', 'esses', 'esset', 'essemus', 'essetis', 'essent',
    'esse', 'fore', 'fuisse', 'futurus',

    # ── Forms of *posse* (to be able) ─────────────────────────────────────
    'possum', 'potes', 'potest', 'possumus', 'potestis', 'possunt',
    'poteram', 'poteras', 'poterat', 'poteramus', 'poteratis', 'poterant',
    'potero', 'poteris', 'poterit', 'poterimus', 'poteritis', 'poterunt',
    'potui', 'potuisti', 'potuit', 'potuimus', 'potuistis', 'potuerunt',
    'possim', 'possis', 'possit', 'possimus', 'possitis', 'possint',
    'possem', 'posses', 'posset', 'possemus', 'possetis', 'possent',
    'posse', 'potuisse',

    # ── Common irregular verb forms (auxiliary‑like usage) ────────────────
    'uolo', 'uis', 'uult', 'uolumus', 'uultis', 'uolunt',
    'uellem', 'uelles', 'uellet', 'uellemus', 'uelletis', 'uellent',
    'uelim', 'uelis', 'uelit', 'uelimus', 'uelitis', 'uelint',
    'nolo', 'nonuis', 'nonuult', 'nolumus', 'nonuultis', 'nolunt',
    'malo', 'mauis', 'mauult', 'malumus', 'mauultis', 'malunt',
    'eo', 'is', 'it', 'imus', 'itis', 'eunt',
    'fero', 'fers', 'fert', 'ferimus', 'fertis', 'ferunt',
    'fio', 'fis', 'fit', 'fimus', 'fitis', 'fiunt',
}


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
# Feature extraction
# ===========================================================================

def tokenize_column(text: str) -> list[str]:
    """Split a whitespace‑joined token/lemma string into a list."""
    if not text or not text.strip():
        return []
    return text.strip().split()


def build_frequency_vector(tokens: list[str],
                           feature_set: set[str]) -> dict[str, float]:
    """
    Build a relative‑frequency dict for a single segment.

    Returns {feature_word: proportion} for every word in feature_set.
    If a feature never appears, its proportion is 0.0.
    The sum of all proportions ≈ 1.0 (minor rounding).
    """
    total = len(tokens)
    if total == 0:
        return {f: 0.0 for f in feature_set}

    counts = Counter(tokens)

    # Relative frequency per feature
    return {f: counts.get(f, 0) / total for f in feature_set}


def get_top_n_features(all_tokens: list[list[str]], n: int) -> list[str]:
    """
    Compute the top‑N most‑frequent tokens across the entire corpus.

    all_tokens is a list of token lists (one list per segment).
    Returns a list of the N most frequent token strings, in descending
    frequency order (most frequent first).
    """
    global_counter: Counter = Counter()
    for tokens in all_tokens:
        global_counter.update(tokens)

    # Return the top N, most frequent first
    return [word for word, _count in global_counter.most_common(n)]


def build_feature_set(representation: str, mfw_n: int | None,
                      all_tokens: list[list[str]]) -> list[str]:
    """
    Build the ordered feature list for a matrix specification.

    If mfw_n is not None, returns the top‑N most‑frequent words.
    If mfw_n is None, returns the sorted function‑word list (but only
    those words that actually appear in the corpus — avoids zero‑variance
    features).
    """
    if mfw_n is not None:
        return get_top_n_features(all_tokens, mfw_n)

    # Function words: intersect the curated list with words actually
    # present in the corpus.
    corpus_vocab: set[str] = set()
    for tokens in all_tokens:
        corpus_vocab.update(tokens)

    # Sort for deterministic column order
    present = sorted(FUNCTION_WORDS & corpus_vocab)
    return present


# ===========================================================================
# Matrix builder
# ===========================================================================

def build_matrix(rows: list[dict],
                 feature_list: list[str],
                 representation: str) -> list[dict]:
    """
    Build a matrix as a list of row dicts.

    Each output row has:
        segment_id, author_group, work, book,
        total_tokens,
        feature_1, feature_2, …  (relative frequencies)
    """
    matrix: list[dict] = []

    for row in rows:
        tokens = tokenize_column(row.get(representation, ''))

        vec = build_frequency_vector(tokens, set(feature_list))

        out_row = {
            'segment_id':   row.get('segment_id', ''),
            'author_group': get_author_group(row),
            'work':         row.get('work', ''),
            'book':         row.get('book', ''),
            'total_tokens': len(tokens),
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
                  'total_tokens'] + feature_list

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(matrix)

    print(f"  → {len(matrix)} rows × {len(feature_list)} features → {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("  Feature Matrix Builder — MFW + Function Words")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load corpora
    # ------------------------------------------------------------------
    print("\nLoading corpora …")
    books_rows    = load_csv(BOOKS_CSV)
    chapters_rows = load_csv(CHAPTERS_CSV)

    print(f"  Books:    {len(books_rows)} rows")
    print(f"  Chapters: {len(chapters_rows)} rows")

    # ------------------------------------------------------------------
    # 2. Pre‑extract all token / lemma sequences
    #    (needed for global top‑N computation)
    # ------------------------------------------------------------------
    print("\nTokenising …")

    # Books
    books_tokens = [tokenize_column(r['tokens']) for r in books_rows]
    books_lemmas = [tokenize_column(r['lemmas']) for r in books_rows]

    # Chapters
    chapters_tokens = [tokenize_column(r['tokens']) for r in chapters_rows]
    chapters_lemmas = [tokenize_column(r['lemmas']) for r in chapters_rows]

    # ------------------------------------------------------------------
    # 3. Build matrices for each specification
    # ------------------------------------------------------------------
    total_matrices = 0

    for spec_name, spec in MATRIX_SPECS.items():
        rep  = spec['representation']
        mfw  = spec['mfw_n']

        for agg in AGGREGATION_LEVELS:
            # Select the right data
            if agg == 'books':
                rows = books_rows
                all_tokens = (books_tokens if rep == 'tokens'
                              else books_lemmas)
            else:
                rows = chapters_rows
                all_tokens = (chapters_tokens if rep == 'tokens'
                              else chapters_lemmas)

            # Build feature list
            feature_list = build_feature_set(rep, mfw, all_tokens)

            if not feature_list:
                print(f"\n  ⚠  No features for {spec_name}_{agg} — skipping")
                continue

            # Describe what we're building
            feat_desc = (f"top-{mfw} MFW" if mfw else
                         f"{len(feature_list)} function words "
                         f"(of {len(FUNCTION_WORDS)} curated)")
            print(f"\n{'─'*50}")
            print(f"  {spec_name}_{agg}")
            print(f"  Representation: {rep} | Features: {feat_desc}")
            print(f"{'─'*50}")

            # Build matrix
            matrix = build_matrix(rows, feature_list, rep)

            # Write
            fname = f'features_{spec_name}_{agg}.csv'
            fpath = os.path.join(OUTPUT_DIR, fname)
            write_matrix(matrix, feature_list, fpath)

            total_matrices += 1

            # --- Quick validation ---
            validate_matrix(matrix, feature_list, spec_name, agg)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ✓  {total_matrices} feature matrices written to {OUTPUT_DIR}/")
    print(f"{'='*60}")

    print(f"\n  Matrix key:")
    print(f"    author_group values:")
    print(f"      'caesar'     — DBG Books I–VII")
    print(f"      'hirtius'    — DBG Book VIII")
    print(f"      'caesar_dbc' — DBC Complete")
    print(f"\n  Next step: scripts/08_stylo_export.py (plain‑text export for R/stylo)")
    print(f"              or proceed to PCA / distance analysis with these matrices.")


# ===========================================================================
# Validation
# ===========================================================================

def validate_matrix(matrix: list[dict], feature_list: list[str],
                    spec_name: str, agg: str):
    """Print a compact validation summary for a matrix."""
    n_rows = len(matrix)
    n_feat = len(feature_list)
    author_counts = Counter(r['author_group'] for r in matrix)

    # Check for constant‑zero features (no variance)
    zero_variance: list[str] = []
    for feat in feature_list:
        vals = [r[feat] for r in matrix]
        if all(v == 0.0 for v in vals):
            zero_variance.append(feat)

    # Check for row sums (should be ≈1.0 for MFW; may be <1.0 for function
    # words that don't cover the full vocabulary)
    sum_min = min(sum(r[f] for f in feature_list) for r in matrix)
    sum_max = max(sum(r[f] for f in feature_list) for r in matrix)

    print(f"    Rows: {n_rows} | Features: {n_feat}")
    print(f"    Author distribution: "
          f"caesar={author_counts['caesar']} "
          f"hirtius={author_counts['hirtius']} "
          f"caesar_dbc={author_counts['caesar_dbc']}")
    print(f"    Feature coverage (row sum): "
          f"min={sum_min:.3f}  max={sum_max:.3f}")

    if zero_variance:
        if len(zero_variance) <= 5:
            print(f"    ⚠  {len(zero_variance)} zero‑variance feature(s): "
                  f"{zero_variance}")
        else:
            print(f"    ⚠  {len(zero_variance)} zero‑variance features "
                  f"(first 5: {zero_variance[:5]})")


if __name__ == '__main__':
    main()
