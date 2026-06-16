#!/usr/bin/env python3
"""
scripts/21_calibration_analysis.py

Runs the SAME analytical machinery (feature extraction, Delta/cosine distance,
DBC-anchor Spearman, Mantel test, exact permutation + bootstrap CI) on the
calibration corpora and produces a direct side-by-side comparison with Caesar.

Corpora tested:
  1. Cicero Ad Atticum (merged yearly) — POSITIVE CONTROL
     Known serial composition, 68–44 BC. n = 10 units.
     Anchor: latest year (44 BC, Cicero's last dated year-unit).
     Mantel gap: |known_date_i − known_date_j|.

  2. DBC Pseudo-Books (7 sequential chunks) — NEGATIVE CONTROL
     Single concentrated work (49–48 BC), arbitrarily segmented.
     Anchor: pseudo-book 7.
     Mantel gap: |i − j| (narrative order, not time).

  3. Caesar DBG I–VII — EXPERIMENTAL
     Results copied from outputs/robustness_summary.csv (full-corpus baseline).

Outputs:
  outputs/calibration_results.csv — side-by-side comparison table
  figures/calibration_comparison.png — grouped bar chart of |r| by corpus
"""

import csv
import os
import sys
import re
import warnings
from collections import OrderedDict, Counter
from itertools import permutations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance_fn

csv.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')
CORPUS_DIR   = os.path.join(PROJECT_ROOT, 'data', 'corpus')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Monte Carlo permutations for Cicero (10! = 3.6M, too large for exact) ──
N_PERM_CICERO = 100_000
# Exact for DBC pseudo-books (7! = 5,040)
N_PERM_DBC    = 5_040

# ── Labels ─────────────────────────────────────────────────────────────
ROMAN_10 = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
ROMAN_7  = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']


# ═══════════════════════════════════════════════════════════════════════
# 1. FUNCTION WORD LIST (same as Caesar)
# ═══════════════════════════════════════════════════════════════════════
def load_caesar_function_words():
    """
    Extract the exact 299 function-word features from the Caesar matrix.
    This guarantees the calibration uses identical features.
    """
    path = os.path.join(OUTPUTS_DIR,
                        'features_function_words_tokens_books.csv')
    with open(path, 'r', newline='') as f:
        header = next(csv.reader(f))
    meta_cols = {'segment_id', 'author_group', 'work', 'book', 'total_tokens'}
    return [c for c in header if c not in meta_cols]


def build_corpus_vocab(texts):
    """Build vocabulary from list of (tokenised or whitespace-split) texts."""
    vocab = Counter()
    for t in texts:
        vocab.update(t.split())
    return vocab


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION (mirrors Scripts 07/08)
# ═══════════════════════════════════════════════════════════════════════
def load_calibration_corpus(csv_path):
    """
    Load a calibration CSV. Returns:
      units: list of dicts with text, tokens, lemmas, order_index, known_date
      anchor_text: str — text of the last unit (the anchor)
      anchor_tokens: str — tokenised anchor
      anchor_lemmas: str — lemmatised anchor
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Sort by order_index
    rows.sort(key=lambda r: int(r.get('order_index', 0)))

    units = []
    for r in rows:
        units.append({
            'unit_id':    r['unit_id'],
            'text':       r['text'],
            'tokens':     r.get('tokens', r['text']),
            'lemmas':     r.get('lemmas', r['text']),
            'order':      int(r['order_index']),
            'known_date': r.get('known_date', ''),
        })

    # Last unit = anchor
    anchor = units[-1]

    return units, anchor


def extract_function_word_features(units, anchor, func_words, level='tokens'):
    """
    Build proportion vectors using the Caesar function-word list.
    level: 'tokens' or 'lemmas'
    """
    field = 'tokens' if level == 'tokens' else 'lemmas'

    # Build vocabulary from all units
    all_texts = [u[field] for u in units] + [anchor[field]]
    vocab = build_corpus_vocab(all_texts)
    present_fw = sorted(f for f in func_words if f in vocab)

    n_units = len(units)

    # Build matrix (n_units + 1 for anchor, arranged as: units 1..n, anchor)
    n_total = n_units + 1
    X = np.zeros((n_total, len(present_fw)))

    for i, u in enumerate(units):
        tokens = u[field].split()
        total = len(tokens)
        if total == 0:
            continue
        counts = Counter(tokens)
        for j, fw in enumerate(present_fw):
            X[i, j] = counts.get(fw, 0) / total

    # Anchor row
    anchor_tokens = anchor[field].split()
    anchor_total = len(anchor_tokens)
    if anchor_total > 0:
        counts = Counter(anchor_tokens)
        for j, fw in enumerate(present_fw):
            X[n_units, j] = counts.get(fw, 0) / anchor_total

    return X, present_fw


def extract_mfw_features(units, anchor, mfw_n, level='tokens'):
    """
    Build proportion vectors using the top N most frequent words.
    """
    field = 'tokens' if level == 'tokens' else 'lemmas'

    all_texts = [u[field] for u in units] + [anchor[field]]
    vocab = build_corpus_vocab(all_texts)
    top_n = [w for w, _ in vocab.most_common(mfw_n)]

    n_units = len(units)
    n_total = n_units + 1
    X = np.zeros((n_total, mfw_n))

    for i, u in enumerate(units):
        tokens = u[field].split()
        total = len(tokens)
        if total == 0:
            continue
        counts = Counter(tokens)
        for j, w in enumerate(top_n):
            X[i, j] = counts.get(w, 0) / total

    anchor_tokens = anchor[field].split()
    anchor_total = len(anchor_tokens)
    if anchor_total > 0:
        counts = Counter(anchor_tokens)
        for j, w in enumerate(top_n):
            X[n_units, j] = counts.get(w, 0) / anchor_total

    return X, top_n


def extract_char_ngram_features(units, anchor, n_size):
    """
    Build proportion vectors for character n-grams.
    N-grams are extracted from the 'text' field (always tokens for char n-grams).
    """
    def get_ngrams(text, n):
        text = re.sub(r'\s+', '_', text)  # preserve word boundaries
        ngrams = Counter()
        for i in range(len(text) - n + 1):
            ngrams[text[i:i + n]] += 1
        return ngrams

    # Build global n-gram vocabulary
    global_ngrams = Counter()
    for u in units:
        global_ngrams.update(get_ngrams(u['text'], n_size))
    global_ngrams.update(get_ngrams(anchor['text'], n_size))

    # Take top 500 (matching Caesar's char3gram), 1000 for char4gram
    max_feats = {2: None, 3: 500, 4: 1000}
    top_n = max_feats.get(n_size, 500)
    if top_n:
        top_ngrams = [g for g, _ in global_ngrams.most_common(top_n)]
    else:
        top_ngrams = sorted(global_ngrams.keys())

    n_units = len(units)
    n_total = n_units + 1
    X = np.zeros((n_total, len(top_ngrams)))

    for i, u in enumerate(units):
        ngrams = get_ngrams(u['text'], n_size)
        total = sum(ngrams.values())
        if total == 0:
            continue
        for j, ng in enumerate(top_ngrams):
            X[i, j] = ngrams.get(ng, 0) / total

    anchor_ngrams = get_ngrams(anchor['text'], n_size)
    anchor_total = sum(anchor_ngrams.values())
    if anchor_total > 0:
        for j, ng in enumerate(top_ngrams):
            X[n_units, j] = anchor_ngrams.get(ng, 0) / anchor_total

    return X, top_ngrams


# ═══════════════════════════════════════════════════════════════════════
# 3. DISTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
def compute_delta_matrix(X):
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0, ddof=0)
    std[std < 1e-12] = 1e-12
    Z = (X - mean) / std
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.mean(np.abs(Z[i] - Z[j]))
            D[i, j] = d
            D[j, i] = d
    return D


def compute_cosine_matrix(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance_fn(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def matrix_to_upper_triangle(D):
    n = D.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(D[i, j])
    return np.array(vals)


# ═══════════════════════════════════════════════════════════════════════
# 4. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════
def dbc_anchor_test(dists_to_anchor, method='exact', n_mc=100_000, rng=None):
    """
    Spearman r between order (1..n) and distance to anchor.
    One-sided: negative r = later units closer to anchor (annual drift).
    method: 'exact' (n ≤ 7) or 'mc' (Monte Carlo).
    Returns (r_obs, p_value, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = len(dists_to_anchor)
    books = np.arange(1, n + 1, dtype=float)
    r_obs, _ = stats.spearmanr(books, dists_to_anchor)

    if method == 'exact':
        count = 0
        total = 0
        for perm in permutations(range(n)):
            d_perm = dists_to_anchor[list(perm)]
            r_perm, _ = stats.spearmanr(books, d_perm)
            if r_perm <= r_obs:
                count += 1
            total += 1
        p_val = count / total
    else:
        count = 0
        for _ in range(n_mc):
            d_perm = rng.permutation(dists_to_anchor)
            r_perm, _ = stats.spearmanr(books, d_perm)
            if r_perm <= r_obs:
                count += 1
        p_val = (count + 1) / (n_mc + 1)

    # Bootstrap CI
    boot_rs = []
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        r_boot, _ = stats.spearmanr(books[idx], dists_to_anchor[idx])
        boot_rs.append(r_boot)
    boot_rs = np.array(boot_rs)
    ci_low = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)

    return r_obs, p_val, ci_low, ci_high


def mantel_test(D_style, G_gap, method='exact', n_mc=100_000, rng=None):
    """
    Mantel test: Pearson r(upper(D_style), upper(G_gap)).
    One-sided: positive r (style distance grows with time gap).
    Returns (r_obs, p_value, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = D_style.shape[0]
    style_vec = matrix_to_upper_triangle(D_style)
    gap_vec   = matrix_to_upper_triangle(G_gap)
    r_obs, _  = stats.pearsonr(style_vec, gap_vec)

    if method == 'exact':
        count = 0
        total = 0
        for perm in permutations(range(n)):
            plist = list(perm)
            D_perm = D_style[plist][:, plist]
            s_perm = matrix_to_upper_triangle(D_perm)
            r_perm, _ = stats.pearsonr(s_perm, gap_vec)
            if r_perm >= r_obs:
                count += 1
            total += 1
        p_val = count / total
    else:
        count = 0
        for _ in range(n_mc):
            plist = rng.permutation(n).tolist()
            D_perm = D_style[plist][:, plist]
            s_perm = matrix_to_upper_triangle(D_perm)
            r_perm, _ = stats.pearsonr(s_perm, gap_vec)
            if r_perm >= r_obs:
                count += 1
        p_val = (count + 1) / (n_mc + 1)

    # Bootstrap CI
    style_vec = matrix_to_upper_triangle(D_style)
    gap_vec   = matrix_to_upper_triangle(G_gap)
    n_pairs = len(style_vec)
    boot_rs = []
    for _ in range(10000):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        r_boot, _ = stats.pearsonr(style_vec[idx], gap_vec[idx])
        boot_rs.append(r_boot)
    boot_rs = np.array(boot_rs)
    ci_low = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)

    return r_obs, p_val, ci_low, ci_high


def build_year_gap_matrix(years):
    """Build |year_i − year_j| matrix from list of years."""
    years = np.array(years, dtype=float)
    n = len(years)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = abs(years[i] - years[j])
    return G


# ═══════════════════════════════════════════════════════════════════════
# 5. LOAD CAESAR BASELINE RESULTS
# ═══════════════════════════════════════════════════════════════════════
def load_caesar_baseline():
    """
    Load Caesar results from robustness_summary.csv (full corpus baseline).
    Returns dict mapping (feature_set, distance) → {dbc_r, dbc_p, mantel_r, mantel_p}.
    """
    path = os.path.join(OUTPUTS_DIR, 'robustness_summary.csv')
    if not os.path.exists(path):
        print("  ⚠ Caesar robustness_summary.csv not found. "
              "Caesar bars will be omitted.")
        return {}

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader
                if r['condition'] == 'Full corpus (no exclusions)']

    # Map robustness feature_set names to our simplified labels
    fs_map = {
        'Function Words': 'function_words',
        'MFW 100': 'mfw100',
        'MFW 200': 'mfw200',
        'MFW 300': 'mfw300',
        'Char 2-grams': 'char2gram',
        'Char 3-grams': 'char3gram',
        'Char 4-grams': 'char4gram',
        'Function Words (lemmas)': 'function_words_lemmas',
        'MFW 100 (lemmas)': 'mfw100_lemmas',
        'MFW 200 (lemmas)': 'mfw200_lemmas',
        'MFW 300 (lemmas)': 'mfw300_lemmas',
    }

    results = {}
    for r in rows:
        key = (fs_map.get(r['feature_set'], r['feature_set']),
               r['distance'])
        results[key] = {
            'dbc_r': float(r['dbc_r']),
            'dbc_p': float(r['dbc_p']),
            'mantel_r': float(r['mantel_r']),
            'mantel_p': float(r['mantel_p']),
        }

    # Also load token-level only for char n-grams (which don't have lexical variants)
    # Copy token results to 'all' lexical for char n-grams
    return results


# ═══════════════════════════════════════════════════════════════════════
# 6. FEATURE SET DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════
# Feature sets to extract for calibration corpora:
# word-based (tokens + lemmas) for function words and MFW; char n-grams (tokens only)

WORD_FEATURE_SETS = OrderedDict([
    ('function_words',   {'label': 'Function Words',    'type': 'func'}),
    ('mfw100',           {'label': 'MFW 100',           'type': 'mfw', 'n': 100}),
    ('mfw200',           {'label': 'MFW 200',           'type': 'mfw', 'n': 200}),
    ('mfw300',           {'label': 'MFW 300',           'type': 'mfw', 'n': 300}),
])

CHAR_FEATURE_SETS = OrderedDict([
    ('char2gram',        {'label': 'Char 2-grams',      'n': 2}),
    ('char3gram',        {'label': 'Char 3-grams',      'n': 3}),
    ('char4gram',        {'label': 'Char 4-grams',      'n': 4}),
])

DISTANCES = ['Delta', 'Cosine']


# ═══════════════════════════════════════════════════════════════════════
# 7. MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
def analyze_corpus(name, csv_path, gap_type, perm_method, func_words,
                   rng):
    """
    Run full analysis on one calibration corpus.
    gap_type: 'year' (Cicero) or 'narrative' (DBC pseudo-books)
    perm_method: 'exact' or 'mc'
    Returns list of result dicts.
    """
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    if not os.path.exists(csv_path):
        print(f"  ✗ CSV not found: {csv_path}")
        return []

    units, anchor = load_calibration_corpus(csv_path)
    n_units = len(units)
    print(f"  Units: {n_units}  |  Anchor: {anchor['unit_id']}")

    # ── Build gap matrix ─────────────────────────────────────────
    if gap_type == 'year':
        years_list = [int(u['known_date']) for u in units]
        G_gap = build_year_gap_matrix(years_list)
        print(f"  Year range: {min(years_list)} to {max(years_list)}")
    else:
        G_gap = build_year_gap_matrix(list(range(1, n_units + 1)))
        print(f"  Gap type: narrative order (|i − j|)")

    results = []

    # ── Word features × tokens + lemmas ───────────────────────────
    for feat_key, cfg in WORD_FEATURE_SETS.items():
        for level in ['tokens', 'lemmas']:
            level_label = 'lemmas' if level == 'lemmas' else 'tokens'
            full_label = f"{cfg['label']} ({level_label})"

            # Extract features
            if cfg['type'] == 'func':
                X, feat_names = extract_function_word_features(
                    units, anchor, func_words, level=level)
            else:
                X, feat_names = extract_mfw_features(
                    units, anchor, cfg['n'], level=level)

            # Distance matrices
            D_delta = compute_delta_matrix(X)
            D_cos   = compute_cosine_matrix(X)

            # DBC Anchor
            anchor_idx = n_units  # last row
            dbg_idxs = list(range(n_units))

            dbc_delta = np.array([D_delta[i, anchor_idx] for i in dbg_idxs])
            dbc_cos   = np.array([D_cos[i, anchor_idx]   for i in dbg_idxs])

            rD, pD, ciD_lo, ciD_hi = dbc_anchor_test(
                dbc_delta, method=perm_method, n_mc=N_PERM_CICERO, rng=rng)
            rC, pC, ciC_lo, ciC_hi = dbc_anchor_test(
                dbc_cos,   method=perm_method, n_mc=N_PERM_CICERO, rng=rng)

            # Mantel
            D_dbg_delta = D_delta[np.ix_(dbg_idxs, dbg_idxs)]
            D_dbg_cos   = D_cos[np.ix_(dbg_idxs, dbg_idxs)]

            rMD, pMD, ciMD_lo, ciMD_hi = mantel_test(
                D_dbg_delta, G_gap, method=perm_method,
                n_mc=N_PERM_CICERO, rng=rng)
            rMC, pMC, ciMC_lo, ciMC_hi = mantel_test(
                D_dbg_cos,   G_gap, method=perm_method,
                n_mc=N_PERM_CICERO, rng=rng)

            for dist, r_anc, p_anc, ci_lo_a, ci_hi_a, r_man, p_man, ci_lo_m, ci_hi_m in [
                ('Delta',  rD, pD, ciD_lo, ciD_hi, rMD, pMD, ciMD_lo, ciMD_hi),
                ('Cosine', rC, pC, ciC_lo, ciC_hi, rMC, pMC, ciMC_lo, ciMC_hi),
            ]:
                results.append({
                    'corpus': name,
                    'feature_set': feat_key,
                    'feat_label': full_label,
                    'distance': dist,
                    'dbc_r': r_anc,
                    'dbc_p': p_anc,
                    'dbc_ci_low': ci_lo_a,
                    'dbc_ci_high': ci_hi_a,
                    'mantel_r': r_man,
                    'mantel_p': p_man,
                    'mantel_ci_low': ci_lo_m,
                    'mantel_ci_high': ci_hi_m,
                })

            print(f"    {full_label:<28s}  "
                  f"DBC r={rD:+.3f} p={pD:.3f}  "
                  f"Man r={rMD:+.3f} p={pMD:.3f}")

    # ── Character n-gram features ──────────────────────────────────
    for feat_key, cfg in CHAR_FEATURE_SETS.items():
        full_label = cfg['label']

        X, feat_names = extract_char_ngram_features(units, anchor, cfg['n'])

        D_delta = compute_delta_matrix(X)
        D_cos   = compute_cosine_matrix(X)

        anchor_idx = n_units
        dbg_idxs = list(range(n_units))

        dbc_delta = np.array([D_delta[i, anchor_idx] for i in dbg_idxs])
        dbc_cos   = np.array([D_cos[i, anchor_idx]   for i in dbg_idxs])

        rD, pD, ciD_lo, ciD_hi = dbc_anchor_test(
            dbc_delta, method=perm_method, n_mc=N_PERM_CICERO, rng=rng)
        rC, pC, ciC_lo, ciC_hi = dbc_anchor_test(
            dbc_cos,   method=perm_method, n_mc=N_PERM_CICERO, rng=rng)

        D_dbg_delta = D_delta[np.ix_(dbg_idxs, dbg_idxs)]
        D_dbg_cos   = D_cos[np.ix_(dbg_idxs, dbg_idxs)]

        rMD, pMD, ciMD_lo, ciMD_hi = mantel_test(
            D_dbg_delta, G_gap, method=perm_method,
            n_mc=N_PERM_CICERO, rng=rng)
        rMC, pMC, ciMC_lo, ciMC_hi = mantel_test(
            D_dbg_cos,   G_gap, method=perm_method,
            n_mc=N_PERM_CICERO, rng=rng)

        for dist, r_anc, p_anc, ci_lo_a, ci_hi_a, r_man, p_man, ci_lo_m, ci_hi_m in [
            ('Delta',  rD, pD, ciD_lo, ciD_hi, rMD, pMD, ciMD_lo, ciMD_hi),
            ('Cosine', rC, pC, ciC_lo, ciC_hi, rMC, pMC, ciMC_lo, ciMC_hi),
        ]:
            results.append({
                'corpus': name,
                'feature_set': feat_key,
                'feat_label': full_label,
                'distance': dist,
                'dbc_r': r_anc,
                'dbc_p': p_anc,
                'dbc_ci_low': ci_lo_a,
                'dbc_ci_high': ci_hi_a,
                'mantel_r': r_man,
                'mantel_p': p_man,
                'mantel_ci_low': ci_lo_m,
                'mantel_ci_high': ci_hi_m,
            })

        print(f"    {full_label:<28s}  "
              f"DBC r={rD:+.3f} p={pD:.3f}  "
              f"Man r={rMD:+.3f} p={pMD:.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# 8. PLOT
# ═══════════════════════════════════════════════════════════════════════
def make_comparison_plot(all_rows, out_path):
    """
    Grouped bar chart: |DBC r| and |Mantel r| for Cicero, DBC pseudo-books,
    and Caesar across feature sets.
    """
    # Collect feature sets in display order
    all_fs_keys = ['function_words', 'mfw100', 'mfw200', 'mfw300',
                   'char2gram', 'char3gram', 'char4gram']
    all_fs_labels = ['Func.\nWords', 'MFW\n100', 'MFW\n200', 'MFW\n300',
                     'Char\n2-gr', 'Char\n3-gr', 'Char\n4-gr']

    # Group data: corpus → feat_key → [r values for DBC, r values for Mantel]
    corpora = OrderedDict()
    for corpus_name in ['Cicero (positive)', 'DBC pseudo (negative)', 'Caesar DBG']:
        corpora[corpus_name] = {
            'dbc': {k: [] for k in all_fs_keys},
            'mantel': {k: [] for k in all_fs_keys},
        }

    for row in all_rows:
        fs = row['feature_set']
        if fs not in all_fs_keys:
            # Map lemma variants to their base key for combined display
            if '_lemmas' in fs:
                fs = fs.replace('_lemmas', '')
            else:
                continue

        corpus_lookup = {
            'Cicero (positive)': 'Cicero (positive)',
            'Cicero Ad Atticum (yearly)': 'Cicero (positive)',
            'DBC pseudo (negative)': 'DBC pseudo (negative)',
            'DBC Pseudo-Books': 'DBC pseudo (negative)',
            'Caesar DBG': 'Caesar DBG',
        }
        corp = corpus_lookup.get(row['corpus'], row['corpus'])
        if corp not in corpora:
            continue

        corpora[corp]['dbc'][fs].append(row['dbc_r'])
        corpora[corp]['mantel'][fs].append(row['mantel_r'])

    # Compute mean |r| per corpus × feature_set
    plot_data_dbc = {}
    plot_data_man = {}
    for corp_name, corp_data in corpora.items():
        plot_data_dbc[corp_name] = []
        plot_data_man[corp_name] = []
        for fs in all_fs_keys:
            dbc_vals = corp_data['dbc'][fs]
            man_vals = corp_data['mantel'][fs]
            plot_data_dbc[corp_name].append(
                np.mean(np.abs(dbc_vals)) if dbc_vals else 0)
            plot_data_man[corp_name].append(
                np.mean(np.abs(man_vals)) if man_vals else 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    corpus_names = list(corpora.keys())
    colors = ['#2166ac', '#b2182b', '#4daf4a']  # blue, red, green
    x = np.arange(len(all_fs_keys))
    bar_width = 0.25

    for ax_idx, (test_name, plot_data, ylabel) in enumerate([
        ('DBC Anchor', plot_data_dbc, 'Mean |Spearman r|'),
        ('Mantel Test', plot_data_man, 'Mean |Pearson r|'),
    ]):
        ax = axes[ax_idx]
        for i, corp_name in enumerate(corpus_names):
            offset = (i - 1) * bar_width
            bars = ax.bar(x + offset, plot_data[corp_name], bar_width,
                          label=corp_name, color=colors[i],
                          edgecolor='white', linewidth=0.5)

            # Annotate with n=
            for j, bar in enumerate(bars):
                n_sig = 0
                n_total = 0
                # Count significance for this cell from raw data
                fs = all_fs_keys[j]
                corp_data = corpora[corp_name]
                if test_name == 'DBC Anchor':
                    vals = corp_data['dbc'].get(fs, [])
                else:
                    vals = corp_data['mantel'].get(fs, [])
                n_total = len(vals)
                if n_total > 0:
                    height_val = abs(plot_data[corp_name][j])
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            height_val + 0.02,
                            f'n={n_total}',
                            ha='center', va='bottom', fontsize=6,
                            color='#555555')

        ax.set_xticks(x)
        ax.set_xticklabels(all_fs_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{test_name}: Directional Effect Size', fontsize=11,
                     fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        # Annotation box
        ax.text(0.02, 0.97,
                'Positive control should be HIGH\n'
                'Negative control should be LOW\n'
                'Caesar should match positive',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Calibration: Cross-Author Comparison of Chronological Signal',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  ✓ Plot saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  CALIBRATION ANALYSIS")
    print("  Cross-Author Validation of Chronometric Method")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)

    # ── Load function-word list ───────────────────────────────────
    func_words = load_caesar_function_words()
    print(f"  Function words loaded: {len(func_words)}")
    print()

    # ── Load Caesar baseline ──────────────────────────────────────
    caesar_results = load_caesar_baseline()

    all_results = []

    # ════════════════════════════════════════════════════════════════
    # POSITIVE CONTROL: Cicero Ad Atticum
    # ════════════════════════════════════════════════════════════════
    cicero_csv = os.path.join(
        CORPUS_DIR,
        'calib_cicero_atticum_yearly_merged_normalized_lemmatized.csv')
    if not os.path.exists(cicero_csv):
        cicero_csv = os.path.join(
            CORPUS_DIR,
            'calib_cicero_atticum_yearly_merged_normalized.csv')

    if os.path.exists(cicero_csv):
        cicero_results = analyze_corpus(
            'Cicero Ad Atticum (yearly)',
            cicero_csv,
            gap_type='year',
            perm_method='mc',
            func_words=func_words,
            rng=rng,
        )
        all_results.extend(cicero_results)
    else:
        print(f"\n  ✗ Cicero CSV not found: {cicero_csv}")
        print(f"    Run scripts/20c_normalize_lemmatize_calib.py first.")

    # ════════════════════════════════════════════════════════════════
    # NEGATIVE CONTROL: DBC Pseudo-Books
    # ════════════════════════════════════════════════════════════════
    dbc_csv = os.path.join(
        CORPUS_DIR,
        'calib_dbc_pseudo_books_normalized_lemmatized.csv')
    if not os.path.exists(dbc_csv):
        dbc_csv = os.path.join(
            CORPUS_DIR,
            'calib_dbc_pseudo_books_normalized.csv')

    if os.path.exists(dbc_csv):
        dbc_results = analyze_corpus(
            'DBC Pseudo-Books',
            dbc_csv,
            gap_type='narrative',
            perm_method='exact',
            func_words=func_words,
            rng=rng,
        )
        all_results.extend(dbc_results)
    else:
        print(f"\n  ✗ DBC pseudo-books CSV not found: {dbc_csv}")

    # ════════════════════════════════════════════════════════════════
    # ADD CAESAR BASELINE
    # ════════════════════════════════════════════════════════════════
    for (fs, dist), vals in caesar_results.items():
        all_results.append({
            'corpus':        'Caesar DBG',
            'feature_set':   fs,
            'feat_label':    fs,
            'distance':      dist,
            'dbc_r':         vals['dbc_r'],
            'dbc_p':         vals['dbc_p'],
            'dbc_ci_low':    0,
            'dbc_ci_high':   0,
            'mantel_r':      vals['mantel_r'],
            'mantel_p':      vals['mantel_p'],
            'mantel_ci_low': 0,
            'mantel_ci_high': 0,
        })

    # ════════════════════════════════════════════════════════════════
    # SAVE CSV
    # ════════════════════════════════════════════════════════════════
    csv_out = os.path.join(OUTPUTS_DIR, 'calibration_results.csv')
    fieldnames = ['corpus', 'feature_set', 'feat_label', 'distance',
                  'dbc_r', 'dbc_p', 'dbc_ci_low', 'dbc_ci_high',
                  'mantel_r', 'mantel_p', 'mantel_ci_low', 'mantel_ci_high']
    with open(csv_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\n  ✓ CSV saved: {csv_out} ({len(all_results)} rows)")

    # ════════════════════════════════════════════════════════════════
    # PLOT
    # ════════════════════════════════════════════════════════════════
    plot_path = os.path.join(FIGURES_DIR, 'calibration_comparison.png')
    make_comparison_plot(all_results, plot_path)

    # ════════════════════════════════════════════════════════════════
    # INTERPRETATION
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)

    cicero_rows = [r for r in all_results
                   if 'Cicero' in r['corpus']]
    dbc_pseudo_rows = [r for r in all_results
                       if 'DBC Pseudo' in r['corpus']]
    caesar_rows = [r for r in all_results
                   if r['corpus'] == 'Caesar DBG']

    def summarize(rows, label):
        if not rows:
            print(f"\n  {label}: NO DATA")
            return

        dbc_rs = [r['dbc_r'] for r in rows if r['distance'] == 'Delta']
        dbc_ps = [r['dbc_p'] for r in rows if r['distance'] == 'Delta']
        man_rs = [r['mantel_r'] for r in rows if r['distance'] == 'Delta']
        man_ps = [r['mantel_p'] for r in rows if r['distance'] == 'Delta']


        dbc_dir_correct = sum(1 for rv in dbc_rs if rv < 0)
        dbc_sig = sum(1 for pv in dbc_ps if pv < 0.05)
        man_dir_correct = sum(1 for rv in man_rs if rv > 0)
        man_sig = sum(1 for pv in man_ps if pv < 0.05)

        print(f"\n  {label}:")
        print(f"    DBC Anchor (Delta):")
        print(f"      Mean r: {np.mean(dbc_rs):+.3f}  "
              f"Median r: {np.median(dbc_rs):+.3f}  "
              f"Range: [{np.min(dbc_rs):+.3f}, {np.max(dbc_rs):+.3f}]")
        print(f"      Direction correct (negative): {dbc_dir_correct}/{len(dbc_rs)}")
        print(f"      Significant (p < 0.05): {dbc_sig}/{len(dbc_ps)}")
        print(f"    Mantel (Delta):")
        print(f"      Mean r: {np.mean(man_rs):+.3f}  "
              f"Median r: {np.median(man_rs):+.3f}  "
              f"Range: [{np.min(man_rs):+.3f}, {np.max(man_rs):+.3f}]")
        print(f"      Direction correct (positive): {man_dir_correct}/{len(man_rs)}")
        print(f"      Significant (p < 0.05): {man_sig}/{len(man_ps)}")

    summarize(cicero_rows,      "CICERO (POSITIVE — should show drift)")
    summarize(dbc_pseudo_rows,  "DBC PSEUDO (NEGATIVE — should show null)")
    summarize(caesar_rows,      "CAESAR DBG (EXPERIMENTAL — being tested)")

    # ── Verdict ────────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  CALIBRATION VERDICT")
    print(f"  {'─' * 60}")


    cicero_dbc_mean = np.mean([r['dbc_r'] for r in cicero_rows
                                if r['distance'] == 'Delta']) if cicero_rows else 0
    pseudo_dbc_mean = np.mean([r['dbc_r'] for r in dbc_pseudo_rows
                                if r['distance'] == 'Delta']) if dbc_pseudo_rows else 0
    caesar_dbc_mean = np.mean([r['dbc_r'] for r in caesar_rows
                                if r['distance'] == 'Delta']) if caesar_rows else 0

    # Check (a) positive control
    if cicero_rows and cicero_dbc_mean < -0.3:
        print(f"\n  (a) Cicero (positive control):")
        print(f"      ✓ DBC Anchor mean r = {cicero_dbc_mean:+.3f} — "
              f"METHOD DETECTS KNOWN DRIFT")
    elif cicero_rows:
        print(f"\n  (a) Cicero (positive control):")
        print(f"      ✗ DBC Anchor mean r = {cicero_dbc_mean:+.3f} — "
              f"METHOD FAILS to detect known drift")
        print(f"      ⚠ This is a SERIOUS concern. The method may lack "
              f"sensitivity for chronological signal.")
        print(f"      Possible explanation: Cicero's familiar-letter register "
              f"is too variable within-year to show clear cross-year drift.")
    else:
        print(f"\n  (a) Cicero: NO DATA — run the corpus builder first.")

    # Check (b) negative control
    if dbc_pseudo_rows and abs(pseudo_dbc_mean) < 0.3:
        print(f"\n  (b) DBC pseudo-books (negative control):")
        print(f"      ✓ DBC Anchor mean r = {pseudo_dbc_mean:+.3f} — "
              f"NULL RESULT as predicted")
        print(f"      Method does NOT produce false positives on arbitrary "
              f"segments of a single concentrated work.")
    elif dbc_pseudo_rows:
        print(f"\n  (b) DBC pseudo-books (negative control):")
        print(f"      ✗ DBC Anchor mean r = {pseudo_dbc_mean:+.3f} — "
              f"UNEXPECTED SIGNAL")
        print(f"      ⚠ This means narrative progression within a single work "
              f"can produce a chronometric-like signal. The Caesar DBG result "
              f"must be interpreted with this confound acknowledged.")
    else:
        print(f"\n  (b) DBC pseudo-books: NO DATA.")

    # Check (c) Caesar regime
    if caesar_rows:
        print(f"\n  (c) Caesar DBG position:")
        if abs(caesar_dbc_mean) > 0.5 and cicero_rows and abs(cicero_dbc_mean) > 0.3:
            print(f"      Caesar DBC r = {caesar_dbc_mean:+.3f} falls in the "
                  f"'annual-like' regime.")
            print(f"      Effect size resembles the positive control more than "
                  f"the negative control.")
        elif abs(caesar_dbc_mean) > 0.5:
            print(f"      Caesar DBC r = {caesar_dbc_mean:+.3f} — strong signal "
                  f"but calibration corpora unavailable for comparison.")
        else:
            print(f"      Caesar DBC r = {caesar_dbc_mean:+.3f} — weak signal.")

    print(f"\n  CAVEATS (from docs/calibration_design.md):")
    print(f"    - Cicero letters differ in GENRE (private correspondence vs.")
    print(f"      Caesar's public commentarius). Register confound is real.")
    print(f"    - Cicero merged yearly bins span 20 years vs. Caesar's 6 years.")
    print(f"      Different time scales may affect signal strength.")
    print(f"    - DBC pseudo-books share the same author and genre as DBG.")
    print(f"      This is the tightest possible negative control but only tests")
    print(f"      one type of false positive (narrative-structure confound).")
    print(f"    - n = 10 (Cicero) uses Monte Carlo permutation (100,000).")
    print(f"      n = 7 (DBC pseudo, Caesar) uses exact permutation (5,040).")
    print()

    print("=" * 70)
    print("  CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
