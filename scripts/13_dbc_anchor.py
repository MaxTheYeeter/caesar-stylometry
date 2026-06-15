#!/usr/bin/env python3
"""
scripts/13_dbc_anchor.py

Tests the discriminating prediction between annual vs. bulk composition
hypotheses using the De Bello Civili (DBC) stylistic anchor.

Hypothesis A (annual composition):
    Distance-to-DBC DECREASES from Book I (furthest) to Book VII (closest).
    Later DBG books are stylistically more similar to late-Caesar DBC.

Hypothesis B (bulk composition):
    Distance-to-DBC is roughly EQUAL for all DBG books I-VII.
    No ordering trend exists.

Quantifies via Spearman rank correlation between book order (1..7) and
distance-to-DBC, with bootstrap 95% CI and permutation p-value.
Book VIII (Hirtius) is excluded from trends but shown for reference.

Uses pre-computed Delta matrices (Python Burrows + R stylo) and computes
cosine distance from feature matrices as a complementary metric.
"""

import csv
import sys
import os
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

csv.field_size_limit(sys.maxsize)


# ── Label mapping across matrix formats ─────────────────────────────────
def standardise_label(raw_label):
    """
    Map matrix labels to standard form: '1'..'7', 'VIII', 'DBC'.
    Also returns (is_caesar_dbg, is_hirtius, is_dbc, book_number).
    """
    label = raw_label.strip().strip('"')
    # Python Delta: Caesar_1..Caesar_7, Hirtius_VIII, DBC
    if label.startswith('Caesar_') and 'DBG' not in label:
        parts = label.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            n = int(parts[1])
            if n <= 7:
                return str(n), True, False, False, n
    if label == 'Hirtius_VIII':
        return 'VIII', False, True, False, 8
    if label == 'DBC':
        return 'DBC', False, False, True, None
    # R Delta: Caesar_DBC, Caesar_DBG-01..Caesar_DBG-07, Hirtius_DBG-08
    if label.startswith('Caesar_DBG-'):
        n = int(label.split('-')[-1])
        if n <= 7:
            return str(n), True, False, False, n
    if label == 'Hirtius_DBG-08':
        return 'VIII', False, True, False, 8
    if label == 'Caesar_DBC':
        return 'DBC', False, False, True, None
    # Feature matrix: dbg_book01..dbg_book08, dbc_complete
    if label.startswith('dbg_book'):
        n = int(label.replace('dbg_book', ''))
        if n <= 7:
            return str(n), True, False, False, n
        if n == 8:
            return 'VIII', False, True, False, 8
    if label == 'dbc_complete':
        return 'DBC', False, False, True, None
    return None, False, False, False, None


# ── Loading helpers ─────────────────────────────────────────────────────
def load_python_delta_matrix(path):
    """Load Burrows Delta matrix produced by earlier Python scripts."""
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_labels = [standardise_label(h)[0] for h in headers]
        for row in reader:
            row_label = standardise_label(row[0])[0]
            values = [float(v) for v in row[1:]]
            rows.append((row_label, values))
    # Build dict[row][col] = distance
    matrix = {}
    for row_label, values in rows:
        matrix[row_label] = {}
        for i in range(len(values)):
            col_lbl = col_labels[i + 1] if i + 1 < len(col_labels) else None
            if col_lbl is not None:
                matrix[row_label][col_lbl] = values[i]
    return matrix


def load_r_delta_matrix(path):
    """
    Load stylo Delta matrix (R format: quoted labels, first cell empty).
    Returns dict[row][col] = distance.
    """
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        col_labels = [standardise_label(h)[0] for h in headers]
        for row in reader:
            row_label = standardise_label(row[0])[0]
            values = [float(v) for v in row[1:]]
            rows.append((row_label, values))
    matrix = {}
    for row_label, values in rows:
        matrix[row_label] = {}
        for i in range(len(values)):
            col_lbl = col_labels[i + 1] if i + 1 < len(col_labels) else None
            if col_lbl is not None:
                matrix[row_label][col_lbl] = values[i]
    return matrix


def load_feature_matrix(path):
    """
    Load a feature matrix CSV.
    Returns:
        labels: np.array of str (book numbers and 'VIII', 'DBC')
        book_nums: np.array of int (1-8) or -1 for DBC
        feature_names: list of str
        feature_data: np.array (n_samples × n_features) — proportions
    """
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Identify metadata columns
        meta_cols = ['segment_id', 'author_group', 'work', 'book',
                     'total_tokens', 'total_ngrams']
        feature_start = 0
        for i, col in enumerate(header):
            if col not in meta_cols:
                feature_start = i
                break
        feature_names = header[feature_start:]
        labels = []
        book_nums = []
        for row in reader:
            seg_id = row[0]
            std_label, is_caesar, is_hirtius, is_dbc, book_n = \
                standardise_label(seg_id)
            if std_label is None:
                continue
            if is_caesar:
                bn = int(std_label)
            elif is_hirtius:
                bn = 8
            else:
                bn = -1
            labels.append(std_label)
            book_nums.append(bn)
            features = np.array([float(v) for v in row[feature_start:]],
                                dtype=np.float64)
            rows.append(features)
    return (np.array(labels),
            np.array(book_nums),
            feature_names,
            np.array(rows))


def extract_dbc_distances(matrix, dbg_labels):
    """
    From a pairwise distance dict, extract distance-to-DBC for each DBG book.
    Returns dict: book_label -> distance_to_DBC
    """
    result = {}
    for label in dbg_labels:
        if label in matrix and 'DBC' in matrix[label]:
            result[label] = matrix[label]['DBC']
        elif 'DBC' in matrix and label in matrix['DBC']:
            result[label] = matrix['DBC'][label]
    return result


def compute_cosine_distances(feature_data, labels):
    """
    Compute pairwise cosine distances between all samples.
    Returns dict[row_label][col_label] = distance.
    """
    n = len(labels)
    matrix = {labels[i]: {} for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[labels[i]][labels[j]] = 0.0
            else:
                matrix[labels[i]][labels[j]] = cosine_distance(
                    feature_data[i], feature_data[j])
    return matrix


# ── Statistics ──────────────────────────────────────────────────────────
def spearman_bootstrap_ci(x, y, n_bootstrap=10000, ci=95, rng=None):
    """
    Bootstrap confidence interval for Spearman rank correlation.
    Returns (r, p_value, ci_low, ci_high).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r_obs, p_obs = stats.spearmanr(x, y)
    n = len(x)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r_boot, _ = stats.spearmanr(x[idx], y[idx])
        boot_rs.append(r_boot)
    boot_rs = np.array(boot_rs)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_rs, alpha)
    ci_high = np.percentile(boot_rs, 100 - alpha)
    return r_obs, p_obs, ci_low, ci_high


def spearman_permutation_p(x, y, rng=None):
    """
    Exact permutation test for Spearman r (null: no association).
    With n=7, enumerates all 7! = 5040 permutations.
    Returns (r_obs, p_value, n_permutations).
    """
    from itertools import permutations
    if rng is None:
        rng = np.random.RandomState(42)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r_obs, _ = stats.spearmanr(x, y)
    n = len(x)
    # Exact enumeration for n <= 7
    count_extreme = 0
    total = 0
    for perm in permutations(range(n)):
        y_perm = y[list(perm)]
        r_perm, _ = stats.spearmanr(x, y_perm)
        if abs(r_perm) >= abs(r_obs):
            count_extreme += 1
        total += 1
    p_val = count_extreme / total
    return r_obs, p_val, total


# ── Plotting ────────────────────────────────────────────────────────────
def make_dbc_distance_plot(distances_dict, feature_label, out_path):
    """
    Create a single-figure plot with panels for each distance type.
    distances_dict: {'Delta': {book_label: dist, ...}, 'Cosine': {...}, ...}
    """
    n_panels = len(distances_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5),
                              squeeze=False)
    axes = axes[0]

    for ax, (dist_type, dist_data) in zip(axes, distances_dict.items()):
        # Separate DBG I-VII and VIII
        book_nums = []
        book_dists = []
        for label, dist in dist_data.items():
            if label.isdigit() and 1 <= int(label) <= 7:
                book_nums.append(int(label))
                book_dists.append(dist)
        # Sort by book number
        sorted_pairs = sorted(zip(book_nums, book_dists))
        book_nums = np.array([p[0] for p in sorted_pairs])
        book_dists = np.array([p[1] for p in sorted_pairs])

        # Hirtius (Book VIII)
        hirtius_dist = dist_data.get('VIII', None)

        # Compute Spearman r
        if len(book_nums) >= 3:
            r_obs, p_obs, ci_low, ci_high = spearman_bootstrap_ci(
                book_nums, book_dists)
            _, p_perm, n_perm = spearman_permutation_p(book_nums, book_dists)
        else:
            r_obs, p_obs = np.nan, np.nan
            ci_low, ci_high = np.nan, np.nan
            p_perm, n_perm = np.nan, 0

        # Plot DBG I-VII
        ax.scatter(book_nums, book_dists, c='#2166ac', s=80, zorder=5,
                   label='DBG I-VII (Caesar)')

        # Annotate each point with book number
        for bn, bd in zip(book_nums, book_dists):
            ax.annotate(str(bn), (bn, bd),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=9, ha='center', color='#2166ac')

        # Fit and plot trend line
        if len(book_nums) >= 2:
            slope, intercept, r_value, p_value, std_err = \
                stats.linregress(book_nums, book_dists)
            x_line = np.linspace(0.5, 7.5, 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, '--', color='#d6604d', linewidth=1.5,
                    alpha=0.7, label='OLS trend')

        # Plot Hirtius (Book VIII) as reference
        if hirtius_dist is not None:
            ax.scatter([8], [hirtius_dist], c='#b2182b', s=80, zorder=5,
                       marker='s', label='Book VIII (Hirtius)')
            ax.annotate('VIII', (8, hirtius_dist),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=9, ha='center', color='#b2182b')

        # Aesthetics
        ax.set_xlabel('Book Number')
        ax.set_ylabel(f'{dist_type} Distance to DBC')
        ax.set_title(f'{feature_label}\n{dist_type} Distance to DBC',
                      fontsize=10, fontweight='bold')
        ax.set_xlim(0.3, 8.7)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_xticklabels(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'])
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Annotation box
        if not np.isnan(r_obs):
            ann_text = (
                f"Spearman r = {r_obs:+.3f}\n"
                f"Bootstrap 95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]\n"
                f"Permutation p = {p_perm:.4f} (exact, N={int(n_perm)})"
            )
        else:
            ann_text = "Insufficient data"
        ax.text(0.97, 0.97, ann_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                          alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {out_path}")


# ── Feature set registry ─────────────────────────────────────────────────
FEATURE_SETS = OrderedDict([
    # function words
    ('function_words_tokens', {
        'label': 'Function Words (Tokens)',
        'py_delta': 'delta_python_distance_function_words_tokens_books.csv',
        'r_delta': None,
        'feature_csv': 'features_function_words_tokens_books.csv',
        'mfw': None,
    }),
    ('function_words_lemmas', {
        'label': 'Function Words (Lemmas)',
        'py_delta': 'delta_python_distance_function_words_lemmas_books.csv',
        'r_delta': None,
        'feature_csv': 'features_function_words_lemmas_books.csv',
        'mfw': None,
    }),
    # MFW 100
    ('mfw100_tokens', {
        'label': 'MFW 100 (Tokens)',
        'py_delta': 'delta_python_distance_mfw100_tokens_books.csv',
        'r_delta': 'delta_distance_tokens_mfw100.csv',
        'feature_csv': 'features_mfw100_tokens_books.csv',
        'mfw': 100,
    }),
    ('mfw100_lemmas', {
        'label': 'MFW 100 (Lemmas)',
        'py_delta': 'delta_python_distance_mfw100_lemmas_books.csv',
        'r_delta': 'delta_distance_lemmas_mfw100.csv',
        'feature_csv': 'features_mfw100_lemmas_books.csv',
        'mfw': 100,
    }),
    # MFW 200
    ('mfw200_tokens', {
        'label': 'MFW 200 (Tokens)',
        'py_delta': None,
        'r_delta': 'delta_distance_tokens_mfw200.csv',
        'feature_csv': 'features_mfw200_tokens_books.csv',
        'mfw': 200,
    }),
    ('mfw200_lemmas', {
        'label': 'MFW 200 (Lemmas)',
        'py_delta': None,
        'r_delta': 'delta_distance_lemmas_mfw200.csv',
        'feature_csv': 'features_mfw200_lemmas_books.csv',
        'mfw': 200,
    }),
    # MFW 300
    ('mfw300_tokens', {
        'label': 'MFW 300 (Tokens)',
        'py_delta': None,
        'r_delta': 'delta_distance_tokens_mfw300.csv',
        'feature_csv': 'features_mfw300_tokens_books.csv',
        'mfw': 300,
    }),
    ('mfw300_lemmas', {
        'label': 'MFW 300 (Lemmas)',
        'py_delta': None,
        'r_delta': 'delta_distance_lemmas_mfw300.csv',
        'feature_csv': 'features_mfw300_lemmas_books.csv',
        'mfw': 300,
    }),
    # Character n-grams
    ('char2gram', {
        'label': 'Char 2-grams',
        'py_delta': None,
        'r_delta': 'delta_distance_tokens_c2gram_mfw500.csv',
        'feature_csv': 'features_char2gram_books.csv',
        'mfw': None,
    }),
    ('char3gram', {
        'label': 'Char 3-grams',
        'py_delta': None,
        'r_delta': 'delta_distance_tokens_c3gram_mfw500.csv',
        'feature_csv': 'features_char3gram_books.csv',
        'mfw': None,
    }),
    ('char4gram', {
        'label': 'Char 4-grams',
        'py_delta': None,
        'r_delta': 'delta_distance_tokens_c4gram_mfw1000.csv',
        'feature_csv': 'features_char4gram_books.csv',
        'mfw': None,
    }),
])


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  DBC ANCHOR TEST — Distance to De Bello Civili")
    print("=" * 65)
    print()

    rng = np.random.RandomState(42)

    # Collect all results for summary
    all_results = []

    for feat_key, cfg in FEATURE_SETS.items():
        print(f"\n{'─' * 50}")
        print(f"  FEATURE SET: {cfg['label']}")
        print(f"{'─' * 50}")

        distances = OrderedDict()  # dist_type -> {label: value}

        # ── Try loading Python Delta ──
        if cfg['py_delta']:
            py_path = os.path.join(OUTPUTS_DIR, cfg['py_delta'])
            if os.path.exists(py_path):
                print(f"  Loading Python Delta: {cfg['py_delta']}")
                py_matrix = load_python_delta_matrix(py_path)
                dbg_labels = [str(i) for i in range(1, 9)]
                py_dists = extract_dbc_distances(py_matrix, dbg_labels)
                if py_dists:
                    distances['Delta (Python)'] = py_dists
                    print(f"    Caesar I-VII distances: "
                          f"{[round(py_dists.get(str(i), np.nan), 4) for i in range(1, 8)]}")
                    print(f"    Hirtius VIII: {round(py_dists.get('VIII', np.nan), 4)}")
            else:
                print(f"  Python Delta file not found: {py_path}")

        # ── Try loading R Delta ──
        if cfg['r_delta']:
            r_path = os.path.join(OUTPUTS_DIR, cfg['r_delta'])
            if os.path.exists(r_path):
                print(f"  Loading R Delta: {cfg['r_delta']}")
                r_matrix = load_r_delta_matrix(r_path)
                dbg_labels_r = [str(i) for i in range(1, 9)]
                r_dists = extract_dbc_distances(r_matrix, dbg_labels_r)
                if r_dists:
                    key = 'Delta (stylo)' if 'Delta (Python)' in distances \
                          else 'Delta'
                    distances[key] = r_dists
                    print(f"    Caesar I-VII distances: "
                          f"{[round(r_dists.get(str(i), np.nan), 4) for i in range(1, 8)]}")
                    print(f"    Hirtius VIII: {round(r_dists.get('VIII', np.nan), 4)}")
            else:
                print(f"  R Delta file not found: {r_path}")

        # ── Compute Cosine from feature matrix ──
        feat_path = os.path.join(OUTPUTS_DIR, cfg['feature_csv'])
        if os.path.exists(feat_path):
            print(f"  Computing Cosine from: {cfg['feature_csv']}")
            flabels, fbook_nums, fnames, fdata = load_feature_matrix(feat_path)
            if cfg['mfw']:
                fdata_limited = fdata[:, :cfg['mfw']]
            else:
                fdata_limited = fdata
            cos_matrix = compute_cosine_distances(fdata_limited, flabels)
            dbg_labels_cos = [str(i) for i in range(1, 9)]
            cos_dists = extract_dbc_distances(cos_matrix, dbg_labels_cos)
            if cos_dists:
                distances['Cosine'] = cos_dists
                print(f"    Caesar I-VII distances: "
                      f"{[round(cos_dists.get(str(i), np.nan), 4) for i in range(1, 8)]}")
                print(f"    Hirtius VIII: {round(cos_dists.get('VIII', np.nan), 4)}")
        else:
            print(f"  Feature file not found: {feat_path}")

        if not distances:
            print(f"  ⚠ No distances available for {cfg['label']}")
            continue

        # ── Compute statistics for each distance type ──
        print(f"\n  Statistics:")
        for dist_type, dist_data in distances.items():
            # Extract DBG I-VII
            dbg_pairs = [(int(lbl), dist_data[lbl])
                         for lbl in dist_data if lbl.isdigit()
                         and 1 <= int(lbl) <= 7]
            dbg_pairs.sort()
            if len(dbg_pairs) < 3:
                print(f"    {dist_type}: insufficient data")
                continue
            books = np.array([p[0] for p in dbg_pairs])
            dists = np.array([p[1] for p in dbg_pairs])

            r_obs, p_obs, ci_low, ci_high = spearman_bootstrap_ci(
                books, dists, rng=rng)
            _, p_perm, n_perm = spearman_permutation_p(
                books, dists, rng=rng)

            # Distance trend: compare early (I-III) vs late (V-VII)
            early_mean = np.mean([d for b, d in dbg_pairs if b <= 3])
            late_mean = np.mean([d for b, d in dbg_pairs if b >= 5])
            hirtius = dist_data.get('VIII', None)

            # Direction interpretation:
            # Negative r = distance falls as book number rises = annual signal
            # Positive r = distance rises with book number = contrary to annual
            if r_obs < -0.3:
                direction = 'A (annual)'
            elif abs(r_obs) <= 0.3:
                direction = 'B (bulk)'
            else:
                direction = '? (pos r)'

            print(f"    {dist_type}:")
            print(f"      Spearman r = {r_obs:+.4f}  "
                  f"(bootstrap 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], "
                  f"perm. p = {p_perm:.4f})")
            print(f"      DBG I-III mean = {early_mean:.4f}  |  "
                  f"DBG V-VII mean = {late_mean:.4f}")
            if hirtius is not None:
                print(f"      Hirtius VIII = {hirtius:.4f}")

            # Store for summary
            all_results.append({
                'feature_set': cfg['label'],
                'dist_type': dist_type,
                'spearman_r': r_obs,
                'p_perm': p_perm,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'early_mean': early_mean,
                'late_mean': late_mean,
                'hirtius': hirtius,
                'direction': direction,
            })

        # ── Plot ──
        safe_key = feat_key.replace('/', '_')
        out_path = os.path.join(FIGURES_DIR,
                                f'dbc_anchor_{safe_key}.png')
        make_dbc_distance_plot(distances, cfg['label'], out_path)

    # ════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("  SUMMARY: Distance-to-DBC Analysis")
    print("=" * 95)
    print(f"{'Feature Set':<30s} {'Metric':<20s} "
          f"{'r':>7s} {'p_perm':>8s} {'CI low':>8s} {'CI high':>8s} "
          f"{'Early':>8s} {'Late':>8s} {'Hirtius':>8s} {'Dir':>10s}")
    print("-" * 95)
    for r in all_results:
        h_str = f"{r['hirtius']:.4f}" if r['hirtius'] is not None else "N/A"
        print(f"{r['feature_set']:<30s} {r['dist_type']:<20s} "
              f"{r['spearman_r']:+7.3f} {r['p_perm']:8.4f} "
              f"{r['ci_low']:+8.4f} {r['ci_high']:+8.4f} "
              f"{r['early_mean']:8.4f} {r['late_mean']:8.4f} "
              f"{h_str:>8s} "
              f"{r['direction']:>10s}")

    # ════════════════════════════════════════════════════════════════
    # CONSENSUS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 95)
    print("  CONSENSUS")
    print("=" * 95)

    n_annual = sum(1 for r in all_results if r['direction'] == 'A (annual)')
    n_bulk   = sum(1 for r in all_results if r['direction'] == 'B (bulk)')
    n_other  = len(all_results) - n_annual - n_bulk
    print(f"  Feature set × metric combinations: {len(all_results)}")
    print(f"    Consistent with ANNUAL (r < -0.3):  {n_annual}")
    print(f"    Consistent with BULK (|r| < 0.3):   {n_bulk}")
    print(f"    Other (positive r or unclear):       {n_other}")

    # List significant results
    sig_results = [r for r in all_results if r['p_perm'] < 0.05]
    if sig_results:
        print(f"\n  Statistically significant (p < 0.05): {len(sig_results)}")
        for r in sig_results:
            print(f"    {r['feature_set']} — {r['dist_type']}: "
                  f"r = {r['spearman_r']:+.3f}, p = {r['p_perm']:.4f}")
    else:
        print(f"\n  No results reach p < 0.05 significance.")

    # Directional consistency:
    # "Late books closer" means early_mean > late_mean
    # (distance for early books is larger than distance for late books)
    direction_checks = [r for r in all_results
                        if r['early_mean'] and r['late_mean']]
    n_late_closer = sum(1 for r in direction_checks
                        if r['early_mean'] > r['late_mean'])
    n_early_closer = sum(1 for r in direction_checks
                         if r['late_mean'] > r['early_mean'])
    n_equal = len(direction_checks) - n_late_closer - n_early_closer

    print(f"\n  Direction of DBC distance (all feature sets):")
    print(f"    Late DBG books (V-VII) closer to DBC: {n_late_closer}")
    print(f"    Early DBG books (I-III) closer to DBC: {n_early_closer}")
    if n_equal > 0:
        print(f"    Tied:                                   {n_equal}")

    if n_late_closer > n_early_closer:
        print(f"\n    → Late books are closer to DBC in "
              f"{n_late_closer}/{len(direction_checks)} comparisons")
        print(f"    → SUPPORTS Hypothesis A (annual composition)")
    elif n_early_closer > n_late_closer:
        print(f"\n    → Early books are closer to DBC in "
              f"{n_early_closer}/{len(direction_checks)} comparisons")
        print(f"    → CONTRADICTS Hypothesis A")
    else:
        print(f"\n    → No consistent directional pattern")

    print("\n  CAVEATS:")
    print("    - n = 7 DBG books (I-VII) — very small sample")
    print("    - Spearman r is sensitive to rank ordering, not linearity")
    print("    - Bootstrap CIs are wide with n=7")
    print("    - Exact permutation test uses all 7! = 5,040 permutations")
    print("    - This is a bivariate exploration; "
          "formal chronology test follows in script 14")
    print()
    print("  All plots saved to:", FIGURES_DIR)
    print("=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
