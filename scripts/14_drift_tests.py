#!/usr/bin/env python3
"""
scripts/14_drift_tests.py

FORMAL DIRECTIONAL DRIFT TESTS (small‑n safe)

Three complementary analyses, each with permutation‑based significance:
  1. MANTEL TEST — correlate book‑to‑book stylistic distance matrix
     D_style[i,j] with chronological‑gap matrix G[i,j] = |year_i − year_j|.
     If style drifts over time, books further apart in years should be
     stylistically more distant.

  2. DBC ANCHOR — Spearman correlation between book order (I→VII) and
     each book's distance to the later De Bello Civili anchor. Annual
     composition predicts later books are closer to DBC (negative r).

  3. PCA PC1 — Spearman correlation between book order and the first
     principal component coordinate (unsupervised ordination). Annual
     composition predicts books spread along PC1 in chronological order.

All p‑values are exact permutation tests (n=7 → 5,040 permutations).
Bootstrap 95% CIs are reported for Spearman r. No parametric assumptions.

References
----------
  - Mantel, N. (1967). "The detection of disease clustering…"
  - Burrows, J. (2002). "Delta: a measure of stylistic difference…"
"""

import csv
import os
import sys
import warnings
from collections import OrderedDict
from itertools import permutations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance_fn
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

csv.field_size_limit(sys.maxsize)

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# ═══════════════════════════════════════════════════════════════════════
# 1. BOOK YEARS (standard campaign years)
# ═══════════════════════════════════════════════════════════════════════
DBG_YEARS = {
    1: 58,   # Book I  — Helvetian campaign
    2: 57,   # Book II — Belgic campaign
    3: 56,   # Book III — Alpine / Veneti campaign
    4: 55,   # Book IV — German / first Britain
    5: 54,   # Book V  — Second Britain / revolts
    6: 53,   # Book VI — Treveri / German / ethnographic
    7: 52,   # Book VII — Vercingetorix / Alesia
    8: 51,   # Book VIII — Hirtius (control, excluded from Caesar trends)
}
# DBC (De Bello Civili) = 49–48 BC.  Use 49 as single ref year (narrative start).
DBC_YEAR = 49

# Build gap matrix for DBG I-VII
def build_year_gap_matrix(n_books=7):
    """Return n×n matrix where G[i,j] = |year_i - year_j| (in years)."""
    years = np.array([DBG_YEARS[i + 1] for i in range(n_books)], dtype=float)
    G = np.zeros((n_books, n_books))
    for i in range(n_books):
        for j in range(n_books):
            G[i, j] = abs(years[i] - years[j])
    return G, years

# For DBC anchor
def build_year_gap_to_dbc(n_books=7):
    """Return array where gap[i] = |year_i - DBC_YEAR|."""
    years = np.array([DBG_YEARS[i + 1] for i in range(n_books)], dtype=float)
    return abs(years - DBC_YEAR), years


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE SET REGISTRY
# ═══════════════════════════════════════════════════════════════════════
FEATURE_SETS = OrderedDict([
    ('function_words_tokens', {
        'label': 'Function Words (Tokens)',
        'csv': 'features_function_words_tokens_books.csv',
        'type': 'token',
    }),
    ('function_words_lemmas', {
        'label': 'Function Words (Lemmas)',
        'csv': 'features_function_words_lemmas_books.csv',
        'type': 'lemma',
    }),
    ('mfw100_tokens', {
        'label': 'MFW 100 (Tokens)',
        'csv': 'features_mfw100_tokens_books.csv',
        'type': 'token',
    }),
    ('mfw100_lemmas', {
        'label': 'MFW 100 (Lemmas)',
        'csv': 'features_mfw100_lemmas_books.csv',
        'type': 'lemma',
    }),
    ('mfw200_tokens', {
        'label': 'MFW 200 (Tokens)',
        'csv': 'features_mfw200_tokens_books.csv',
        'type': 'token',
    }),
    ('mfw200_lemmas', {
        'label': 'MFW 200 (Lemmas)',
        'csv': 'features_mfw200_lemmas_books.csv',
        'type': 'lemma',
    }),
    ('mfw300_tokens', {
        'label': 'MFW 300 (Tokens)',
        'csv': 'features_mfw300_tokens_books.csv',
        'type': 'token',
    }),
    ('mfw300_lemmas', {
        'label': 'MFW 300 (Lemmas)',
        'csv': 'features_mfw300_lemmas_books.csv',
        'type': 'lemma',
    }),
    ('char2gram', {
        'label': 'Char 2-grams',
        'csv': 'features_char2gram_books.csv',
        'type': 'token',
    }),
    ('char3gram', {
        'label': 'Char 3-grams',
        'csv': 'features_char3gram_books.csv',
        'type': 'token',
    }),
    ('char4gram', {
        'label': 'Char 4-grams',
        'csv': 'features_char4gram_books.csv',
        'type': 'token',
    }),
])


# ═══════════════════════════════════════════════════════════════════════
# 3. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_feature_matrix(path):
    """
    Load a feature matrix CSV (book-level).
    Returns:
        labels: list of str — book number strings '1'..'8' and 'DBC'
        all_data: np.ndarray (n_samples × n_features) — proportion features
        dbg_mask: bool array — True for DBG I-VII
        hirtius_idx: int or None — index of Book VIII
        dbc_idx: int or None — index of DBC
    """
    all_rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Identify metadata columns
        meta_cols = {'segment_id', 'author_group', 'work', 'book',
                     'total_tokens', 'total_ngrams'}
        feature_start = 0
        for i, col in enumerate(header):
            if col not in meta_cols:
                feature_start = i
                break
        feature_names = header[feature_start:]
        for row in reader:
            seg_id = row[0]
            features = np.array([float(v) for v in row[feature_start:]],
                                dtype=np.float64)
            all_rows.append((seg_id, features))

    # Map segment IDs to standard labels
    labels = []
    data_list = []
    dbg_mask = []
    hirtius_idx = None
    dbc_idx = None

    for idx, (seg_id, feats) in enumerate(all_rows):
        if seg_id.startswith('dbg_book'):
            book_num = int(seg_id.replace('dbg_book', ''))
            if book_num <= 7:
                labels.append(str(book_num))
                dbg_mask.append(True)
            elif book_num == 8:
                labels.append('VIII')
                dbg_mask.append(False)
                hirtius_idx = idx
            else:
                labels.append(seg_id)
                dbg_mask.append(False)
        elif seg_id == 'dbc_complete':
            labels.append('DBC')
            dbg_mask.append(False)
            dbc_idx = idx
        else:
            labels.append(seg_id)
            dbg_mask.append(False)
        data_list.append(feats)

    all_data = np.array(data_list)
    # Recompute indices after filtering
    dbg_mask_arr = np.array(dbg_mask)
    hirtius_final = None
    dbc_final = None
    for i, lbl in enumerate(labels):
        if lbl == 'VIII':
            hirtius_final = i
        elif lbl == 'DBC':
            dbc_final = i

    return (np.array(labels), all_data, dbg_mask_arr,
            hirtius_final, dbc_final, feature_names)


def extract_dbg_subset(labels, data, dbg_mask):
    """Extract DBG I-VII rows, ordered by book number."""
    dbg_indices = np.where(dbg_mask)[0]
    # Sort by book number
    pairs = [(i, int(labels[i])) for i in dbg_indices]
    pairs.sort(key=lambda x: x[1])
    ordered_indices = [p[0] for p in pairs]
    ordered_labels = [labels[i] for i in ordered_indices]
    ordered_data = data[ordered_indices]
    return ordered_labels, ordered_data, ordered_indices


# ═══════════════════════════════════════════════════════════════════════
# 4. DISTANCE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
def compute_delta_matrix(data):
    """
    Classic Burrows Delta: z-score features across samples, then
    mean absolute z-score difference between each pair of samples.
    Standardize across ALL samples provided (including VIII and DBC).
    Returns symmetric distance matrix.
    """
    X = data.copy()
    # Z-score across samples (columns = features)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=0)
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


def compute_cosine_distance_matrix(data):
    """Compute pairwise cosine distance (1 − cosine similarity)."""
    n = data.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance_fn(data[i], data[j])
            D[i, j] = d
            D[j, i] = d
    return D


def matrix_to_upper_triangle(D):
    """Flatten upper triangle (excluding diagonal) of a square matrix."""
    n = D.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(D[i, j])
    return np.array(vals)


# ═══════════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════
def mantel_exact(D_style, G_gap, rng=None):
    """
    Exact Mantel test for n ≤ 7.
    H₀: no correlation between stylistic distance and chronological gap.
    Reports Mantel r = Pearson r(upper(D_style), upper(G_gap)).
    Enumerates all n! permutations of row/column order.
    Returns: (r_obs, p_value, n_permutations)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    n = D_style.shape[0]
    style_vec = matrix_to_upper_triangle(D_style)
    gap_vec = matrix_to_upper_triangle(G_gap)

    r_obs, _ = stats.pearsonr(style_vec, gap_vec)

    # Exact enumeration
    count_extreme = 0
    total = 0
    for perm in permutations(range(n)):
        # Permute both rows and columns of style matrix
        perm_list = list(perm)
        D_perm = D_style[perm_list][:, perm_list]
        style_perm = matrix_to_upper_triangle(D_perm)
        r_perm, _ = stats.pearsonr(style_perm, gap_vec)
        if r_perm >= r_obs:  # one-sided: positive correlation expected
            count_extreme += 1
        total += 1

    p_val = count_extreme / total
    return r_obs, p_val, total


def mantel_bootstrap_ci(D_style, G_gap, n_bootstrap=10000, ci=95, rng=None):
    """
    Bootstrap CI for Mantel r by resampling the pairwise observations.
    (Reasonable with 21 pairs from n=7 books.)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    style_vec = matrix_to_upper_triangle(D_style)
    gap_vec = matrix_to_upper_triangle(G_gap)
    n_pairs = len(style_vec)

    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        r_boot, _ = stats.pearsonr(style_vec[idx], gap_vec[idx])
        boot_rs.append(r_boot)
    boot_rs = np.array(boot_rs)
    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_rs, alpha)
    ci_high = np.percentile(boot_rs, 100 - alpha)
    return ci_low, ci_high


def spearman_exact_permutation(x, y, rng=None):
    """
    Exact permutation test for Spearman r (one-sided: r < 0 for annual).
    Enumerates all n! permutations.  With n=7 → 5,040.
    Returns: (r_obs, p_value, n_permutations)
    """
    if rng is None:
        rng = np.random.RandomState(42)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r_obs, _ = stats.spearmanr(x, y)
    n = len(x)

    count_extreme = 0
    total = 0
    for perm in permutations(range(n)):
        y_perm = y[list(perm)]
        r_perm, _ = stats.spearmanr(x, y_perm)
        # For DBC anchor: negative r expected (annual hypothesis)
        # For PC1: could be positive or negative — use two-sided
        if r_perm <= r_obs:  # one-sided: more negative = more extreme
            count_extreme += 1
        total += 1

    p_val = count_extreme / total
    return r_obs, p_val, total


def spearman_bootstrap_ci(x, y, n_bootstrap=10000, ci=95, rng=None):
    """Bootstrap CI for Spearman r."""
    if rng is None:
        rng = np.random.RandomState(42)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    return ci_low, ci_high


# ═══════════════════════════════════════════════════════════════════════
# 6. PCA HELPER
# ═══════════════════════════════════════════════════════════════════════
def compute_pca_pc1(all_data, labels, dbg_mask):
    """
    Fit PCA on ALL samples (including VIII, DBC), extract PC1 for DBG I-VII.
    Returns: (dbg_book_nums, pc1_values, pc1_variance_explained)
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_data)
    pca = PCA(n_components=min(7, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    pc1_var = pca.explained_variance_ratio_[0]

    # Extract DBG I-VII
    dbg_indices = np.where(dbg_mask)[0]
    dbg_labels_ordered = sorted([(int(labels[i]), i) for i in dbg_indices])
    book_nums = np.array([p[0] for p in dbg_labels_ordered])
    pc1_vals = np.array([X_pca[p[1], 0] for p in dbg_labels_ordered])
    return book_nums, pc1_vals, pc1_var


# ═══════════════════════════════════════════════════════════════════════
# 7. PLOTTING
# ═══════════════════════════════════════════════════════════════════════
def make_drift_plot(feature_label, feat_key,
                    mantel_data, dbc_data, pca_data,
                    out_path):
    """
    Three-panel figure:
      Left: Mantel scatter — D_style vs |year gap|
      Middle: DBC Anchor — distance to DBC vs book order
      Right: PCA PC1 vs book order
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(feature_label, fontsize=11, fontweight='bold', y=1.01)

    # ── Panel 1: Mantel test ──
    ax = axes[0]
    if mantel_data is not None:
        gaps, dists, r_mantel, p_mantel, ci_lo, ci_hi = mantel_data
        ax.scatter(gaps, dists, c='#2166ac', s=30, alpha=0.7, zorder=5)
        # OLS line
        if len(gaps) > 1:
            slope, intercept, _, _, _ = stats.linregress(gaps, dists)
            x_line = np.linspace(min(gaps) - 0.5, max(gaps) + 0.5, 50)
            ax.plot(x_line, slope * x_line + intercept,
                    '--', color='#d6604d', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Chronological gap (years)')
        ax.set_ylabel('Stylistic distance')
        ann = (f"Mantel r = {r_mantel:+.3f}\n"
               f"p = {p_mantel:.4f} (exact)\n"
               f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=7,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_title('Mantel Test: Distance ~ Time Gap', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: DBC Anchor ──
    ax = axes[1]
    if dbc_data is not None:
        books, dists_dbc, r_dbc, p_dbc, ci_lo_dbc, ci_hi_dbc, \
            hirtius_dist = dbc_data
        ax.scatter(books, dists_dbc, c='#2166ac', s=60, zorder=5,
                   label='DBG I-VII')
        for b, d in zip(books, dists_dbc):
            ax.annotate(str(b), (b, d), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha='center',
                        color='#2166ac')
        if len(books) > 1:
            slope, intercept, _, _, _ = stats.linregress(books, dists_dbc)
            x_line = np.linspace(0.5, 7.5, 50)
            ax.plot(x_line, slope * x_line + intercept,
                    '--', color='#d6604d', linewidth=1.5, alpha=0.7)
        if hirtius_dist is not None:
            ax.scatter([8], [hirtius_dist], c='#b2182b', s=60, zorder=5,
                       marker='s', label='VIII (Hirtius)')
            ax.annotate('VIII', (8, hirtius_dist), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha='center', color='#b2182b')
        ax.set_xlabel('Book Number')
        ax.set_ylabel('Distance to DBC')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
        ax.set_xticklabels(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'])
        ax.set_xlim(0.3, 8.7)
        ax.legend(fontsize=7, loc='best')
        ann = (f"Spearman r = {r_dbc:+.3f}\n"
               f"p = {p_dbc:.4f} (exact)\n"
               f"95% CI [{ci_lo_dbc:+.3f}, {ci_hi_dbc:+.3f}]")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=7,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_title('DBC Anchor: Distance to DBC', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: PCA PC1 ──
    ax = axes[2]
    if pca_data is not None:
        books_pca, pc1, var_pct, r_pca, p_pca, ci_lo_pca, ci_hi_pca = pca_data
        ax.scatter(books_pca, pc1, c='#2166ac', s=60, zorder=5)
        for b, v in zip(books_pca, pc1):
            ax.annotate(str(b), (b, v), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha='center',
                        color='#2166ac')
        if len(books_pca) > 1:
            slope, intercept, _, _, _ = stats.linregress(books_pca, pc1)
            x_line = np.linspace(0.5, 7.5, 50)
            ax.plot(x_line, slope * x_line + intercept,
                    '--', color='#d6604d', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Book Number')
        ax.set_ylabel('PC1 coordinate')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'])
        ax.set_xlim(0.3, 7.7)
        ann = (f"PC1 = {var_pct:.1%} variance\n"
               f"Spearman |r| = {abs(r_pca):.3f}\n"
               f"p = {p_pca:.4f} (exact, 2-sided)\n"
               f"95% CI [{ci_lo_pca:+.3f}, {ci_hi_pca:+.3f}]")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes, fontsize=7,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
    ax.set_title('PCA: PC1 vs Book Order', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 75)
    print("  DIRECTIONAL DRIFT TESTS — Small‑n Permutation Inference")
    print("=" * 75)
    print()
    print("  Three tests per feature set × distance metric:")
    print("    1. Mantel test: D_style ~ |year gap|")
    print("    2. DBC anchor: distance-to-DBC ~ book order")
    print("    3. PCA PC1: coordinate ~ book order")
    print()
    print(f"  n = 7 Caesarian books → 7! = 5,040 exact permutations")
    print()

    rng = np.random.RandomState(42)

    # Build year-gap matrix (constant across feature sets)
    G_gap, years = build_year_gap_matrix(7)
    gap_to_dbc, _ = build_year_gap_to_dbc(7)

    # Store all results for summary
    all_mantel = []
    all_dbc = []
    all_pca = []

    for feat_key, cfg in FEATURE_SETS.items():
        print(f"\n{'─' * 65}")
        print(f"  {cfg['label']}")
        print(f"{'─' * 65}")

        feat_path = os.path.join(OUTPUTS_DIR, cfg['csv'])
        if not os.path.exists(feat_path):
            print(f"  ⚠ File not found: {feat_path}")
            continue

        labels, all_data, dbg_mask, hirtius_idx, dbc_idx, feat_names = \
            load_feature_matrix(feat_path)
        dbg_labels, dbg_data, dbg_indices = \
            extract_dbg_subset(labels, all_data, dbg_mask)

        print(f"  Loaded: {all_data.shape[0]} samples × {all_data.shape[1]} features")
        print(f"  DBG I-VII: {dbg_data.shape[0]} samples")

        # Data for plotting
        mantel_plot_data = None
        dbc_plot_data = None
        pca_plot_data = None

        # ── Distance matrices ────────────────────────────────────
        # Delta
        D_delta_full = compute_delta_matrix(all_data)
        D_delta_dbg = D_delta_full[dbg_indices][:, dbg_indices]

        # Cosine
        D_cos_full = compute_cosine_distance_matrix(all_data)
        D_cos_dbg = D_cos_full[dbg_indices][:, dbg_indices]

        # ── 1. MANTEL TEST ────────────────────────────────────────
        print(f"\n  ── Mantel Tests ──")

        for dist_name, D_dbg in [('Delta', D_delta_dbg),
                                  ('Cosine', D_cos_dbg)]:
            r_m, p_m, n_perm = mantel_exact(D_dbg, G_gap, rng=rng)
            ci_lo, ci_hi = mantel_bootstrap_ci(D_dbg, G_gap, rng=rng)

            print(f"    {dist_name}: Mantel r = {r_m:+.4f}  "
                  f"p = {p_m:.4f}  "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
                  f"(exact, N={n_perm})")

            all_mantel.append({
                'feature_set': cfg['label'],
                'dist_type': dist_name,
                'mantel_r': r_m,
                'p_mantel': p_m,
                'ci_low': ci_lo,
                'ci_high': ci_hi,
                'n_perm': n_perm,
            })

            # Use Delta for the plot (primary metric)
            if dist_name == 'Delta':
                gaps_flat = matrix_to_upper_triangle(G_gap)
                dists_flat = matrix_to_upper_triangle(D_dbg)
                mantel_plot_data = (gaps_flat, dists_flat,
                                    r_m, p_m, ci_lo, ci_hi)

        # ── 2. DBC ANCHOR ─────────────────────────────────────────
        print(f"\n  ── DBC Anchor ──")

        if dbc_idx is not None:
            # Extract distances to DBC
            dbc_delta = {}
            dbc_cos = {}
            for i, lbl in enumerate(dbg_labels):
                dbc_delta[lbl] = D_delta_full[dbg_indices[i], dbc_idx]
                dbc_cos[lbl] = D_cos_full[dbg_indices[i], dbc_idx]

            for dist_name, dbc_dists in [('Delta', dbc_delta),
                                          ('Cosine', dbc_cos)]:
                books_arr = np.array([int(lbl) for lbl in dbg_labels])
                dists_arr = np.array([dbc_dists[lbl] for lbl in dbg_labels])

                r_s, p_s, n_perm_s = spearman_exact_permutation(
                    books_arr, dists_arr, rng=rng)
                ci_lo_s, ci_hi_s = spearman_bootstrap_ci(
                    books_arr, dists_arr, rng=rng)

                print(f"    {dist_name}: Spearman r = {r_s:+.4f}  "
                      f"p = {p_s:.4f}  "
                      f"95% CI [{ci_lo_s:+.4f}, {ci_hi_s:+.4f}]  "
                      f"(exact, N={n_perm_s})")

                all_dbc.append({
                    'feature_set': cfg['label'],
                    'dist_type': dist_name,
                    'spearman_r': r_s,
                    'p_dbc': p_s,
                    'ci_low': ci_lo_s,
                    'ci_high': ci_hi_s,
                })

                if dist_name == 'Delta':
                    hirtius_delta = D_delta_full[hirtius_idx, dbc_idx] \
                        if hirtius_idx is not None else None
                    dbc_plot_data = (books_arr, dists_arr,
                                     r_s, p_s, ci_lo_s, ci_hi_s,
                                     hirtius_delta)
        else:
            print(f"    DBC not found in feature matrix")

        # ── 3. PCA PC1 ────────────────────────────────────────────
        print(f"\n  ── PCA PC1 ──")

        book_nums_pca, pc1_vals, pc1_var = \
            compute_pca_pc1(all_data, labels, dbg_mask)

        # Two-sided Spearman (PC1 could go either direction)
        r_pca, _ = stats.spearmanr(book_nums_pca, pc1_vals)
        # Two-sided permutation test
        count_extreme = 0
        total = 0
        for perm in permutations(range(len(book_nums_pca))):
            pc1_perm = pc1_vals[list(perm)]
            r_perm, _ = stats.spearmanr(book_nums_pca, pc1_perm)
            if abs(r_perm) >= abs(r_pca):
                count_extreme += 1
            total += 1
        p_pca_two_sided = count_extreme / total

        ci_lo_pca, ci_hi_pca = spearman_bootstrap_ci(
            book_nums_pca, pc1_vals, rng=rng)

        print(f"    PC1 variance: {pc1_var:.1%}")
        print(f"    Spearman r = {r_pca:+.4f}  "
              f"p = {p_pca_two_sided:.4f} (2-sided exact)  "
              f"95% CI [{ci_lo_pca:+.4f}, {ci_hi_pca:+.4f}]")

        all_pca.append({
            'feature_set': cfg['label'],
            'pc1_var': pc1_var,
            'spearman_r': r_pca,
            'p_pca': p_pca_two_sided,
            'ci_low': ci_lo_pca,
            'ci_high': ci_hi_pca,
        })

        pca_plot_data = (book_nums_pca, pc1_vals, pc1_var,
                         r_pca, p_pca_two_sided, ci_lo_pca, ci_hi_pca)

        # ── Plot ──────────────────────────────────────────────────
        safe_key = feat_key.replace('/', '_')
        out_path = os.path.join(FIGURES_DIR, f'drift_{safe_key}.png')
        make_drift_plot(cfg['label'], feat_key,
                        mantel_plot_data, dbc_plot_data, pca_plot_data,
                        out_path)

    # ════════════════════════════════════════════════════════════════
    # 9. SUMMARY
    # ════════════════════════════════════════════════════════════════

    # ── Mantel Summary ──
    print("\n" + "=" * 90)
    print("  MANTEL TEST SUMMARY: D_style ~ |year gap|")
    print("=" * 90)
    print(f"{'Feature Set':<28s} {'Metric':<10s} "
          f"{'Mantel r':>9s} {'p':>8s} {'95% CI low':>10s} "
          f"{'95% CI high':>10s} {'Sig?':>5s}")
    print("-" * 90)
    mantel_sig = 0
    mantel_total = 0
    for r in all_mantel:
        sig = '*' if r['p_mantel'] < 0.05 else ''
        if sig:
            mantel_sig += 1
        mantel_total += 1
        print(f"{r['feature_set']:<28s} {r['dist_type']:<10s} "
              f"{r['mantel_r']:+9.4f} {r['p_mantel']:8.4f} "
              f"{r['ci_low']:+10.4f} {r['ci_high']:+10.4f} {sig:>5s}")
    print(f"\n  Significant (p < 0.05): {mantel_sig}/{mantel_total}")

    # ── DBC Anchor Summary ──
    print("\n" + "=" * 90)
    print("  DBC ANCHOR SUMMARY: Distance-to-DBC ~ Book Order")
    print("=" * 90)
    print(f"{'Feature Set':<28s} {'Metric':<10s} "
          f"{'Spearman r':>10s} {'p':>8s} {'95% CI low':>10s} "
          f"{'95% CI high':>10s} {'Sig?':>5s}")
    print("-" * 90)
    dbc_sig = 0
    dbc_total = 0
    for r in all_dbc:
        sig = '*' if r['p_dbc'] < 0.05 else ''
        if sig:
            dbc_sig += 1
        dbc_total += 1
        print(f"{r['feature_set']:<28s} {r['dist_type']:<10s} "
              f"{r['spearman_r']:+10.4f} {r['p_dbc']:8.4f} "
              f"{r['ci_low']:+10.4f} {r['ci_high']:+10.4f} {sig:>5s}")
    print(f"\n  Significant (p < 0.05): {dbc_sig}/{dbc_total}")

    # ── PCA Summary ──
    print("\n" + "=" * 90)
    print("  PCA PC1 SUMMARY: PC1 coordinate ~ Book Order")
    print("=" * 90)
    print(f"{'Feature Set':<28s} {'PC1 var':>8s} "
          f"{'Spearman |r|':>12s} {'p (2-sided)':>12s} "
          f"{'95% CI':>18s} {'Sig?':>5s}")
    print("-" * 90)
    pca_sig = 0
    pca_total = 0
    for r in all_pca:
        sig = '*' if r['p_pca'] < 0.05 else ''
        if sig:
            pca_sig += 1
        pca_total += 1
        ci_str = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        print(f"{r['feature_set']:<28s} {r['pc1_var']:8.1%} "
              f"{abs(r['spearman_r']):12.4f} {r['p_pca']:12.4f} "
              f"{ci_str:>18s} {sig:>5s}")
    print(f"\n  Significant (p < 0.05): {pca_sig}/{pca_total}")

    # ════════════════════════════════════════════════════════════════
    # 10. CONSENSUS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  CONSENSUS: Does style drift directionally with time?")
    print("=" * 90)

    # Mantel consensus
    mantel_positive = sum(1 for r in all_mantel if r['mantel_r'] > 0)
    mantel_sig_any = sum(1 for r in all_mantel if r['p_mantel'] < 0.05)
    print(f"\n  MANTEL TEST:")
    print(f"    Positive Mantel r (style distance grows with time gap): "
          f"{mantel_positive}/{mantel_total}")
    print(f"    Significant (p < 0.05): {mantel_sig_any}/{mantel_total}")
    if mantel_sig_any > 0:
        print(f"    → Stylistic distance between books INCREASES with "
              f"chronological separation")
        print(f"    → Consistent with Hypothesis A (annual/drift composition)")

    # DBC consensus
    dbc_negative = sum(1 for r in all_dbc if r['spearman_r'] < 0)
    dbc_sig_any = sum(1 for r in all_dbc if r['p_dbc'] < 0.05)
    print(f"\n  DBC ANCHOR:")
    print(f"    Negative r (later books closer to DBC): {dbc_negative}/{dbc_total}")
    print(f"    Significant (p < 0.05): {dbc_sig_any}/{dbc_total}")
    if dbc_sig_any > 0 and dbc_negative > dbc_total / 2:
        print(f"    → Later DBG books are closer to late-Caesar DBC")
        print(f"    → Consistent with Hypothesis A (annual composition)")

    # PCA consensus
    pca_sig_any = sum(1 for r in all_pca if r['p_pca'] < 0.05)
    print(f"\n  PCA PC1:")
    print(f"    Significant ordering along PC1 (p < 0.05): "
          f"{pca_sig_any}/{pca_total}")
    if pca_sig_any > 0:
        print(f"    → Books are not randomly ordered in PC space")

    # Overall
    total_tests = mantel_total + dbc_total + pca_total
    total_sig = mantel_sig + dbc_sig + pca_sig
    print(f"\n  OVERALL:")
    print(f"    Total tests: {total_tests}")
    print(f"    Total significant (p < 0.05): {total_sig}")
    print(f"    All tests use exact permutation (5,040 enumerations each)")
    print(f"    Bootstrap CIs account for small-n uncertainty")

    # Final verdict
    print(f"\n  ┌{'─' * 80}┐")
    if mantel_sig_any >= mantel_total * 0.3 and dbc_sig_any >= dbc_total * 0.3:
        print(f"  │ VERDICT: Evidence SUPPORTS Hypothesis A "
              f"(annual / serial composition).            │")
        print(f"  │ Stylistic signal drifts directionally with "
              f"time across multiple                  │")
        print(f"  │ feature representations and distance "
              f"metrics.                                │")
    elif mantel_sig_any == 0 and dbc_sig_any == 0 and pca_sig_any == 0:
        print(f"  │ VERDICT: No significant chronological signal "
              f"detected.                              │")
        print(f"  │ Results are CONSISTENT with Hypothesis B "
              f"(bulk composition),                   │")
        print(f"  │ but do not confirm it — absence of "
              f"evidence ≠ evidence of absence.        │")
    else:
        print(f"  │ VERDICT: MIXED — some feature sets show "
              f"chronological signal,                │")
        print(f"  │ others do not. See per-feature breakdown "
              f"above.                              │")
    print(f"  └{'─' * 80}┘")

    print(f"\n  CAVEATS:")
    print(f"    - n = 7. Every finding is a correlation over 7 data points.")
    print(f"    - Content-topic confound: books differ in subject matter.")
    print(f"    - Book length varies (4.1K–12.7K tokens). Short books "
          f"have noisier features.")
    print(f"    - The Mantel test is sensitive to distance metric choice.")
    print(f"    - 'Significant' at p < 0.05 means ≤252/5040 permutations "
          f"produce an equal or more extreme result.")
    print()
    print(f"  All plots saved to: {FIGURES_DIR}")
    print("=" * 75)
    print("  ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == '__main__':
    main()
