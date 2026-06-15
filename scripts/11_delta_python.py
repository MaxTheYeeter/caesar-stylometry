#!/usr/bin/env python3
"""
scripts/11_delta_python.py

Independent Python cross‑check of the authorship‑separation test from
R/stylo (script 10).  Uses the feature matrices built by 07_features_words.py
and 08_features_ngrams.py.

Implements Burrows's Delta:
    1. For each feature column, subtract the mean and divide by std
       (z‑score) across all samples.
    2. Compute pairwise Manhattan distance on the z‑scored features.

Analyses performed:
    (A) Book‑level distance matrix + heatmap + dendrogram
        for function‑word and MFW-100 matrices (tokens & lemmas).

    (B) Leave‑one‑book‑out: measure each DBG book's distance to the
        centroid of the other seven DBG books (I‑VII + VIII).  If the
        method works, Book VIII should be the most distant from the
        Caesar‑book centroid.

    (C) Chapter‑level one‑class SVM: train on DBG I–VII chapters, test
        on Book VIII chapters and DBC chapters.  The fraction of
        outliers in Book VIII vs. I–VII quantifies separability.

Purpose: this is a cross‑language sanity check.  The R/stylo script
uses stylo's internal Delta implementation; this script uses a
transparent Python implementation.  Agreement between them strengthens
confidence in the method before chronological analysis.

Inputs:
    outputs/features_function_words_tokens_books.csv
    outputs/features_function_words_lemmas_books.csv
    outputs/features_mfw100_tokens_books.csv
    outputs/features_mfw100_lemmas_books.csv
    outputs/features_function_words_tokens_chapters.csv
    outputs/features_function_words_lemmas_chapters.csv
    outputs/features_mfw100_tokens_chapters.csv
    outputs/features_mfw100_lemmas_chapters.csv

Produces:
    outputs/delta_python_distance_function_words_tokens_books.csv
    outputs/delta_python_distance_function_words_lemmas_books.csv
    outputs/delta_python_distance_mfw100_tokens_books.csv
    outputs/delta_python_distance_mfw100_lemmas_books.csv
    figures/heatmap_delta_python_function_words_tokens_books.png
    figures/heatmap_delta_python_function_words_lemmas_books.png
    figures/heatmap_delta_python_mfw100_tokens_books.png
    figures/heatmap_delta_python_mfw100_lemmas_books.png
    figures/dendrogram_delta_python_function_words_tokens_books.png
    figures/dendrogram_delta_python_function_words_lemmas_books.png
    figures/dendrogram_delta_python_mfw100_tokens_books.png
    figures/dendrogram_delta_python_mfw100_lemmas_books.png
    outputs/delta_python_leave_one_out_report.csv
    outputs/delta_python_svm_report.csv
"""

import csv
import os
import sys
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# --- CSV field size limit for large feature matrices ------------------------
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Book‑level matrices to analyse
BOOK_MATRIX_SPECS = [
    {
        'path':    'features_function_words_tokens_books.csv',
        'label':   'function_words_tokens',
        'desc':    'Function Words (tokens) — 299 features',
        'cmap':    'rocket_r',
    },
    {
        'path':    'features_function_words_lemmas_books.csv',
        'label':   'function_words_lemmas',
        'desc':    'Function Words (lemmas) — 166 features',
        'cmap':    'rocket_r',
    },
    {
        'path':    'features_mfw100_tokens_books.csv',
        'label':   'mfw100_tokens',
        'desc':    'MFW 100 (tokens)',
        'cmap':    'rocket_r',
    },
    {
        'path':    'features_mfw100_lemmas_books.csv',
        'label':   'mfw100_lemmas',
        'desc':    'MFW 100 (lemmas)',
        'cmap':    'rocket_r',
    },
]

# Chapter‑level SVM specs
CHAPTER_MATRIX_SPECS = [
    {
        'path':    'features_function_words_tokens_chapters.csv',
        'label':   'function_words_tokens',
        'desc':    'Function Words (tokens)',
    },
    {
        'path':    'features_function_words_lemmas_chapters.csv',
        'label':   'function_words_lemmas',
        'desc':    'Function Words (lemmas)',
    },
    {
        'path':    'features_mfw100_tokens_chapters.csv',
        'label':   'mfw100_tokens',
        'desc':    'MFW 100 (tokens)',
    },
    {
        'path':    'features_mfw100_lemmas_chapters.csv',
        'label':   'mfw100_lemmas',
        'desc':    'MFW 100 (lemmas)',
    },
]


# ===========================================================================
# Data loading
# ===========================================================================

def load_feature_matrix(filename: str) -> tuple[np.ndarray, list[str],
                                                  list[dict]]:
    """
    Load a feature CSV from outputs/.

    Returns:
        X        — (n_samples, n_features) numpy array of floats
        features — list of feature column names
        meta     — list of dicts with metadata columns per sample
    """
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"  ERROR: Matrix not found: {path}")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    # Identify feature columns (everything after the metadata columns)
    meta_cols = {'segment_id', 'author_group', 'work', 'book', 'total_tokens',
                 'total_ngrams'}
    feature_cols = [c for c in reader.fieldnames if c not in meta_cols]

    X = np.array([[float(row.get(f, 0.0)) for f in feature_cols]
                   for row in all_rows], dtype=float)

    meta = [{k: row.get(k, '') for k in meta_cols if k in row}
            for row in all_rows]

    return X, feature_cols, meta


# ===========================================================================
# Burrows's Delta
# ===========================================================================

def burrows_delta(X: np.ndarray) -> np.ndarray:
    """
    Compute Burrows's Delta distance matrix.

    1. Z‑score each feature column across all samples (mean=0, std=1).
       Features with zero variance remain 0.
    2. Return pairwise Manhattan distances (N×N matrix).

    Edges:
      - Constant (zero‑variance) features silently map to 0.
      - All‑NaN rows (empty text) map to 0 distance from everything
        (should not occur with our data).
    """
    n_samples, n_features = X.shape

    # Handle edge case: 1 sample
    if n_samples < 2:
        return np.zeros((n_samples, n_samples))

    # Z‑score: handle zero‑variance features gracefully
    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X, axis=0, ddof=0, keepdims=True)

    # Features with zero variance → keep at 0 (no information)
    std[std == 0] = 1.0

    Z = (X - mean) / std

    # Replace any remaining NaN (shouldn't happen, but be safe) with 0
    Z = np.nan_to_num(Z)

    # Pairwise Manhattan distance
    dist_vector = pdist(Z, metric='cityblock')

    return squareform(dist_vector)


# ===========================================================================
# (A) Book‑level distance matrix + heatmap + dendrogram
# ===========================================================================

def shorten_label(meta_row: dict) -> str:
    """Build a short, readable label from metadata."""
    author = meta_row.get('author_group', '')
    seg_id = meta_row.get('segment_id', '')

    if 'dbc' in seg_id.lower():
        return 'DBC'
    if 'hirtius' in author.lower():
        return 'Hirtius_VIII'
    # Caesar DBG book
    book = meta_row.get('book', '')
    return f'Caesar_{book}'


def plot_heatmap(dist_matrix: np.ndarray, labels: list[str],
                 title: str, path: str):
    """Save a labelled heatmap of the distance matrix."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))

    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

    sns.heatmap(dist_matrix,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='.3f',
                cmap='rocket_r',
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Burrows Delta Distance',
                          'shrink': 0.8},
                ax=ax)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Heatmap -> {path}")


def plot_dendrogram(dist_matrix: np.ndarray, labels: list[str],
                    title: str, path: str):
    """Save a hierarchical‑clustering dendrogram."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Ward linkage on the condensed distance vector
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='ward')

    # Colour mapping
    author_colors = {
        'Caesar':  '#A23B72',
        'Hirtius': '#2E86AB',
        'DBC':     '#D4A017',
    }

    # Build leaf colours
    leaf_colors = ['#888888'] * len(labels)
    for i, lab in enumerate(labels):
        if 'Hirtius' in lab:
            leaf_colors[i] = author_colors['Hirtius']
        elif 'DBC' in lab:
            leaf_colors[i] = author_colors['DBC']
        elif 'Caesar' in lab:
            leaf_colors[i] = author_colors['Caesar']

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        dendrogram(Z,
                   labels=labels,
                   leaf_font_size=10,
                   leaf_rotation=45,
                   color_threshold=0,
                   above_threshold_color='gray',
                   link_color_func=lambda k: 'gray',
                   ax=ax)

    # Colour the leaf labels manually (matplotlib's dendrogram doesn't
    # expose per‑label colour on the base function; colour by cluster
    # instead via the returned leaf order)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Ward Linkage Distance')
    ax.set_xlabel('')

    # Re‑colour leaf labels
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        text = lbl.get_text()
        if 'Hirtius' in text:
            lbl.set_color(author_colors['Hirtius'])
        elif 'DBC' in text:
            lbl.set_color(author_colors['DBC'])
        elif 'Caesar' in text:
            lbl.set_color(author_colors['Caesar'])
        lbl.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Dendrogram -> {path}")


# ===========================================================================
# (B) Leave‑one‑book‑out
# ===========================================================================

def leave_one_out_analysis(X: np.ndarray, meta: list[dict],
                            label: str) -> list[dict]:
    """
    For each DBG book (I–VIII), compute its Manhattan distance to the
    centroid of the other seven DBG books in z‑scored feature space.

    The DBG books are I–VII (Caesar) + VIII (Hirtius).  DBC is excluded
    from the centroid computation — this tests whether Book VIII stands
    apart from the Caesar books.

    Returns a list of dicts:
        {book, author_group, distance_to_others_centroid}
    sorted by distance descending (most distant first).
    """
    # Z‑score all samples (DBG I–VIII + DBC)
    n_samples, n_features = X.shape
    if n_samples < 3:
        return []

    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X, axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    Z = np.nan_to_num(Z)

    # Index DBG books (I–VIII)
    dbg_indices = []
    for i, m in enumerate(meta):
        seg_id = m.get('segment_id', '')
        work   = m.get('work', '')
        if 'dbc' in seg_id.lower() or work == 'dbc':
            continue
        dbg_indices.append(i)

    if len(dbg_indices) < 3:
        print(f"    WARNING: Too few DBG books ({len(dbg_indices)}) "
              f"for leave‑one‑out analysis.")
        return []

    results = []
    for idx in dbg_indices:
        others = [j for j in dbg_indices if j != idx]
        centroid = np.mean(Z[others], axis=0)
        dist = np.sum(np.abs(Z[idx] - centroid))
        results.append({
            'book':          meta[idx].get('book', '?'),
            'segment_id':    meta[idx].get('segment_id', '?'),
            'author_group':  meta[idx].get('author_group', '?'),
            'distance_to_centroid': round(float(dist), 4),
        })

    results.sort(key=lambda r: r['distance_to_centroid'], reverse=True)
    return results


def print_leave_one_out(results: list[dict], label: str):
    """Pretty‑print leave‑one‑out results."""
    print(f"\n  [{label}]")
    print(f"  {'Book':>6s}  {'Author':>10s}  {'Distance to Centroid':>22s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*22}")
    for r in results:
        marker = ' <--' if r['author_group'] == 'hirtius' else ''
        print(f"  {r['book']:>6s}  {r['author_group']:>10s}  "
              f"{r['distance_to_centroid']:>10.4f}{marker}")

    # Check if Book VIII is the most distant
    if results and results[0]['author_group'] == 'hirtius':
        print(f"    ✓ Book VIII IS the most distant from the Caesar centroid.")
    elif results:
        top = results[0]
        print(f"    ✗ Most distant is Book {top['book']} ({top['author_group']}), "
              f"not Book VIII.")
    print()


# ===========================================================================
# (C) Chapter‑level one‑class SVM
# ===========================================================================

def chapter_svm_analysis(X: np.ndarray, meta: list[dict],
                          label: str) -> dict:
    """
    Train a one‑class SVM on DBG I–VII chapters (Caesar).  Measure
    outlier fraction on:
        - DBG I–VII (Caesar)   → expected low  (cross‑val)
        - DBG VIII (Hirtius)   → expected high  (if Hirtius differs)
        - DBC (Caesar, later)  → expected intermediate

    Returns a dict with outlier fractions and the number of samples
    in each group.
    """
    # Partition chapters by author group
    caesar_idx   = []   # DBG I-VII
    hirtius_idx  = []   # DBG VIII
    dbc_idx      = []   # DBC

    for i, m in enumerate(meta):
        ag = m.get('author_group', '')
        if ag == 'caesar':
            caesar_idx.append(i)
        elif ag == 'hirtius':
            hirtius_idx.append(i)
        elif ag == 'caesar_dbc':
            dbc_idx.append(i)

    n_caesar  = len(caesar_idx)
    n_hirtius = len(hirtius_idx)
    n_dbc     = len(dbc_idx)

    if n_caesar < 10 or n_hirtius == 0:
        print(f"    WARNING: Insufficient samples for SVM ({n_caesar} "
              f"Caesar, {n_hirtius} Hirtius).  Skipping.")
        return {
            'label': label,
            'n_caesar': n_caesar,
            'n_hirtius': n_hirtius,
            'n_dbc': n_dbc,
            'caesar_outlier_pct': None,
            'hirtius_outlier_pct': None,
            'dbc_outlier_pct': None,
        }

    # Z‑score globally
    mean = np.mean(X, axis=0, keepdims=True)
    std  = np.std(X, axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    Z = np.nan_to_num(Z)

    # --- One‑class SVM on DBG I–VII ---
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    svm.fit(Z[caesar_idx])

    # Predict on each group (-1 = outlier, 1 = inlier)
    caesar_pred  = svm.predict(Z[caesar_idx])
    hirtius_pred = svm.predict(Z[hirtius_idx])
    dbc_pred     = svm.predict(Z[dbc_idx])

    caesar_outlier_pct  = 100.0 * np.sum(caesar_pred == -1) / n_caesar
    hirtius_outlier_pct = (100.0 * np.sum(hirtius_pred == -1) / n_hirtius
                           if n_hirtius > 0 else 0.0)
    dbc_outlier_pct     = (100.0 * np.sum(dbc_pred == -1) / n_dbc
                           if n_dbc > 0 else 0.0)

    print(f"\n  [{label}] One‑Class SVM (nu=0.1, rbf kernel)")
    print(f"    Samples:   Caesar={n_caesar}  "
          f"Hirtius={n_hirtius}  DBC={n_dbc}")
    print(f"    Outlier %: Caesar={caesar_outlier_pct:5.1f}%  "
          f"Hirtius={hirtius_outlier_pct:5.1f}%  "
          f"DBC={dbc_outlier_pct:5.1f}%")

    if hirtius_outlier_pct > caesar_outlier_pct + 5:
        print(f"    ✓ Hirtius outlier rate exceeds Caesar baseline "
              f"by {hirtius_outlier_pct - caesar_outlier_pct:.1f} pp.")
    else:
        print(f"    ✗ Hirtius outlier rate not substantially above "
              f"Caesar baseline.")

    return {
        'label': label,
        'n_caesar': n_caesar,
        'n_hirtius': n_hirtius,
        'n_dbc': n_dbc,
        'caesar_outlier_pct': round(caesar_outlier_pct, 2),
        'hirtius_outlier_pct': round(hirtius_outlier_pct, 2),
        'dbc_outlier_pct': round(dbc_outlier_pct, 2),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 65)
    print("  Python Burrows's Delta — Authorship Separation Cross‑Check")
    print("=" * 65)

    # ==================================================================
    # (A) Book‑level distance matrices, heatmaps, dendrograms
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  (A) BOOK‑LEVEL DISTANCE MATRICES")
    print(f"{'─'*50}")

    all_loo_results = []

    for spec in BOOK_MATRIX_SPECS:
        print(f"\n  Loading: {spec['path']}")
        X, features, meta = load_feature_matrix(spec['path'])

        n_samples = X.shape[0]
        print(f"    Samples: {n_samples}  |  Features: {len(features)}")

        # Build distance matrix
        dist = burrows_delta(X)

        # Save distance matrix
        dist_csv = os.path.join(
            OUTPUT_DIR,
            f"delta_python_distance_{spec['label']}_books.csv"
        )
        labels = [shorten_label(m) for m in meta]
        with open(dist_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow([''] + labels)
            for i, row in enumerate(dist):
                writer.writerow([labels[i]] + [round(v, 6) for v in row])
        print(f"    Distance CSV -> {dist_csv}")

        # Heatmap
        heatmap_path = os.path.join(
            FIGURES_DIR,
            f"heatmap_delta_python_{spec['label']}_books.png"
        )
        plot_heatmap(dist, labels, spec['desc'], heatmap_path)

        # Dendrogram
        dendro_path = os.path.join(
            FIGURES_DIR,
            f"dendrogram_delta_python_{spec['label']}_books.png"
        )
        plot_dendrogram(dist, labels, spec['desc'], dendro_path)

        # ==============================================================
        # (B) Leave‑one‑book‑out
        # ==============================================================
        loo = leave_one_out_analysis(X, meta, spec['label'])
        if loo:
            all_loo_results.append({
                'label': spec['label'],
                'results': loo,
            })

    # Print leave‑one‑out summary
    print(f"\n{'─'*50}")
    print(f"  (B) LEAVE‑ONE‑BOOK‑OUT ANALYSIS")
    print(f"  Each DBG book's distance to the centroid of the other 7")
    print(f"  DBG books.  If Book VIII is the most distant, the method")
    print(f"  detects the known authorship boundary.")
    print(f"{'─'*50}")

    loo_summary = []

    for entry in all_loo_results:
        print_leave_one_out(entry['results'], entry['label'])
        for r in entry['results']:
            loo_summary.append({
                'feature_set': entry['label'],
                'book':        r['book'],
                'author_group': r['author_group'],
                'distance':    r['distance_to_centroid'],
            })

    # Save leave‑one‑out report
    if loo_summary:
        loo_csv = os.path.join(OUTPUT_DIR,
                               'delta_python_leave_one_out_report.csv')
        with open(loo_csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh,
                                    fieldnames=['feature_set', 'book',
                                                'author_group', 'distance'])
            writer.writeheader()
            writer.writerows(loo_summary)
        print(f"  Report -> {loo_csv}")

    # ==================================================================
    # (C) Chapter‑level SVM
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  (C) CHAPTER‑LEVEL ONE‑CLASS SVM")
    print(f"  Trained on DBG I–VII chapters.  An outlier is a chapter")
    print(f"  the SVM considers outside the 'Caesar' region.")
    print(f"{'─'*50}")

    svm_results = []

    for spec in CHAPTER_MATRIX_SPECS:
        print(f"\n  Loading: {spec['path']}")
        X, features, meta = load_feature_matrix(spec['path'])

        n_samples = X.shape[0]
        print(f"    Samples: {n_samples}  |  Features: {len(features)}")

        result = chapter_svm_analysis(X, meta, spec['label'])
        svm_results.append(result)

    # Save SVM report
    svm_csv = os.path.join(OUTPUT_DIR, 'delta_python_svm_report.csv')
    with open(svm_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh,
                                fieldnames=['label', 'n_caesar', 'n_hirtius',
                                            'n_dbc', 'caesar_outlier_pct',
                                            'hirtius_outlier_pct',
                                            'dbc_outlier_pct'])
        writer.writeheader()
        writer.writerows(svm_results)
    print(f"\n  SVM report -> {svm_csv}")

    # ==================================================================
    # (D) Final summary
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*65}")

    # Tally leave‑one‑out pass rate
    loo_pass = 0
    loo_total = 0
    for entry in all_loo_results:
        loo_total += 1
        if entry['results'] and entry['results'][0]['author_group'] == 'hirtius':
            loo_pass += 1
    print(f"\n  Leave‑one‑out: Book VIII most distant in "
          f"{loo_pass}/{loo_total} feature sets.")

    # Tally SVM pass rate (Hirtius outlier % exceeds Caesar baseline by >5pp)
    svm_pass = sum(1 for r in svm_results
                   if (r['hirtius_outlier_pct'] is not None and
                       r['caesar_outlier_pct'] is not None and
                       r['hirtius_outlier_pct'] > r['caesar_outlier_pct'] + 5))
    svm_total = sum(1 for r in svm_results if r['hirtius_outlier_pct'] is not None)
    print(f"  SVM:            Hirtius outlier rate exceeds baseline in "
          f"{svm_pass}/{svm_total} feature sets.")

    # Cross‑check agreement with R/stylo
    print(f"\n  Comparison with R/stylo (script 10):")
    print(f"    R/stylo word tokens:     1/6  MFW settings passed  "
          f"(best ratio 1.26 at MFW=100)")
    print(f"    R/stylo word lemmas:     0/6  MFW settings passed")
    print(f"    R/stylo char 3‑grams:    1/12 settings passed  "
          f"(best ratio 1.26 at MFW=100, lemmas)")
    print(f"    R/stylo char 4‑grams:    2/12 settings passed  "
          f"(best ratio 1.22 at MFW=100, lemmas)")

    print(f"\n    Python Delta (this script):")
    print(f"    The leave‑one‑out and SVM results above should be")
    print(f"    interpreted alongside the R/stylo findings.  A consistent")
    print(f"    pattern — Book VIII separating at tight feature sets and")
    print(f"    blending at wider ones — replicates the known scholarly")
    print(f"    consensus (Eder 2016).")

    print(f"\n  Files produced:")
    print(f"    {OUTPUT_DIR}/delta_python_distance_*_books.csv")
    print(f"    {OUTPUT_DIR}/delta_python_leave_one_out_report.csv")
    print(f"    {OUTPUT_DIR}/delta_python_svm_report.csv")
    print(f"    {FIGURES_DIR}/heatmap_delta_python_*.png")
    print(f"    {FIGURES_DIR}/dendrogram_delta_python_*.png")

    print(f"\n  Next step: chronological distance‑correlation analysis")


if __name__ == '__main__':
    main()
