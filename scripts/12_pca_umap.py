#!/usr/bin/env python3
"""
scripts/12_pca_umap.py

Dimensionality reduction for exploratory visual analysis of the style
feature matrices.

Applies PCA (scikit-learn) and UMAP (umap-learn) to selected feature
matrices to project book‑level and chapter‑level segments into 2D.

Plots (saved to figures/):
    pca_book_<feature_set>.png
    umap_book_<feature_set>.png
    pca_chapters_<feature_set>.png
    umap_chapters_<feature_set>.png

Analytical question (from the project overview):
    Do Caesar's Books I–VII spread out in an ORDERED, directional way
    (consistent with annual / serial composition) or cluster as a
    homogeneous blob (consistent with bulk / single‑period composition)?

This is EXPLORATORY and VISUAL only.  Formal significance testing via
permutation tests follows in a later script (scripts/13_chronology_test.py).

Key caveats acknowledged in every plot annotation:
    - n = 7 Caesarian books — any visual trend is a correlation over 7 points
    - UMAP at book level (n=9) is near its minimum viable sample size
    - Content‑topic confound: books differ in subject matter
    - All findings require permutation‑test validation
"""

import csv
import os
import re
import sys
import warnings
from collections import OrderedDict
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import spearmanr

# --- sklearn ----------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- UMAP -------------------------------------------------------------------
import umap

# --- CSV field size ---------------------------------------------------------
csv.field_size_limit(sys.maxsize)


# ===========================================================================
# Paths
# ===========================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')

os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# Feature sets to analyse
# ===========================================================================

# Book‑level: the primary unit of analysis (9 segments — 7 Caesar + Hirtius + DBC)
BOOK_FEATURE_SETS = OrderedDict([
    ('function_words_tokens', 'Function Words (tokens, 299 feat.)'),
    ('function_words_lemmas', 'Function Words (lemmas, 166 feat.)'),
    ('mfw100_tokens',        'MFW 100 (tokens)'),
    ('mfw100_lemmas',        'MFW 100 (lemmas)'),
    ('char3gram',            'Character 3‑grams (500 feat.)'),
    ('char4gram',            'Character 4‑grams (1000 feat.)'),
])

# Chapter‑level: supplementary — 644 segments
CHAPTER_FEATURE_SETS = OrderedDict([
    ('function_words_tokens', 'Function Words (tokens, 299 feat.)'),
    ('function_words_lemmas', 'Function Words (lemmas, 166 feat.)'),
    ('mfw100_tokens',        'MFW 100 (tokens)'),
    ('mfw100_lemmas',        'MFW 100 (lemmas)'),
])


# ===========================================================================
# Colour / marker scheme
# ===========================================================================

# DBG Books I–VII:  continuous colormap from blue (Book I) to red (Book VII)
DBG_CMAP = plt.cm.viridis   # perceptually uniform, colourblind‑friendly

# Hirtius Book VIII:  black diamond
HIRTIUS_COLOR  = '#000000'
HIRTIUS_MARKER = 'D'
HIRTIUS_SIZE   = 100

# DBC Complete:  gold star
DBC_COLOR  = '#D4A017'
DBC_MARKER = '*'
DBC_SIZE   = 180

# Fallback for unknown
UNKNOWN_COLOR = '#AAAAAA'
UNKNOWN_MARKER = 'x'


def get_book_number(meta_row: dict) -> Optional[int]:
    """Extract book number as int, or None for DBC complete."""
    seg_id = meta_row.get('segment_id', '')
    work   = meta_row.get('work', '')
    book   = meta_row.get('book', '')

    if seg_id == 'dbc_complete':
        return None  # DBC is the anchor — no book number

    try:
        return int(book)
    except (ValueError, TypeError):
        return None


def get_style(meta_row: dict) -> dict:
    """
    Return a dict of matplotlib styling kwargs for a single segment.
    """
    book_num = get_book_number(meta_row)
    seg_id   = meta_row.get('segment_id', '')
    author   = meta_row.get('author_group', '')

    # DBC anchor
    if book_num is None or seg_id == 'dbc_complete':
        return {
            'marker': DBC_MARKER,
            'color':  DBC_COLOR,
            's':      DBC_SIZE,
            'edgecolors': 'black',
            'linewidths': 0.6,
        }

    # Hirtius Book VIII
    if author == 'hirtius' or book_num == 8:
        return {
            'marker': HIRTIUS_MARKER,
            'color':  HIRTIUS_COLOR,
            's':      HIRTIUS_SIZE,
            'edgecolors': 'white',
            'linewidths': 0.5,
        }

    # Caesar DBG I–VII: colour by book number on the viridis colormap
    if 1 <= book_num <= 7:
        fraction = (book_num - 1) / 6.0   # 0.0 → 1.0
        return {
            'marker': 'o',
            'color':  DBG_CMAP(fraction),
            's':      80,
            'edgecolors': 'black',
            'linewidths': 0.4,
        }

    # Fallback
    return {
        'marker': UNKNOWN_MARKER,
        'color':  UNKNOWN_COLOR,
        's':      60,
    }


# ===========================================================================
# Data loading
# ===========================================================================

def load_feature_matrix(filename: str) -> tuple[np.ndarray, list[str],
                                                  list[dict], list[str]]:
    """
    Load a feature CSV from outputs/.

    Returns:
        X         — (n_samples, n_features) numpy array
        features  — list of feature column names
        meta      — list of metadata dicts per sample
        seg_ids   — parallel list of segment_id strings
    """
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        all_rows = list(reader)

    meta_cols = {'segment_id', 'author_group', 'work', 'book',
                 'total_tokens', 'total_ngrams'}
    feature_cols = [c for c in reader.fieldnames if c not in meta_cols]

    X = np.array([[float(row.get(f, 0.0)) for f in feature_cols]
                   for row in all_rows], dtype=float)

    meta = [{k: row.get(k, '') for k in meta_cols if k in row}
            for row in all_rows]

    seg_ids = [row.get('segment_id', f'sample_{i}')
               for i, row in enumerate(all_rows)]

    return X, feature_cols, meta, seg_ids


# ===========================================================================
# Dimensionality reduction
# ===========================================================================

def run_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Run PCA on z‑scored features.  Returns (n_samples, 2) coordinates.

    Handles edge case: fewer samples than features (PCA caps at
    min(n_samples, n_features) components).
    """
    n_samples, n_features = X.shape

    # Z‑score
    X_scaled = StandardScaler().fit_transform(X)

    # PCA: request 2 components but cap at the matrix rank
    max_comp = min(n_samples, n_features)
    n_comp = min(n_components, max_comp)

    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)

    # Pad to 2 columns if necessary (shouldn't happen for n≥2)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros_like(coords[:, 0])])

    return coords, pca


def run_umap(X: np.ndarray, n_neighbors: int = 5,
             n_components: int = 2) -> np.ndarray:
    """
    Run UMAP on z‑scored features.  Returns (n_samples, 2) coordinates.

    n_neighbors is capped at n_samples - 2 to avoid errors on tiny
    datasets (e.g., book‑level with 9 samples).
    """
    n_samples = X.shape[0]

    if n_samples < 3:
        # UMAP cannot work with fewer than 3 samples
        return np.zeros((n_samples, 2)), None

    # Adjust n_neighbors for small datasets
    nn = min(n_neighbors, n_samples - 2)
    if nn < 2:
        nn = 2

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=nn,
        min_dist=0.1,
        metric='manhattan',         # consistent with Burrows's Delta
        random_state=42,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        coords = reducer.fit_transform(X_scaled)

    return coords, reducer


# ===========================================================================
# Plotting
# ===========================================================================

def add_colorbar_for_books(fig, ax):
    """Add a colorbar showing the book→colour mapping."""
    import matplotlib.colorbar as mcolorbar

    sm = plt.cm.ScalarMappable(
        cmap=DBG_CMAP,
        norm=plt.Normalize(vmin=1, vmax=7)
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Caesar DBG Book (I → VII)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def add_legend_elements(ax):
    """Add legend entries for Hirtius and DBC."""
    hirtius_handle = mlines.Line2D(
        [], [], marker=HIRTIUS_MARKER, color=HIRTIUS_COLOR,
        markersize=8, linestyle='None',
        label='Hirtius (DBG VIII)', markeredgecolor='white',
        markeredgewidth=0.5)
    dbc_handle = mlines.Line2D(
        [], [], marker=DBC_MARKER, color=DBC_COLOR,
        markersize=10, linestyle='None',
        label='DBC (Caesar, later)', markeredgecolor='black',
        markeredgewidth=0.6)
    ax.legend(handles=[hirtius_handle, dbc_handle],
              fontsize=7, loc='best', framealpha=0.85)


def plot_book_level(coords_pca: np.ndarray, pca_model,
                    coords_umap: np.ndarray, umap_model,
                    meta: list[dict], seg_ids: list[str],
                    feature_label: str, feature_key: str):
    """
    Plot PCA + UMAP for book‑level data (9 segments).

    Left panel:  PCA
    Right panel: UMAP

    Each Caesar book (I–VII) shown as a coloured circle with book number
    label.  Hirtius VIII shown as a black diamond.  DBC shown as a gold star.

    A colourbar maps continuous colour to book number.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    n = coords_pca.shape[0]

    # --- PCA panel ---
    for i in range(n):
        style = get_style(meta[i])
        ax1.scatter(coords_pca[i, 0], coords_pca[i, 1],
                    **style, zorder=3)

        # Label Caesar books with number
        bk = get_book_number(meta[i])
        if bk is not None and 1 <= bk <= 7:
            ax1.annotate(str(bk),
                         (coords_pca[i, 0], coords_pca[i, 1]),
                         textcoords="offset points",
                         xytext=(4, 4),
                         fontsize=7, fontweight='bold',
                         color=DBG_CMAP((bk - 1) / 6.0),
                         alpha=0.9)

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = (pca_model.explained_variance_ratio_[1] * 100
            if pca_model.explained_variance_ratio_.shape[0] > 1 else 0.0)

    ax1.set_xlabel(f'PC1 ({var1:.1f}% var.)', fontsize=9)
    ax1.set_ylabel(f'PC2 ({var2:.1f}% var.)', fontsize=9)
    ax1.set_title('PCA', fontsize=11, fontweight='bold')
    add_legend_elements(ax1)

    # --- UMAP panel ---
    for i in range(n):
        style = get_style(meta[i])
        ax2.scatter(coords_umap[i, 0], coords_umap[i, 1],
                    **style, zorder=3)

        bk = get_book_number(meta[i])
        if bk is not None and 1 <= bk <= 7:
            ax2.annotate(str(bk),
                         (coords_umap[i, 0], coords_umap[i, 1]),
                         textcoords="offset points",
                         xytext=(4, 4),
                         fontsize=7, fontweight='bold',
                         color=DBG_CMAP((bk - 1) / 6.0),
                         alpha=0.9)

    ax2.set_xlabel('UMAP 1', fontsize=9)
    ax2.set_ylabel('UMAP 2', fontsize=9)
    ax2.set_title('UMAP', fontsize=11, fontweight='bold')
    add_legend_elements(ax2)

    # Shared colorbar
    add_colorbar_for_books(fig, ax2)

    # Suptitle
    fig.suptitle(feature_label, fontsize=13, fontweight='bold', y=1.01)

    # --- Annotation block at bottom ---
    # Compute Spearman correlation between PC1 and book number
    dbg_meta = [(i, get_book_number(meta[i]))
                for i in range(n) if get_book_number(meta[i]) is not None
                and get_book_number(meta[i]) <= 7]

    annotation_text = ""
    if len(dbg_meta) >= 4:
        dbg_indices, dbg_books = zip(*dbg_meta)
        dbg_pc1 = coords_pca[list(dbg_indices), 0]
        corr, pval = spearmanr(dbg_books, dbg_pc1)
        annotation_text += (
            f"Spearman r(PC1, book) = {corr:+.3f}  "
            f"(p = {pval:.3f})\n"
        )

    annotation_text += (
        "n = 7 Caesarian books.  Any trend is a correlation over 7 points.\n"
        "Formal significance: permutation test in scripts/13_chronology_test.py\n"
        "UMAP near minimum sample size (n=9); interpret with caution."
    )

    fig.text(0.5, -0.02, annotation_text,
             ha='center', va='top', fontsize=7, style='italic',
             color='#666666',
             transform=fig.transFigure)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'pca_umap_book_{feature_key}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {path}")

    # Print a short text interpretation
    print_interpretation(dbg_books, dbg_pc1, coords_pca, meta, corr, pval)


def plot_chapter_level(coords_pca: np.ndarray, pca_model,
                       coords_umap: np.ndarray, umap_model,
                       meta: list[dict], seg_ids: list[str],
                       feature_label: str, feature_key: str):
    """
    Plot PCA + UMAP for chapter‑level data (644 segments).

    Points are coloured by book (using a discrete colormap across
    DBG 1–8 and DBC 1–3) with low opacity so density patterns emerge.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Build a book‑to‑colour mapping
    all_books = set()
    for m in meta:
        bk = get_book_number(m)
        work = m.get('work', '')
        seg_id = m.get('segment_id', '')
        if bk is not None:
            if work == 'dbc' or 'dbc' in seg_id.lower():
                all_books.add(('DBC', bk))
            else:
                all_books.add(('DBG', bk))

    # Sort: DBG 1-8 then DBC 1-3
    dbg_books = sorted([b for w, b in all_books if w == 'DBG'])
    dbc_books = sorted([b for w, b in all_books if w == 'DBC'])

    n_colors = len(dbg_books) + len(dbc_books)
    # Use tab20 for discrete book colours
    color_cycle = plt.cm.tab20(np.linspace(0, 1, n_colors))

    book_to_color = {}
    for i, bk in enumerate(dbg_books):
        book_to_color[('DBG', bk)] = color_cycle[i]
    for i, bk in enumerate(dbc_books):
        book_to_color[('DBC', bk)] = color_cycle[len(dbg_books) + i]

    # --- PCA panel ---
    for i in range(coords_pca.shape[0]):
        bk = get_book_number(meta[i])
        work = meta[i].get('work', '')
        seg_id = meta[i].get('segment_id', '')

        if bk is None:
            color = UNKNOWN_COLOR
            marker = UNKNOWN_MARKER
            size = 30
        else:
            wkey = 'DBC' if (work == 'dbc' or 'dbc' in seg_id.lower()) else 'DBG'
            color = book_to_color.get((wkey, bk), UNKNOWN_COLOR)
            marker = 'o'
            size = 15

        ax1.scatter(coords_pca[i, 0], coords_pca[i, 1],
                    c=[color], marker=marker, s=size,
                    alpha=0.45, edgecolors='none', zorder=2)

    var1 = pca_model.explained_variance_ratio_[0] * 100
    var2 = (pca_model.explained_variance_ratio_[1] * 100
            if pca_model.explained_variance_ratio_.shape[0] > 1 else 0.0)

    ax1.set_xlabel(f'PC1 ({var1:.1f}% var.)', fontsize=9)
    ax1.set_ylabel(f'PC2 ({var2:.1f}% var.)', fontsize=9)
    ax1.set_title(f'PCA — Chapters (n={coords_pca.shape[0]})',
                  fontsize=11, fontweight='bold')

    # --- UMAP panel ---
    for i in range(coords_umap.shape[0]):
        bk = get_book_number(meta[i])
        work = meta[i].get('work', '')
        seg_id = meta[i].get('segment_id', '')

        if bk is None:
            color = UNKNOWN_COLOR
            marker = UNKNOWN_MARKER
            size = 30
        else:
            wkey = 'DBC' if (work == 'dbc' or 'dbc' in seg_id.lower()) else 'DBG'
            color = book_to_color.get((wkey, bk), UNKNOWN_COLOR)
            marker = 'o'
            size = 15

        ax2.scatter(coords_umap[i, 0], coords_umap[i, 1],
                    c=[color], marker=marker, s=size,
                    alpha=0.45, edgecolors='none', zorder=2)

    ax2.set_xlabel('UMAP 1', fontsize=9)
    ax2.set_ylabel('UMAP 2', fontsize=9)
    ax2.set_title(f'UMAP — Chapters (n={coords_umap.shape[0]})',
                  fontsize=11, fontweight='bold')

    # Legend for chapter plot: one entry per book
    legend_handles = []
    for bk in dbg_books:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', linestyle='None',
                          color=book_to_color[('DBG', bk)],
                          markersize=6, label=f'DBG {bk}'))
    for bk in dbc_books:
        legend_handles.append(
            mlines.Line2D([], [], marker='o', linestyle='None',
                          color=book_to_color[('DBC', bk)],
                          markersize=6, label=f'DBC {bk}'))

    # Place legend to the right of the right panel
    ax2.legend(handles=legend_handles, fontsize=5,
               loc='upper left', bbox_to_anchor=(1.02, 1.0),
               ncol=2, framealpha=0.7)

    fig.suptitle(feature_label, fontsize=13, fontweight='bold', y=1.01)

    fig.text(0.5, -0.02,
             "Each point = one chapter (n=644).  "
             "Opacity reduced to show density.  "
             "Formal significance via permutation test in script 13.",
             ha='center', va='top', fontsize=7, style='italic',
             color='#666666', transform=fig.transFigure)

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f'pca_umap_chapters_{feature_key}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {path}")


# ===========================================================================
# Interpretation helper
# ===========================================================================

def print_interpretation(dbg_books, dbg_pc1, coords_pca, meta,
                          corr, pval):
    """Print a short textual interpretation of the book‑level PCA."""
    if len(dbg_books) < 4:
        return

    print(f"      PC1 vs book: Spearman r = {corr:+.3f} (p = {pval:.3f})")

    # Check DBC position relative to Caesar books
    dbc_idx = None
    for i, m in enumerate(meta):
        if m.get('segment_id', '') == 'dbc_complete':
            dbc_idx = i
            break

    if dbc_idx is not None and len(dbg_books) >= 3:
        caesar_pc1 = coords_pca[[i for i, _m in enumerate(meta)
                                  if get_book_number(_m) is not None
                                  and get_book_number(_m) <= 7], 0]
        dbc_pc1 = coords_pca[dbc_idx, 0]

        # Which Caesar extreme is DBC closer to?
        early_books = [i for i, _m in enumerate(meta)
                       if get_book_number(_m) in (1, 2, 3)]
        late_books  = [i for i, _m in enumerate(meta)
                       if get_book_number(_m) in (5, 6, 7)]

        if early_books and late_books:
            early_mean = coords_pca[early_books, 0].mean()
            late_mean  = coords_pca[late_books, 0].mean()

            dist_early = abs(dbc_pc1 - early_mean)
            dist_late  = abs(dbc_pc1 - late_mean)

            if dist_late < dist_early:
                print(f"      DBC is closer to LATE books along PC1 "
                      f"(d_late={dist_late:.3f} < d_early={dist_early:.3f})")
                print(f"      → Consistent with Hypothesis A (annual "
                      f"composition: later DBG books closer to later DBC)")
            else:
                print(f"      DBC is closer to EARLY books along PC1 "
                      f"(d_early={dist_early:.3f} < d_late={dist_late:.3f})")

    # Qualitative assessment
    if abs(corr) > 0.6 and pval < 0.10:
        print(f"      → VISUAL: Books spread along PC1 in rough "
              f"chronological order.")
    elif abs(corr) > 0.3:
        print(f"      → VISUAL: Weak directional trend; mostly a blob "
              f"with some ordering.")
    else:
        print(f"      → VISUAL: No clear directional trend; books "
              f"appear as a homogeneous cluster.")

    print(f"      CAVEAT: n=7; formal permutation test pending.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 65)
    print("  PCA & UMAP — Exploratory Visual Analysis")
    print("=" * 65)
    print()

    # ==================================================================
    # Book‑level analysis
    # ==================================================================
    print("─" * 50)
    print("  BOOK‑LEVEL (9 segments)")
    print("─" * 50)

    for feature_key, feature_label in BOOK_FEATURE_SETS.items():
        csv_filename = f'features_{feature_key}_books.csv'
        print(f"\n  Loading: {csv_filename}")

        try:
            X, features, meta, seg_ids = load_feature_matrix(csv_filename)
        except FileNotFoundError as e:
            print(f"    SKIPPED: {e}")
            continue

        print(f"    Samples: {X.shape[0]}  |  Features: {X.shape[1]}")

        # PCA
        coords_pca, pca_model = run_pca(X)
        print(f"    PCA:    PC1={pca_model.explained_variance_ratio_[0]*100:.1f}%  "
              f"PC2={pca_model.explained_variance_ratio_[1]*100:.1f}%")

        # UMAP with small n_neighbors (book‑level has only 9 samples)
        coords_umap, umap_model = run_umap(X, n_neighbors=3)
        if umap_model is not None:
            print(f"    UMAP:   n_neighbors=3 (near minimum for n=9)")
        else:
            print(f"    UMAP:   SKIPPED (too few samples)")
            coords_umap = np.zeros((X.shape[0], 2))

        # Plot
        plot_book_level(coords_pca, pca_model,
                        coords_umap, umap_model,
                        meta, seg_ids, feature_label, feature_key)

    # ==================================================================
    # Chapter‑level analysis
    # ==================================================================
    print(f"\n{'─'*50}")
    print(f"  CHAPTER‑LEVEL (644 segments)")
    print(f"{'─'*50}")

    for feature_key, feature_label in CHAPTER_FEATURE_SETS.items():
        csv_filename = f'features_{feature_key}_chapters.csv'
        print(f"\n  Loading: {csv_filename}")

        try:
            X, features, meta, seg_ids = load_feature_matrix(csv_filename)
        except FileNotFoundError as e:
            print(f"    SKIPPED: {e}")
            continue

        print(f"    Samples: {X.shape[0]}  |  Features: {X.shape[1]}")

        # PCA
        coords_pca, pca_model = run_pca(X)
        print(f"    PCA:    PC1={pca_model.explained_variance_ratio_[0]*100:.1f}%  "
              f"PC2={pca_model.explained_variance_ratio_[1]*100:.1f}%")

        # UMAP with moderate n_neighbors for 644 samples
        coords_umap, umap_model = run_umap(X, n_neighbors=15)

        # Plot
        plot_chapter_level(coords_pca, pca_model,
                           coords_umap, umap_model,
                           meta, seg_ids, feature_label, feature_key)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*65}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*65}")

    print(f"""
  All plots saved to:  {FIGURES_DIR}/

  INTERPRETATION NOTES:
    - This is EXPLORATORY and VISUAL only.
    - Formal significance testing follows in scripts/13_chronology_test.py
      (permutation‑tested distance‑chronology correlation).

    - Hypothesis A (annual composition) predicts:
        * DBG Books I–VII spread out along PC1/UMAP1 in chronological order.
        * DBC (late Caesar) sits near Books V–VII, not I–III.

    - Hypothesis B (bulk composition) predicts:
        * DBG Books I–VII appear as a homogeneous cluster with no ordering.
        * DBC is roughly equidistant from all DBG books.

    - The n=7 constraint means any visual pattern could arise by chance.
      The permutation test in script 13 will quantify this formally.
""")

    print(f"  Next step: scripts/13_chronology_test.py "
          f"(permutation‑tested chronological analysis)")


if __name__ == '__main__':
    main()
