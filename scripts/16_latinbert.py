#!/usr/bin/env python3
"""
scripts/16_latinbert.py

Latin BERT Embedding Distances — Modern Cross-Check

Extracts contextual embeddings for each chapter using a pretrained Latin BERT
model, mean-pools to per-chapter vectors, aggregates to book-level centroids,
and computes embedding-based distance matrices.

Two key analyses (as cross-checks, NOT standalone claims):
  (a) HIRTUS CONTROL: Does Book VIII sit farthest from the DBG I–VII centroid?
  (b) DBC ANCHOR: Is there a monotonic distance-to-DBC trend across I–VII?

CRITICAL CAVEATS:
  - n = 7–8 books. Embedding models have hundreds of millions of parameters.
    Overfitting to 7 data points is trivially easy. These results are a
    RELATIVE CROSS-CHECK on the classical-stylometry findings, not an
    independent authorship claim.
  - Latin BERT was pretrained on heterogeneous Latin corpora (classical,
    medieval, ecclesiastical). Domain shift from Caesar's prose is unknown.
  - CPU-only inference: ~20 minutes for 644 chapters.

Model: LuisAVasquez/simple-latin-bert-uncased (105M params, 25K vocab)
       Based on BERT-base-uncased architecture, trained on Latin.
"""

import csv
import os
import sys
import time
import warnings
from collections import OrderedDict
from itertools import permutations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine as cosine_distance_fn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, 'outputs')
FIGURES_DIR  = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Model configuration ────────────────────────────────────────────────
MODEL_NAME = 'LuisAVasquez/simple-latin-bert-uncased'
BATCH_SIZE = 16          # chapters per forward pass (CPU-friendly)
MAX_LENGTH = 512          # BERT max; all chapters fit comfortably

# ── Campaign year assignments ───────────────────────────────────────────
DBG_YEARS = {1: 58, 2: 57, 3: 56, 4: 55, 5: 54, 6: 53, 7: 52}


# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL LOADING (with graceful offline failure)
# ═══════════════════════════════════════════════════════════════════════
def load_latin_bert():
    """
    Attempt to load Latin BERT. Returns (tokenizer, model) or raises
    a clear error with setup instructions.
    """
    print(f"  Loading model: {MODEL_NAME}")
    print(f"    First-time download: ~420 MB. Subsequent runs use cache.")
    print()

    try:
        import transformers
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit(
            "\n✗ 'transformers' not installed.\n"
            "  Run: pip install transformers\n"
            "  Then re-run this script.\n"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        sys.exit(
            f"\n✗ Failed to load tokenizer for '{MODEL_NAME}'.\n"
            f"  Error: {e}\n"
            f"  Check internet connection or try a different model.\n"
        )

    try:
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        sys.exit(
            f"\n✗ Failed to load model weights for '{MODEL_NAME}'.\n"
            f"  Error: {e}\n"
        )

    # Check for pooler (not needed, just informative)
    has_pooler = hasattr(model, 'pooler') and model.pooler is not None
    if not has_pooler:
        print("    (No pooler layer — using mean-pooling of last hidden state)")

    params = sum(p.numel() for p in model.parameters())
    print(f"    ✓ Loaded: {tokenizer.vocab_size} vocab, {params:,} params")
    print(f"    ✓ Architecture: {type(model).__name__}")
    print()

    return tokenizer, model


# ═══════════════════════════════════════════════════════════════════════
# 2. CHAPTER EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════════
def embed_chapters(chapters, tokenizer, model):
    """
    Embed a list of chapter dicts. Returns list of numpy arrays.
    Mean-pools the last hidden state (excluding padding tokens).
    CPU-friendly batched processing with progress display.
    """
    import torch

    embeddings = []
    n = len(chapters)
    t0 = time.time()

    for batch_start in range(0, n, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_texts = [chapters[i]['text'] for i in range(batch_start, batch_end)]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        # Move to CPU (explicit, since no GPU)
        inputs = {k: v.cpu() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state: (batch, seq_len, hidden_dim)
        hidden = outputs.last_hidden_state.cpu().numpy()
        attention_mask = inputs['attention_mask'].cpu().numpy()

        # Mean-pool over non-padding tokens
        for i in range(hidden.shape[0]):
            mask = attention_mask[i][:, np.newaxis]  # (seq_len, 1)
            masked_hidden = hidden[i] * mask
            pooled = masked_hidden.sum(axis=0) / mask.sum()
            embeddings.append(pooled)

        # Progress
        elapsed = time.time() - t0
        done = batch_end
        pct = 100 * done / n
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"\r    Embedding: {done:3d}/{n} ({pct:4.0f}%)  "
              f"{rate:.1f} ch/s  ETA: {eta:.0f}s  ", end='', flush=True)

    print()
    total_elapsed = time.time() - t0
    print(f"    ✓ {n} chapter embeddings in {total_elapsed:.1f}s "
          f"({n / total_elapsed:.1f} ch/s)")
    print()
    return embeddings


# ═══════════════════════════════════════════════════════════════════════
# 3. BOOK-LEVEL AGGREGATION
# ═══════════════════════════════════════════════════════════════════════
def aggregate_to_books(chapters, embeddings):
    """
    Group chapter embeddings into book centroids (weighted by chapter length
    in characters, since longer chapters contribute more signal).

    Returns:
        book_data: dict keyed by label ('1'..'7', 'VIII', 'DBC') mapping to
                   {'centroid': np.array, 'n_chapters': int}
    """
    # Group indices by label
    groups = OrderedDict()
    for i, ch in enumerate(chapters):
        work = ch['work']
        book = ch['book']

        if work == 'dbg' and 1 <= book <= 7:
            label = str(book)
        elif work == 'dbg' and book == 8:
            label = 'VIII'
        elif work == 'dbc':
            label = 'DBC'
        else:
            continue

        if label not in groups:
            groups[label] = []
        groups[label].append(i)

    # Compute weighted centroids
    book_data = {}
    for label, indices in groups.items():
        # Weight by character length
        weights = np.array([len(chapters[i]['text']) for i in indices],
                           dtype=np.float64)
        weights /= weights.sum()
        centroid = np.zeros_like(embeddings[0])
        for i, w in zip(indices, weights):
            centroid += embeddings[i] * w
        book_data[label] = {
            'centroid': centroid,
            'n_chapters': len(indices),
            'total_chars': sum(len(chapters[i]['text']) for i in indices),
        }

    return book_data


# ═══════════════════════════════════════════════════════════════════════
# 4. DISTANCE & STATISTICS
# ═══════════════════════════════════════════════════════════════════════
def compute_cosine_distance_matrix(centroids, labels):
    """Pairwise cosine distance between book centroids."""
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance_fn(centroids[labels[i]], centroids[labels[j]])
            D[i, j] = d
            D[j, i] = d
    return D


def hirtius_separation_check(centroids, labels):
    """
    Test: is Book VIII the most distant DBG-relative book from the
    DBG I-VII centroid?
    """
    # DBG I-VII centroid
    dbg_labels = [str(i) for i in range(1, 8)]
    dbg_vecs = [centroids[l] for l in dbg_labels]
    caesar_centroid = np.mean(dbg_vecs, axis=0)

    # Distances of each book from Caesar centroid
    distances = {}
    for lbl in dbg_labels + ['VIII', 'DBC']:
        if lbl in centroids:
            distances[lbl] = cosine_distance_fn(centroids[lbl], caesar_centroid)

    # Rank by distance (largest first)
    ranked = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    return ranked, distances


def dbc_anchor_spearman(centroids, labels):
    """
    Compute distance-to-DBC for DBG I-VII and test Spearman r.
    Null: no ordering. Annual hypothesis: negative r (later books closer).
    """
    if 'DBC' not in centroids:
        return None, None, None

    dbc_vec = centroids['DBC']
    books = []
    dists = []
    for i in range(1, 8):
        lbl = str(i)
        if lbl in centroids:
            books.append(i)
            dists.append(cosine_distance_fn(centroids[lbl], dbc_vec))

    books = np.array(books, dtype=float)
    dists = np.array(dists)

    # Spearman r
    r_obs, _ = stats.spearmanr(books, dists)

    # Exact permutation test (one-sided: negative r expected)
    count = 0
    total = 0
    n = len(books)
    for perm in permutations(range(n)):
        d_perm = dists[list(perm)]
        r_perm, _ = stats.spearmanr(books, d_perm)
        if r_perm <= r_obs:
            count += 1
        total += 1

    p_val = count / total
    return r_obs, p_val, (books, dists)


# ═══════════════════════════════════════════════════════════════════════
# 5. PLOTTING
# ═══════════════════════════════════════════════════════════════════════
def make_embedding_plots(centroids, labels, hirtius_ranked, hirtius_dists,
                         dbc_r, dbc_p, dbc_data, out_dir):
    """Two-panel figure: Hirtius separation + DBC anchor."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    dbg_labels = [str(i) for i in range(1, 8)]

    # ── Panel 1: Hirtius separation ──
    ax = axes[0]
    plot_labels = dbg_labels + ['VIII', 'DBC']
    plot_labels_present = [l for l in plot_labels if l in hirtius_dists]
    plot_dists = [hirtius_dists[l] for l in plot_labels_present]
    colors = []
    for l in plot_labels_present:
        if l in dbg_labels:
            colors.append('#2166ac')       # Caesar I-VII
        elif l == 'VIII':
            colors.append('#b2182b')        # Hirtius
        else:
            colors.append('#4daf4a')        # DBC

    bars = ax.bar(range(len(plot_labels_present)), plot_dists, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(plot_labels_present)))
    ax.set_xticklabels([l if l != 'DBC' else 'DBC' for l in plot_labels_present],
                       fontsize=9)
    ax.set_ylabel('Cosine distance to DBG I–VII centroid')
    ax.set_title('Hirtius Separation: Distance from Caesar Centroid',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight Book VIII
    if 'VIII' in hirtius_dists:
        viii_dist = hirtius_dists['VIII']
        max_dbg = max(hirtius_dists[l] for l in dbg_labels if l in hirtius_dists)
        ratio = viii_dist / max_dbg if max_dbg > 0 else 1.0
        ax.text(0.97, 0.97,
                f"Book VIII / max DBG = {ratio:.2f}\n"
                f"{'✓ Hirtius most distant' if ratio > 1 else '⚠ Not most distant'}",
                transform=ax.transAxes, fontsize=8, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ── Panel 2: DBC Anchor ──
    ax = axes[1]
    if dbc_data is not None:
        books_arr, dists_arr = dbc_data
        ax.scatter(books_arr, dists_arr, c='#2166ac', s=80, zorder=5)
        for b, d in zip(books_arr, dists_arr):
            ax.annotate(str(int(b)), (b, d), textcoords="offset points",
                        xytext=(0, 8), fontsize=9, ha='center',
                        color='#2166ac')
        # Trend line
        if len(books_arr) > 1:
            slope, intercept, _, _, _ = stats.linregress(books_arr, dists_arr)
            x_line = np.linspace(0.5, 7.5, 50)
            ax.plot(x_line, slope * x_line + intercept,
                    '--', color='#d6604d', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Book Number')
        ax.set_ylabel('Cosine distance to DBC centroid')
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels(['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'])
        ax.set_xlim(0.3, 7.7)
        ax.set_title('DBC Anchor: Embedding Distance to DBC',
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if dbc_r is not None:
            direction = 'Annual (later closer)' if dbc_r < 0 else \
                        'Reverse (later farther)'
            ax.text(0.97, 0.97,
                    f"Spearman r = {dbc_r:+.3f}\n"
                    f"p = {dbc_p:.4f} (exact)\n"
                    f"→ {direction}",
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round',
                              facecolor='lightgreen' if dbc_r < 0 else 'lightcoral',
                              alpha=0.8))

    plt.tight_layout()
    path = os.path.join(out_dir, 'latinbert_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Plot saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  LATIN BERT — Embedding-Based Cross-Check")
    print("=" * 65)
    print()

    # ── Load model ──
    tokenizer, model = load_latin_bert()

    # ── Load chapters ──
    chapters_csv = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                                'corpus_chapters_normalized.csv')
    if not os.path.exists(chapters_csv):
        chapters_csv = os.path.join(PROJECT_ROOT, 'data', 'corpus',
                                    'corpus_chapters.csv')
    if not os.path.exists(chapters_csv):
        sys.exit(f"✗ Chapter CSV not found. Checked:\n  {chapters_csv}")

    print(f"  Loading chapters: {chapters_csv}")
    chapters = []
    with open(chapters_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        work_idx = header.index('work')
        book_idx = header.index('book')
        text_idx = header.index('text')
        for row in reader:
            chapters.append({
                'work': row[work_idx],
                'book': int(row[book_idx]),
                'text': row[text_idx],
            })

    n_dbg = sum(1 for ch in chapters if ch['work'] == 'dbg')
    n_dbc = sum(1 for ch in chapters if ch['work'] == 'dbc')
    print(f"    DBG chapters: {n_dbg}  |  DBC chapters: {n_dbc}  "
          f"|  Total: {len(chapters)}")
    print()

    # ── Extract embeddings ──
    print("  Extracting chapter embeddings...")
    embeddings = embed_chapters(chapters, tokenizer, model)

    if len(embeddings) != len(chapters):
        sys.exit(f"✗ Mismatch: {len(embeddings)} embeddings vs "
                 f"{len(chapters)} chapters")

    print(f"    Embedding dimension: {embeddings[0].shape[0]}")
    print(f"    Total vectors: {len(embeddings)}")
    print()

    # ── Aggregate to books ──
    print("  Aggregating to book centroids...")
    book_data = aggregate_to_books(chapters, embeddings)

    centroids = {lbl: d['centroid'] for lbl, d in book_data.items()}
    labels_list = list(centroids.keys())

    print(f"    Books: {sorted(labels_list)}")
    for lbl in ['1', '2', '3', '4', '5', '6', '7', 'VIII', 'DBC']:
        if lbl in book_data:
            bd = book_data[lbl]
            print(f"      Book {'I' if lbl == '1' else
                           'II' if lbl == '2' else
                           'III' if lbl == '3' else
                           'IV' if lbl == '4' else
                           'V' if lbl == '5' else
                           'VI' if lbl == '6' else
                           'VII' if lbl == '7' else
                           'VIII' if lbl == 'VIII' else 'DBC'}: "
                  f"{bd['n_chapters']} chapters, {bd['total_chars']:,} chars")
    print()

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS A: Hirtius Separation
    # ════════════════════════════════════════════════════════════════
    print("─" * 50)
    print("  A. HIRTUS SEPARATION CHECK")
    print("─" * 50)

    hirtius_ranked, hirtius_dists = hirtius_separation_check(centroids,
                                                              labels_list)

    print("  Distance from DBG I–VII centroid:")
    for lbl, dist in hirtius_ranked:
        marker = ''
        if lbl == 'VIII':
            marker = '  ← Hirtius'
        elif lbl == 'DBC':
            marker = '  ← Late Caesar'
        print(f"    {lbl:>4s}: {dist:.6f}{marker}")

    # Is Book VIII the most distant?
    dbg_lbls = [str(i) for i in range(1, 8)]
    max_dbg_dist = max(hirtius_dists[l] for l in dbg_lbls if l in hirtius_dists)
    viii_dist = hirtius_dists.get('VIII', 0)
    ratio_viii = viii_dist / max_dbg_dist if max_dbg_dist > 0 else 1.0

    if ratio_viii > 1.0:
        print(f"\n    ✓ Book VIII is {ratio_viii:.2f}× farther from Caesar "
              f"centroid than any DBG book.")
        print(f"      Embedding space CONFIRMS Hirtius as stylistically "
              f"distinct.")
    else:
        print(f"\n    ⚠ Book VIII distance ratio = {ratio_viii:.2f} (≤1.0)")
        print(f"      Hirtius NOT separable from Caesar in embedding space.")
        print(f"      This may reflect BERT's domain-general pretraining "
              f"overwhelming individual author signal at small scale.")

    print()

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS B: DBC Anchor
    # ════════════════════════════════════════════════════════════════
    print("─" * 50)
    print("  B. DBC ANCHOR TEST")
    print("─" * 50)

    dbc_r, dbc_p, dbc_data = dbc_anchor_spearman(centroids, labels_list)

    if dbc_r is not None:
        books_arr, dists_arr = dbc_data
        print(f"  Book → DBC distance (cosine):")
        for b, d in zip(books_arr, dists_arr):
            roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'][int(b) - 1]
            print(f"    Book {roman}: {d:.6f}")
        print(f"\n  Spearman r = {dbc_r:+.4f}")
        print(f"  Exact permutation p = {dbc_p:.4f}  (5,040 permutations)")
        print(f"  n = {len(books_arr)}")

        if dbc_r < 0:
            print(f"  → Later books are CLOSER to DBC (annual direction)")
            if dbc_p < 0.05:
                print(f"  → STATISTICALLY SIGNIFICANT (p < 0.05)")
            else:
                print(f"  → NOT significant at p < 0.05, but direction "
                      f"consistent with Annual hypothesis")
        else:
            print(f"  → Direction is REVERSED (later books FARTHER from DBC)")
    else:
        print("  DBC not found in chapter data.")
    print()

    # ════════════════════════════════════════════════════════════════
    # CROSS-REFERENCE
    # ════════════════════════════════════════════════════════════════
    print("─" * 50)
    print("  C. CROSS-REFERENCE WITH CLASSICAL STYLOMETRY")
    print("─" * 50)
    print()
    print("  Classical stylometry (scripts 13–14):")
    print("    - Hirtius: CONFIRMED separable (SVM, Delta, function words)")
    print("    - DBC Anchor: r = −0.964 to −0.714, significant in 19/24 tests")
    print("    - Mantel drift: 22/22 positive r, 9/22 significant")
    print()
    print(f"  Latin BERT embeddings (this script):")
    print(f"    - Hirtius: {'CONFIRMED' if ratio_viii > 1.0 else 'NOT confirmed'}")
    if dbc_r is not None:
        print(f"    - DBC Anchor: r = {dbc_r:+.4f}, p = {dbc_p:.4f} "
              f"({'sig' if dbc_p < 0.05 else 'ns'})")
    print()

    # ════════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════════
    print("  Generating plots...")
    make_embedding_plots(centroids, labels_list,
                         hirtius_ranked, hirtius_dists,
                         dbc_r, dbc_p, dbc_data,
                         FIGURES_DIR)

    # ════════════════════════════════════════════════════════════════
    # SAVE EMBEDDINGS
    # ════════════════════════════════════════════════════════════════
    emb_out = os.path.join(OUTPUTS_DIR, 'latinbert_book_centroids.npz')
    np.savez_compressed(emb_out,
                        labels=np.array(labels_list),
                        centroids=np.array([centroids[l] for l in labels_list]))
    print(f"  Embedding centroids saved: {emb_out}")
    print()

    # ════════════════════════════════════════════════════════════════
    # CAVEATS
    # ════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  IMPORTANT CAVEATS")
    print("=" * 65)
    print()
    print("  1. n = 7–8 books. BERT has 105M parameters. Overfitting")
    print("     trivially possible. Embedding distances are a RELATIVE")
    print("     cross-check, not an independent authorship claim.")
    print()
    print("  2. Latin BERT was pretrained on heterogeneous Latin corpora.")
    print("     Domain shift from Caesar's military prose is unquantified.")
    print()
    print("  3. Contextual embeddings capture semantics + syntax + genre.")
    print("     Unlike function-word Delta, we cannot separate 'style'")
    print("     from 'content' in embedding space.")
    print()
    print("  4. The agreement (or disagreement) between classical stylometry")
    print("     and embedding distances is INFORMATIVE but should not be")
    print("     treated as a weighted vote. Methods differ fundamentally.")
    print()
    print("=" * 65)
    print("  ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == '__main__':
    main()
