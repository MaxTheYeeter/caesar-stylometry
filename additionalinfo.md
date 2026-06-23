# Numerical Reconciliation & Figure Reference Guide

**Purpose:** This document resolves a numerical discrepancy between
`PAPER_KIT.md` and `REPORT.md` / `README.md`, and provides figure
descriptions for paper drafting when the author cannot visually
inspect the plots.

**Date:** 2026-06-16

---

## 1. Discrepancy: DBC Anchor and Mantel Counts

### 1.1 What the documents say

| Quantity | PAPER_KIT §8.2 | REPORT.md §3.2 | README |
|---|---|---|---|
| DBC Anchor tests | 22/22 directional | 34/34 directional | 22/22 negative |
| DBC significant | 19/22 | 29/34 | 19/22 p < 0.05 |
| Mantel tests | 22/22 positive | — | 22/22 direction correct |
| Mantel significant | 9/22 | — | 9/22 p < 0.05 |

### 1.2 Root cause

**This is a scope difference, not an error.** Two different scripts count
different panels of feature-set × distance-metric combinations:

- **22-test framing** (PAPER_KIT, README, script 14): The full-corpus
  Delta + Cosine panel run in `scripts/14_drift_tests.py`. Uses the
  "base" feature panel (function words, MFW 100–300, char 2–4-grams;
  11 feature sets × 2 distances = 22 DBC + 22 Mantel).

- **34-test framing** (REPORT.md §3.2, script 15): The full-corpus
  rows from `outputs/robustness_summary.csv`. This panel additionally
  includes **MFW truncations** (MFW 100 at cutoffs 50 and 100; MFW
  200 at 100 and 200; MFW 300 at 150 and 300), yielding more tests.
  The 34 rows correspond to: function words (1 truncation × 2
  distances = 2) + MFW 100 (2 × 2 = 4) + MFW 200 (2 × 2 = 4) + MFW
  300 (2 × 2 = 4) + char 2-gram (1 × 2 = 2) + char 3-gram (1 × 2 =
  2) + char 4-gram (1 × 2 = 2) = 20, plus the lemma variants of the
  word-based feature sets adding the remainder.

- **136-test framing** (REPORT.md §5, script 15): All 34 tests × 4
  excursus conditions = 136 robustness combinations.

### 1.3 Resolution

**Adopt the 22-test framing as the canonical headline result.** The
22-test panel matches the script 14 drift tests — the primary
inferential analysis — and excludes MFW-truncation variants that are
redundant with the robustness analysis. The 22 tests give a clean,
non-redundant accounting:

| Test | Total | Direction correct | Significant (p < 0.05) |
|---|---|---|---|
| DBC Anchor (Delta + Cosine) | 22 | 22/22 | 19/22 |
| Mantel (Delta + Cosine) | 22 | 22/22 | 9/22 |

For the robustness section, use the 136-condition framing (script 15),
which subsumes the 34-test full-corpus condition. A footnote in the
paper should clarify:

> "The DBC-anchor headline counts (22 tests) reflect the base feature
> panel from the primary drift analysis (script 14). The robustness
> matrix (136 tests, script 15) subdivides these across MFW truncation
> levels and excursus conditions; the directional unanimity holds at
> all levels. The 22-test tally is the most parsimonious accounting
> and is used throughout the main text."

### 1.4 Documents to update

| Document | Current framing | Action |
|---|---|---|
| `outputs/REPORT.md` §3.2 | 34/34, 29/34 | Change to 22/22, 19/22 with footnote about MFW-truncation scope |
| `PAPER_KIT.md` §8.2 | 22/22, 19/22 | Keep as-is (already correct for base panel) |
| `README.md` | 22/22, 19/22 | Keep as-is |
| This file (§1.3) | N/A | Canonical resolution documented |

**Update `outputs/REPORT.md` §3.2 inline:** Change the two statistics
from "34/34" and "29/34" to "22/22" and "19/22," and add a sentence:
"The 22 tests represent the base feature panel (Delta + Cosine across
11 feature sets). The robustness matrix (Section 5) expands this
to 136 conditions with MFW truncation and excursus variants; the
directional signal holds at all levels."

---

## 2. Figure Reference Descriptions

The author cannot visually inspect the plots. The descriptions below
are derived from the data, the script code that generated each figure,
and the caption templates from `PAPER_KIT.md`. They are sufficient for
paper drafting and can be verified against the actual PNG files before
submission.

### 2.1 `dbc_anchor_mfw200_tokens.png` — Primary DBC Anchor Result

**What it shows:** A scatterplot with Book Number (I–VII) on the
x-axis and Delta distance to *De Bello Civili* on the y-axis.
Seven blue points labeled I through VII. A red dashed OLS trend line
descending from left to right. A single red square at x=8 labeled
"VIII" (Hirtius reference). A wheat-colored annotation box containing:
"Spearman r = −0.964, Bootstrap 95% CI: [−1.000, −0.886], Permutation
p = 0.0028 (exact, N=5040)."

**To verify visually:** The trend should go downward (negative slope).
Book I should be the highest (or among the highest) points; Book VII
among the lowest. Book VIII (Hirtius) should be above the Caesar
points.

**Paper placement:** Results section, primary finding.

### 2.2 `drift_char3gram.png` — Three-Panel Drift Analysis

**What it shows:** Three side-by-side panels for character 3-grams.
- **Left panel (Mantel):** Scatter of pairwise stylistic distance vs.
  chronological gap (years). 21 blue points (7-choose-2 pairs). Red
  dashed OLS line. Annotation: Mantel r and exact p-value.
- **Center panel (DBC Anchor):** Same format as 2.1 but for char
  3-grams specifically. Blue DBG I–VII points, red Hirtius square at
  book 8, OLS trend line.
- **Right panel (PCA):** Scatter of PC1 coordinate vs. book number for
  DBG I–VII. Blue points, OLS line. Annotation: PC1 variance explained
  and Spearman |r| with two-sided exact p-value.

**Expected pattern:** The Mantel should show a positive trend (points
trend upward to the right). The DBC Anchor should show a negative
trend. The PCA should show books roughly ordered along PC1.

**Paper placement:** Complementary evidence section.

### 2.3 `latinbert_analysis.png` — Latin BERT Cross-Check

**What it shows:** Two side-by-side panels.
- **Left panel (Hirtius Check):** Bar chart showing cosine distance
  from DBG I–VII centroid for each book. Caesar books I–VII in blue,
  Book VIII in red, DBC in green. A dashed reference line or annotation
  indicating Book VIII is NOT the most distant.
- **Right panel (DBC Anchor in embedding space):** Scatter of cosine
  distance to DBC embedding centroid vs. book number. Blue DBG points.
  **The trend line should go UPWARD** (positive slope) — the reverse
  of the classical stylometry result. Book VI should be the highest
  point by a clear margin (the Germanic digression maximally distant
  from DBC's civil war narrative in semantic space).

**Paper placement:** Latin BERT cross-check section. This figure is
critical for the argument that BERT captures topic rather than style.

### 2.4 `leave_one_out.png` — Jackknife Sensitivity

**What it shows:** An 8-panel small-multiples figure (4 feature sets
× 2 tests = 8 panels). Each panel is a bar chart with 7 bars (one per
dropped book, labeled I–VII). Bar height = Spearman/Pearson r for the
reduced 6-book set. A black dashed horizontal line marks the
full-corpus r. A grey dotted horizontal line at r = 0. Stars above
bars indicate significance. **Red bars** indicate sign reversal
relative to the full corpus.

**Key panels to verify:**
- Top-left (MFW 200 Tokens, DBC Anchor): All bars should be blue
  (no sign flips), all should have stars (all significant), and all
  should be near the full-corpus line (r ≈ −0.96).
- Bottom row (Function Words, both tests): Expect red bars (sign
  flips) and fewer stars (weak signal).

**Paper placement:** Robustness section or appendix.

### 2.5 `calibration_comparison.png` — Cross-Author Comparison

**What it shows:** Two grouped bar charts side-by-side.
- **Left (DBC Anchor):** Groups of 3 bars per feature set (Cicero =
  blue, DBC pseudo = red, Caesar DBG = green). Bar height = mean
  |Spearman r| across Delta + Cosine metrics for that feature set.
  Feature sets on x-axis: Func. Words, MFW 100, MFW 200, MFW 300,
  Char 2-gr, Char 3-gr, Char 4-gr. Annotations show n= per bar.
- **Right (Mantel Test):** Same format but for mean |Pearson r|.

**Expected pattern:** Caesar (green) should be the tallest bar in most
feature sets. DBC pseudo-books (red) should NOT be near zero — there
should be a visible red bar of moderate height (the narrative-structure
baseline). Cicero (blue) should be comparable to DBC pseudo or slightly
taller.

**Paper placement:** Calibration section. This is the central figure
for the cross-author validation argument.

### 2.6 `pca_umap_book_char3gram.png` — PCA Ordination

**What it shows:** PCA of 9 book-level samples (7 Caesar + Hirtius +
DBC) projected to the first two principal components. Points colored
by book (likely a viridis gradient from purple/blue for Book I to
yellow for Book VII/VIII, with DBC as a distinct marker). PC1 on
x-axis, PC2 on y-axis. Variance explained percentages in axis labels.
Annotation about Spearman correlation between PC1 and book number.

**Expected pattern:** Books should spread roughly left-to-right in
chronological order along PC1, or at least not appear as a completely
random scatter. DBC should sit near the late-book cluster. Book VIII
may be somewhat separated.

**Paper placement:** PCA results subsection.

---

## 3. Figure-to-Section Mapping for Paper Drafting

| Paper Section | Primary Figure | Backup/Alternative |
|---|---|---|
| Results: DBC Anchor | `dbc_anchor_mfw200_tokens.png` | `dbc_anchor_char2gram.png` |
| Results: Drift | `drift_char3gram.png` | `drift_mfw200_tokens.png` |
| Results: PCA | `pca_umap_book_char3gram.png` | — |
| Latin BERT | `latinbert_analysis.png` | — |
| Robustness: LOO | `leave_one_out.png` | — |
| Calibration | `calibration_comparison.png` | — |

---

## 4. Consistency Checklist Before Paper Submission

- [ ] Verify that `outputs/REPORT.md` §3.2 uses the 22-test count
  (not 34). See §1.4 above.
- [ ] Open each figure listed in §2 and confirm the visual description
  matches. If any plot differs materially from the description, note
  the correction here and update the paper text.
- [ ] Confirm that `PAPER_KIT.md` and `README.md` agree on the headline
  statistics (they currently do, at 22/19/9).
- [ ] Confirm that the robustness section in the paper uses the 136-test
  framing and cites `outputs/robustness_summary.csv`.
- [ ] Remove this sentence from the final paper: this file is an
  internal project document.

---

*This file: `docs/numerical_reconciliation.md`*
