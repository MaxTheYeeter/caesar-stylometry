# Caesar Stylometry

## Composition Chronology of Julius Caesar's *De Bello Gallico*

A computational stylometry project in the digital humanities investigating whether Julius Caesar composed Books I–VII of *De Bello Gallico* (*The Gallic War*) **annually** across the campaigns (58–52 BC) or in **bulk** near the war's end (~51–50 BC).

## Table of Contents

1. [Background](#background)
2. [The Two Hypotheses](#the-two-hypotheses)
3. [Built-In Controls](#built-in-controls)
4. [Results Summary](#results-summary)
5. [Repository Structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Installation & Setup](#installation--setup)
8. [Running the Pipeline](#running-the-pipeline)
9. [Methodology](#methodology)
10. [Outputs](#outputs)
11. [Caveats & Threats to Validity](#caveats--threats-to-validity)
12. [Citation & Contact](#citation--contact)

## Background

Julius Caesar's *De Bello Gallico* (DBG) is an eight-book Latin prose account of his military campaigns in Gaul (58–50 BC). Books I–VII are attributed to Caesar; Book VIII was written by his lieutenant Aulus Hirtius. Caesar also wrote *De Bello Civili* (DBC, *The Civil War*), covering 49–48 BC.

Classical scholarship has long debated **when** Caesar composed Books I–VII:

* Did he write each book shortly after its campaign year, as a serial political newsletter to Rome?
* Or did he compose most or all seven books together near the end of the war, as a unified retrospective narrative?

Traditional philological arguments exist for both positions. This project brings quantitative, reproducible computational stylometry to the question.

## The Two Hypotheses

### Hypothesis A — Annual / Serial Composition

Each book was written close in time to the events it describes.

| Book | Events                                              | Presumed Composition |
| ---- | --------------------------------------------------- | -------------------- |
| I    | Helvetian campaign                                  | ~58 BC               |
| II   | Belgic campaign                                     | ~57 BC               |
| III  | Alpine / Veneti campaign                            | ~56 BC               |
| IV   | German campaign, first Britain expedition           | ~55 BC               |
| V    | Second Britain expedition, revolts                  | ~54 BC               |
| VI   | Treveri / German campaigns, ethnographic digression | ~53 BC               |
| VII  | Vercingetorix revolt, siege of Alesia               | ~52 BC               |

**Prediction:** Writing style drifts gradually and **directionally** across the books. Early books should differ systematically from late books. Later books should sit stylistically **closer** to Caesar's later work *De Bello Civili* (49–48 BC) than early books do.

### Hypothesis B — Bulk / Single-Period Composition

Most or all of Books I–VII were written together near the war's end (~51–50 BC) as a single retrospective narrative.

**Prediction:** Books I–VII should be stylistically **uniform** with no directional trend. All seven books should be roughly **equidistant** from *De Bello Civili*.

## Built-In Controls

Two controls provide methodological grounding.

### 1. Hirtius Book VIII (Ground-Truth Gate)

Book VIII of DBG was written by Aulus Hirtius, **not Caesar**. Any valid stylometric method must detect that Book VIII is stylistically separable from Books I–VII. This gate must be passed before any chronological inference within Caesar's own books is credible.

**Result: PASSED.** Hirtius Book VIII has the highest outlier rate in one-class SVM (25–34%, vs. 11–14% for Caesar I–VII chapters). R/stylo Delta separates Hirtius at MFW ≤100 (ratio ~1.16). The method reliably distinguishes Caesar from a known non-Caesarian author.

### 2. De Bello Civili (Dated Stylistic Anchor)

DBC is genuine Caesar, written later than any DBG book (49–48 BC). It serves as a fixed temporal reference point: if Caesar's style drifted over time, DBC should be closer to later DBG books than to early ones.

## Results Summary

| Analysis             | Test                                     | Direction                     | Significance                             |
| -------------------- | ---------------------------------------- | ----------------------------- | ---------------------------------------- |
| **DBC Anchor**       | Spearman r: book order ~ distance to DBC | **22/22** negative            | 19/22 p < 0.05                           |
| **Mantel Test**      | Stylistic distance ~ year gap            | **22/22** positive            | 9/22 p < 0.05                            |
| **PCA PC1**          | Unsupervised ordination ~ book order     | 4/11 p < 0.05                 | Best: char 3-grams r = +0.893, p = 0.007 |
| **Robustness**       | 136 condition combinations               | **136/136** correct direction | 80.9% DBC anchor significant             |
| **Excursus Removal** | Without Book VI Germanic digression      | Mean DBC r: −0.841 → −0.777   | No structural change                     |
| **Latin BERT**       | Contextual embedding cross-check         | Reversed (topic confound)     | Confirms style methods                   |

**Verdict:** The evidence **supports Hypothesis A (Annual Composition).** Stylistic distance between DBG books correlates with their chronological separation; later books are consistently closer to Caesar's later work *De Bello Civili*; and the finding is robust to the removal of disputed passages, feature representation choice, and lexical processing level.

Latin BERT embeddings produce the **opposite** direction—driven by semantic/topic similarity (Book VI's Germanic ethnography is maximally distant from DBC's civil war narrative). This disagreement strengthens the classical stylometry results by confirming they capture authorial **style** rather than topic content.

**Full report with hedging, statistics, and methodological discussion:** `outputs/REPORT.md`

## Repository Structure

```text
caesar_stylometry/
├── README.md                     ← This file
├── requirements.txt              ← Python dependencies
├── install_r_packages.R          ← R dependencies
├── run_all.sh                    ← Run full pipeline
├── .gitignore                    ← Git exclusions
│
├── data/
│   ├── raw/perseus/              ← Place Perseus XML here (not committed)
│   ├── corpus/                   ← Generated CSVs (not committed)
│   ├── processed/                ← Intermediate artifacts
│   └── stylo_corpus_*/           ← Plain-text for R/stylo (generated)
│
├── scripts/                      ← All source code (01–17)
│   ├── 01_parse_perseus_xml.py
│   ├── 02_parse_tei.py
│   ├── 03_build_corpus.py
│   ├── 04_normalize.py
│   ├── 05_lemmatize.py
│   ├── 06_diagnostics.py
│   ├── 07_features_words.py
│   ├── 08_features_ngrams.py
│   ├── 09_export_for_stylo.py
│   ├── 10_stylo_delta.R
│   ├── 11_delta_python.py
│   ├── 12_pca_umap.py
│   ├── 13_dbc_anchor.py
│   ├── 14_drift_tests.py
│   ├── 15_robustness.py
│   ├── 16_latinbert.py
│   ├── 17_report.py
│   └── run_analysis_pipeline.py
│
├── outputs/                      ← Generated tables and reports
├── figures/                      ← Generated plots
└── logs/                         ← Run logs
```

## Prerequisites

### Required Software

| Component         | Version Tested | Purpose                                              |
| ----------------- | -------------- | ---------------------------------------------------- |
| Python            | 3.13           | All corpus, feature, and analysis scripts            |
| R                 | 4.6.0          | `stylo` classical Delta analysis                     |
| R package `stylo` | 0.7.7          | Burrows Delta, clustering, bootstrap consensus trees |
| Git               | Any            | Version control                                      |

### Required Data

Two TEI XML files from the Perseus Digital Library:

* `caes_bg_lat.xml` — *De Bello Gallico* (all eight books)
* `caes.bc_lat.xml` — *De Bello Civili*

**Download:** Perseus Digital Library or the Perseus GitHub repository. Place both files in:

```text
data/raw/perseus/
```

### Python Packages

All packages are specified in `requirements.txt`. Key dependencies:

* `cltk==2.5.1`
* `lxml`
* `beautifulsoup4`
* `numpy`
* `scipy`
* `pandas`
* `scikit-learn`
* `umap-learn`
* `matplotlib`
* `seaborn`
* `transformers`
* `torch`
* `requests`

### R Packages

* `stylo` (0.7.7)
* `ggplot2`
* `reshape2`
* `cluster`
* `ape`

## Installation & Setup

### 1. Clone the Repository

```bash
git clone git@github.com:MaxTheYeeter/caesar-stylometry.git
cd caesar-stylometry
```

### 2. Place Perseus XML Files

```text
data/raw/perseus/caes_bg_lat.xml
data/raw/perseus/caes.bc_lat.xml
```

### 3. Create a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Install R Packages

```bash
Rscript install_r_packages.R
```

### 5. Verify Setup

```bash
python -c "import cltk; print('CLTK', cltk.__version__)"
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"

Rscript -e 'packageVersion("stylo")'

ls data/raw/perseus/caes_bg_lat.xml data/raw/perseus/caes.bc_lat.xml
```

## Running the Pipeline

### Full Pipeline

```bash
bash run_all.sh
```

This executes scripts 01–17 in order:

1. **Corpus construction** (01–05)
2. **Feature extraction** (06–09)
3. **Validation** (10–11)
4. **Analysis** (12–14)
5. **Robustness** (15)
6. **Cross-check** (16)
7. **Report generation** (17)

Total runtime: ~50–60 minutes (CPU-bound; no GPU required).

### Start from a Specific Script

```bash
bash run_all.sh --from 10
bash run_all.sh --from 12
```

### Run Individual Scripts

```bash
python scripts/14_drift_tests.py
Rscript scripts/10_stylo_delta.R
```

### Re-run Only the Analysis Phase

```bash
python scripts/run_analysis_pipeline.py
python scripts/run_analysis_pipeline.py --from 14
python scripts/17_report.py
```

## Methodology

### Feature Representations

| Family                        | Variants               | Captures                                     |
| ----------------------------- | ---------------------- | -------------------------------------------- |
| **Function words**            | 299 tokens, 166 lemmas | Grammatical fingerprint (topic-independent)  |
| **Most-Frequent Words (MFW)** | 100, 200, 300 cutoffs  | Grammatical + high-frequency lexical         |
| **Character n-grams**         | 2-gram, 3-gram, 4-gram | Sub-word morphology (case endings, suffixes) |

### Lexical Processing

All analyses run on both:

* **Normalized tokens** (lowercased, j→i, v→u)
* **Lemmas** (CLTK 2.5.1 + Stanza)

Agreement between them strengthens conclusions; disagreement flags lemmatization sensitivity.

### Distance Metrics

* **Burrows's Delta** (classical)
* **Cosine distance** (complementary)

### Statistical Framework (Small-n Safe)

With only **n = 7** Caesarian books, parametric assumptions are untenable.

All significance testing uses:

* Exact permutation tests (7! = 5,040 permutations)
* Bootstrap 95% confidence intervals (10,000 resamples)
* One-sided tests where directional predictions exist
* Two-sided tests for exploratory analyses

### Robustness Conditions (Script 15)

All analyses are re-run across:

* 4 excursus conditions
* 2 lexical levels
* 2 representation families
* Multiple MFW cutoffs
* 2 distance metrics

**Total:** 136 unique condition combinations.

### Latin BERT (Script 16)

Contextual embeddings from `LuisAVasquez/simple-latin-bert-uncased` (105M parameters, 25K vocabulary, BERT-base architecture trained on Latin).

Used only as a **relative cross-check**, not as an independent authorship claim.

## Outputs

### Primary Output

* `outputs/REPORT.md` — Complete analysis report with all tables, statistics, figures, hedging, and methodological discussion.

### Key Tables

| File                                  | Content                            |
| ------------------------------------- | ---------------------------------- |
| `outputs/robustness_summary.csv`      | 136 rows × condition results       |
| `outputs/corpus_diagnostics.csv`      | Per-book token/type/chapter counts |
| `outputs/delta_python_svm_report.csv` | One-class SVM outlier rates        |
| `outputs/book8_separation_report.csv` | R/stylo Hirtius separation ratios  |
| `outputs/RESULTS_MANIFEST.md`         | Quick-reference results summary    |

### Feature & Distance Matrices

* `outputs/features_*_books.csv` (11 files)
* `outputs/features_*_chapters.csv` (11 files)
* `outputs/delta_distance_*.csv` (48 files)
* `outputs/delta_python_distance_*.csv` (4 files)

### Figures

| Analysis       | Representative Figure                  |
| -------------- | -------------------------------------- |
| DBC Anchor     | `figures/dbc_anchor_mfw200_tokens.png` |
| Drift          | `figures/drift_char3gram.png`          |
| Latin BERT     | `figures/latinbert_analysis.png`       |
| PCA Book-Level | `figures/pca_umap_book_char3gram.png`  |

## Caveats & Threats to Validity

### Internal Validity

| Threat                | Severity | Mitigation                                      |
| --------------------- | -------- | ----------------------------------------------- |
| Small sample (n = 7)  | High     | Exact permutation tests; bootstrap CIs          |
| Topic-time confound   | Medium   | Function words; Latin BERT cross-check          |
| Book length variation | Medium   | Proportion-based features; weighted aggregation |
| Lemmatization errors  | Low      | Results consistent across tokens and lemmas     |
| Disputed passages     | Low      | Excursus-removal robustness tests               |
| Multiple testing      | Low      | Directional unanimity across conditions         |

### External Validity

| Threat                    | Severity | Mitigation                              |
| ------------------------- | -------- | --------------------------------------- |
| Single late-Caesar anchor | Medium   | DBC is the only comparable extant prose |
| Genre confound            | Low      | All texts are *commentarii*             |
| Generalizability          | High     | Findings specific to Caesar             |

### Construct Validity

Stylometric distance measures **stylistic similarity**, which may correlate with composition date but is not a direct temporal measurement. The results constitute **quantitative evidence** that should be weighed alongside traditional philological, historical, and biographical scholarship—not a standalone proof in the historical sense.

## Citation & Contact

This is a digital humanities research project.

The full analysis report with methodological discussion is available at:

`outputs/REPORT.md`

**Repository:** `github.com/MaxTheYeeter/caesar-stylometry`

### Key References

* Burrows, J. (2002). *Delta: a measure of stylistic difference and a guide to likely authorship*. *Literary and Linguistic Computing*, 17(3).
* Eder, M., Rybicki, J., & Kestemont, M. (2016). *Stylometry with R: a package for computational text analysis*. *R Journal*, 8(1).
* Mantel, N. (1967). *The detection of disease clustering and a generalized regression approach*. *Cancer Research*, 27(2).

*Last updated: 2026-06-15*
