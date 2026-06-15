#!/usr/bin/env bash
# run_all.sh — Caesar Stylometry full pipeline
# Usage:
#   bash run_all.sh            # run everything (scripts 01–17)
#   bash run_all.sh --from 10  # skip corpus build, start at analysis phase
#
# Prerequisites:
#   - Python venv created and requirements installed
#   - R installed with stylo package
#   - Perseus XML files in data/raw/perseus/
#   - Latin BERT model accessible (auto-downloads on first run of script 16)

set -e  # Stop on first error
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
START_AT=1

# ── Parse arguments ───────────────────────────────────────────────────
if [ "$1" = "--from" ] && [ -n "$2" ]; then
    START_AT="$2"
    echo "Starting from script $START_AT"
fi

# ── Activate venv ──────────────────────────────────────────────────────
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "✓ venv activated"
else
    echo "✗ venv not found at $PROJECT_ROOT/venv/"
    echo "  Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# ── Timestamp ──────────────────────────────────────────────────────────
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs/pipeline_$RUN_ID"
mkdir -p "$LOG_DIR"
echo "Run ID: $RUN_ID"
echo "Logs:   $LOG_DIR"
echo ""

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════════════
# CORPUS PIPELINE (Python scripts 01–09)
# ═══════════════════════════════════════════════════════════════════════
run_python_script() {
    local num="$1"
    local script="$2"
    local label="$3"
    local log_file="$LOG_DIR/${script%.py}.log"

    if [ "$num" -lt "$START_AT" ]; then
        echo "⏭  [$num/17] $label — SKIPPED"
        return 0
    fi

    echo "▶  [$num/17] $label"
    echo "    Script: $script"

    local t0
    t0=$(date +%s)

    python "$SCRIPTS_DIR/$script" > "$log_file" 2>&1
    local exit_code=$?

    local t1
    t1=$(date +%s)
    local elapsed=$((t1 - t0))

    if [ "$exit_code" -eq 0 ]; then
        echo "    ✓ Done (${elapsed}s) — log: $log_file"
    else
        echo "    ✗ FAILED (exit code $exit_code, ${elapsed}s)"
        echo "    Log: $log_file"
        tail -30 "$log_file"
        exit "$exit_code"
    fi
    echo ""
}

echo "═══════════════════════════════════════════"
echo "  PHASE 1: Corpus Pipeline (01–09)"
echo "═══════════════════════════════════════════"
echo ""

run_python_script 1  "01_parse_perseus_xml.py"       "Parse Perseus TEI XML"
run_python_script 2  "02_parse_tei.py"                "Parse TEI structure"
run_python_script 3  "03_build_corpus.py"             "Build chapter/book CSVs"
run_python_script 4  "04_normalize.py"                "Orthographic normalization"
run_python_script 5  "05_lemmatize.py"                "Lemmatization (CLTK + Stanza)"
run_python_script 6  "06_diagnostics.py"              "Corpus diagnostics"
run_python_script 7  "07_features_words.py"           "Word feature matrices"
run_python_script 8  "08_features_ngrams.py"          "Character n-gram matrices"
run_python_script 9  "09_export_for_stylo.py"         "Export plain-text for R/stylo"

# ═══════════════════════════════════════════════════════════════════════
# VALIDATION (R script 10)
# ═══════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  PHASE 2: Validation (10–11)"
echo "═══════════════════════════════════════════"
echo ""

R_SCRIPT="10_stylo_delta.R"
if [ 10 -ge "$START_AT" ]; then
    echo "▶  [10/17] R/stylo Delta Analysis"
    echo "    Script: $R_SCRIPT"
    t0=$(date +%s)
    Rscript "$SCRIPTS_DIR/$R_SCRIPT" > "$LOG_DIR/${R_SCRIPT%.R}.log" 2>&1
    exit_code=$?
    t1=$(date +%s)
    elapsed=$((t1 - t0))
    if [ "$exit_code" -eq 0 ]; then
        echo "    ✓ Done (${elapsed}s)"
    else
        echo "    ✗ FAILED (exit code $exit_code, ${elapsed}s)"
        echo "    Log: $LOG_DIR/${R_SCRIPT%.R}.log"
        tail -30 "$LOG_DIR/${R_SCRIPT%.R}.log"
        exit "$exit_code"
    fi
else
    echo "⏭  [10/17] R/stylo Delta — SKIPPED"
fi
echo ""

run_python_script 11 "11_delta_python.py"             "Python Burrows Delta + SVM"

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  PHASE 3: Analysis (12–16)"
echo "═══════════════════════════════════════════"
echo ""

run_python_script 12 "12_pca_umap.py"                  "PCA & UMAP visualization"
run_python_script 13 "13_dbc_anchor.py"                "DBC Anchor test"
run_python_script 14 "14_drift_tests.py"               "Directional drift (Mantel + PCA)"
run_python_script 15 "15_robustness.py"                "Robustness checks"
run_python_script 16 "16_latinbert.py"                 "Latin BERT cross-check"

# ═══════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════
echo "═══════════════════════════════════════════"
echo "  PHASE 4: Report"
echo "═══════════════════════════════════════════"
echo ""

run_python_script 17 "17_report.py"                    "Generate final report"

# ═══════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
MINUTES=$((TOTAL / 60))
SECONDS=$((TOTAL % 60))

echo "═══════════════════════════════════════════"
echo "  PIPELINE COMPLETE"
echo "═══════════════════════════════════════════"
echo ""
echo "  Total time: ${MINUTES}m ${SECONDS}s"
echo "  Logs:       $LOG_DIR/"
echo "  Report:     outputs/REPORT.md"
echo ""
echo "  Next:  git add -A && git commit -m 'Full analysis run' && git push"
echo ""
