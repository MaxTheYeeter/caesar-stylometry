#!/usr/bin/env python3
"""
scripts/run_analysis_pipeline.py

Master runner for the stylometric analysis phase (scripts 12–16).

Usage:
    python scripts/run_analysis_pipeline.py [--from N]

Runs each script as a subprocess, captures full stdout/stderr to timestamped
log files, records wall-clock time, and checks exit codes. Produces a
run summary at the end.

All log files go to logs/analysis_run_YYYYMMDD_HHMMSS/
"""

import subprocess
import sys
import os
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

# ── Script registry (in dependency order) ──────────────────────────────
SCRIPTS = [
    {
        'script': '12_pca_umap.py',
        'label':  'PCA & UMAP Visualization',
        'expected_runtime': '~2 min',
    },
    {
        'script': '13_dbc_anchor.py',
        'label':  'DBC Anchor Test',
        'expected_runtime': '~3 min',
    },
    {
        'script': '14_drift_tests.py',
        'label':  'Directional Drift Tests (Mantel + DBC + PCA)',
        'expected_runtime': '~5 min',
    },
    {
        'script': '15_robustness.py',
        'label':  'Robustness Checks (Excursus + Representation)',
        'expected_runtime': '~15–20 min',
    },
    {
        'script': '16_latinbert.py',
        'label':  'Latin BERT Embedding Cross-Check',
        'expected_runtime': '~2–4 min (after model cached)',
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Run stylometric analysis pipeline (scripts 12–16)')
    parser.add_argument('--from', dest='start_at', type=int, default=12,
                        help='First script number to run (default: 12)')
    args = parser.parse_args()

    # ── Set up log directory ───────────────────────────────────────
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(PROJECT_ROOT, 'logs', f'analysis_run_{run_id}')
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print("  ANALYSIS PIPELINE RUNNER")
    print(f"  Run ID:  {run_id}")
    print(f"  Log dir: {log_dir}")
    print("=" * 70)
    print()

    results = []
    pipeline_start = time.time()

    for entry in SCRIPTS:
        script_name = entry['script']
        script_num = int(script_name.split('_')[0])
        if script_num < args.start_at:
            print(f"  ⏭  Skipping script {script_num} ({entry['label']})")
            continue

        script_path = os.path.join(SCRIPTS_DIR, script_name)
        if not os.path.exists(script_path):
            print(f"  ⚠  Script not found: {script_path}")
            results.append({
                'script': script_name, 'label': entry['label'],
                'status': 'MISSING', 'duration': 0, 'log': None,
            })
            continue

        print(f"  ▶  Script {script_num}: {entry['label']}")
        print(f"     Expected: {entry['expected_runtime']}")

        log_stem = f"{script_num}_{script_name.replace('.py', '')}"
        stdout_log = os.path.join(log_dir, f"{log_stem}_stdout.log")
        stderr_log = os.path.join(log_dir, f"{log_stem}_stderr.log")

        t0 = time.time()

        with open(stdout_log, 'w') as out_f, open(stderr_log, 'w') as err_f:
            out_f.write(f"# Script: {script_name}\n")
            out_f.write(f"# Started: {datetime.now().isoformat()}\n")
            out_f.write(f"# {'─' * 60}\n\n")
            err_f.write(f"# Script: {script_name}\n")
            err_f.write(f"# Started: {datetime.now().isoformat()}\n")
            err_f.write(f"# {'─' * 60}\n\n")

            proc = subprocess.Popen(
                [sys.executable, script_path],
                cwd=PROJECT_ROOT,
                stdout=out_f,
                stderr=err_f,
                text=True,
            )
            proc.wait()

        elapsed = time.time() - t0
        status = 'OK' if proc.returncode == 0 else f'FAILED (code {proc.returncode})'

        status_icon = '✓' if proc.returncode == 0 else '✗'
        print(f"     {status_icon}  {status}  [{elapsed:.1f}s]")
        print(f"        stdout: {stdout_log}")
        print(f"        stderr: {stderr_log}")
        print()

        results.append({
            'script': script_name, 'label': entry['label'],
            'status': status, 'duration': elapsed, 'log': stdout_log,
        })

        if proc.returncode != 0:
            print(f"     ⚠  Script {script_num} FAILED. "
                  f"Check {stderr_log} for details.")

    pipeline_elapsed = time.time() - pipeline_start

    # ── Run Summary ─────────────────────────────────────────────────
    summary_path = os.path.join(log_dir, 'RUN_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  ANALYSIS PIPELINE RUN SUMMARY\n")
        f.write(f"  Run ID: {run_id}\n")
        f.write(f"  Completed: {datetime.now().isoformat()}\n")
        f.write(f"  Total wall time: {pipeline_elapsed:.1f}s "
                f"({pipeline_elapsed / 60:.1f} min)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Script':<25s} {'Status':<20s} {'Duration':>10s}\n")
        f.write(f"{'─' * 55}\n")
        for r in results:
            f.write(f"{r['script']:<25s} {r['status']:<20s} "
                    f"{r['duration']:>9.1f}s\n")
        f.write(f"{'─' * 55}\n")
        n_ok = sum(1 for r in results if r['status'] == 'OK')
        n_fail = sum(1 for r in results if 'FAILED' in r['status'])
        n_missing = sum(1 for r in results if r['status'] == 'MISSING')
        f.write(f"\n  Passed:  {n_ok}\n")
        f.write(f"  Failed:  {n_fail}\n")
        f.write(f"  Missing: {n_missing}\n")
        f.write(f"\n  All logs: {log_dir}/\n")

    # ── Console Summary ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RUN COMPLETE")
    print(f"  Total time: {pipeline_elapsed:.1f}s "
          f"({pipeline_elapsed / 60:.1f} min)")
    print("=" * 70)
    print()
    for r in results:
        icon = '✓' if r['status'] == 'OK' else '✗'
        print(f"  {icon}  {r['script']:<25s}  {r['status']:<20s}  "
              f"{r['duration']:.1f}s")
    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_total = len(results)
    n_fail = sum(1 for r in results if 'FAILED' in r['status'])
    print(f"\n  {n_ok}/{n_total} scripts passed")
    print(f"  Full logs: {log_dir}/")
    print(f"  Summary:   {summary_path}")
    print()
    if n_fail > 0 or n_missing > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
