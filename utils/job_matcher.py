# utils/job_matcher.py

import os
import json
import pickle
from BM_25 import build_or_load_bm25, match_students_to_jobs

# Module‑level caches
_JOBS = None
_BM25 = None
_JOB_INDEX = None

def startup_load(base_dir=None):
    """
    Load job data and build (or load) the BM25 model once at app startup.
    Raises on missing/malformed files so startup fails fast if something's wrong.
    """
    global _JOBS, _BM25, _JOB_INDEX

    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # 1) Load and validate JSONL job files
    job_files = ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]
    jobs = []
    for fname in job_files:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Job file missing: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    jobs.append(json.loads(line))
                except json.JSONDecodeError as je:
                    raise ValueError(f"Invalid JSON in {fname} at line {line_no}: {je}")

    _JOBS = jobs

    # 2) Build or load BM25 index & model
    _BM25, _JOB_INDEX = build_or_load_bm25(jobs, cache_dir=base_dir)
    print("✅ Jobs and BM25 model loaded in startup_load()")

def run_bm25_match(student_data):
    """
    Matches students to jobs using the preloaded jobs/BM25 model.
    Returns (matches, pickle_path).
    """
    global _JOBS, _BM25, _JOB_INDEX

    # Ensure startup_load has been called
    if _JOBS is None or _BM25 is None or _JOB_INDEX is None:
        startup_load()

    try:
        # 3) Perform the matching
        matches = match_students_to_jobs(student_data, _JOBS, _BM25, _JOB_INDEX)

        # 4) Optionally cache the results to disk for debugging
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        pkl_path = os.path.join(base_dir, "student_job_matches.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(matches, f)

        print(f"🎯 run_bm25_match: matched {len(matches)} students; results saved to {pkl_path}")
        return matches, pkl_path

    except Exception as e:
        # Log and re‑raise so you see a clear error in your logs
        print("❌ run_bm25_match failed:", str(e))
        raise
