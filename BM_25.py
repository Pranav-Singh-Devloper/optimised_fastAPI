# BM_25.py

import os
import json
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

# Ensure required NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -----------------------------
# Data Loading Utilities
# -----------------------------

def load_students(filepath='students.json'):
    """Load students from a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def load_jsonl_file(filepath):
    """Load jobs from .jsonl file"""
    with open(filepath, 'r') as file:
        return [json.loads(line.strip()) for line in file]

# -----------------------------
# Preprocessing
# -----------------------------

def preprocess_jobs(jobs):
    """
    Convert job descriptions (HTML) into token lists suitable for BM25.
    Returns:
        job_texts: List of token lists for BM25
        job_index: Original indices of jobs that produced valid tokens
    """
    job_texts = []
    job_index = []

    for idx, job in enumerate(jobs):
        title = job.get('title', '')
        tags = job.get('tagsAndSkills', '').replace(',', ' ')
        raw_description = job.get('jobDescription', '')

        # Convert HTML to plain text
        soup = BeautifulSoup(raw_description, 'html.parser')
        plain_description = soup.get_text(separator=' ', strip=True)

        combined_text = f"{title} {tags} {plain_description}".strip()
        if not combined_text:
            continue

        tokens = word_tokenize(combined_text.lower())
        tokens = [t for t in tokens if t.isalpha()]  # keep alphabetic tokens only

        if not tokens:
            continue

        job_texts.append(tokens)
        job_index.append(idx)

    if not job_texts:
        raise ValueError("❌ No valid job descriptions found.")

    return job_texts, job_index

# -----------------------------
# BM25 Core
# -----------------------------

def build_bm25_model(job_texts):
    """Build and return a BM25Okapi model trained on the given token lists."""
    return BM25Okapi(job_texts)

def build_or_load_bm25(jobs, cache_dir="."):
    """
    Check if cached BM25 model and tokenized corpus exist.
    If yes, load from disk; else, build and cache them.
    Returns:
        bm25: BM25Okapi instance
        job_index: Index of jobs
    """
    bm25_path = os.path.join(cache_dir, "bm25_model.pkl")
    corpus_path = os.path.join(cache_dir, "job_corpus.pkl")

    if os.path.exists(bm25_path) and os.path.exists(corpus_path):
        with open(bm25_path, "rb") as f1, open(corpus_path, "rb") as f2:
            bm25 = pickle.load(f1)
            job_index = pickle.load(f2)
        print("✅ Loaded BM25 model and corpus from cache.")
        return bm25, job_index

    # Preprocess and build model
    job_texts, job_index = preprocess_jobs(jobs)
    bm25 = build_bm25_model(job_texts)

    with open(bm25_path, "wb") as f1, open(corpus_path, "wb") as f2:
        pickle.dump(bm25, f1)
        pickle.dump(job_index, f2)

    print("✅ Built and cached BM25 model and corpus.")
    return bm25, job_index

# -----------------------------
# Matching
# -----------------------------

def match_students_to_jobs(students, jobs, bm25, job_index, top_n=10):
    """
    For each student, compute BM25 scores against all jobs and return structured match data.
    Returns a dictionary with student names as keys and a list of matches as values.
    """
    all_matches = {}

    for student in students:
        # Construct full name (fallback to "Unnamed" if missing)
        first_name = student.get('first_name', '')
        last_name = student.get('last_name', '')
        student_name = f"{first_name} {last_name}".strip() or "Unnamed"

        # Extract preferences, skills, interests
        job_preferences = student.get('job_preferences', {})
        job_preferences_list = []
        job_roles = []

        if isinstance(job_preferences, dict):
            for key, value in job_preferences.items():
                if isinstance(value, list):
                    if key.lower() in ['job_roles', 'job_titles']:
                        job_roles.extend(value)
                    else:
                        job_preferences_list.extend(value)
                elif isinstance(value, str):
                    if key.lower() in ['job_roles', 'job_titles']:
                        job_roles.append(value)
                    else:
                        job_preferences_list.append(value)

        skills = student.get('skills', [])
        interests = student.get('interests', [])

        # Weight job roles more heavily
        query_terms = job_roles * 5 + job_preferences_list * 2 + skills + interests
        if not query_terms:
            all_matches[student_name] = []
            continue

        # Tokenize and clean query
        query = " ".join(query_terms)
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalpha()]

        # Compute BM25 scores
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(job_index, scores), key=lambda x: x[1], reverse=True)
        top_matches = ranked[:top_n]

        student_matches = []
        for idx, score in top_matches:
            job = jobs[idx]
            company = job.get('companyName', 'Unknown Company')
            title = job.get('title', 'No Title')
            description_html = job.get('jobDescription', '')
            description_text = BeautifulSoup(description_html, 'html.parser').get_text(separator=' ', strip=True)
            snippet = description_text[:150] + ('...' if len(description_text) > 150 else '')

            student_matches.append({
                'company': company,
                'title': title,
                'score': float(score),
                'snippet': snippet
            })

        all_matches[student_name] = student_matches

    return all_matches

# -----------------------------
# Script mode (optional testing)
# -----------------------------

if __name__ == '__main__':
    students = load_students('students.json')
    jobs = []
    for part_file in ["part_1.jsonl", "part_2.jsonl", "part_3.jsonl"]:
        data = load_jsonl_file(part_file)
        jobs.extend(data)

    print(f"Total jobs loaded: {len(jobs)}")
    bm25, job_index = build_or_load_bm25(jobs)
    matches = match_students_to_jobs(students, jobs, bm25, job_index, top_n=10)

    with open('student_job_matches.pkl', 'wb') as pkl_file:
        pickle.dump(matches, pkl_file)

    print("✅ Matching results saved to 'student_job_matches.pkl'.")
