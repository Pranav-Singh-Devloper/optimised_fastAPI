from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client
from utils.job_matcher import run_bm25_match
from utils.chatbot_runner import analyze_matches
from mangum import Mangum
from BM_25 import load_jobs_from_mongo, build_or_load_bm25
import logging

# Load .env variables
load_dotenv()

# Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI app instance
app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache variables
bm25 = None
job_index = None
jobs = None

# Request body schema
class ProfileRequest(BaseModel):
    intern_name: str
    students: List[Dict[str, Any]]
    interests: str

@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI app is live on Railway!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/match")
def match_students(request: ProfileRequest):
    global bm25, job_index, jobs

    try:
        print("✅ Received POST request to /match")
        print("👨‍🎓 Students received:", len(request.students))

        # Add shared interests to each student's job_preferences
        for student in request.students:
            student.setdefault("job_preferences", {})["interests"] = [
                x.strip() for x in request.interests.split("+")
            ]

        # Lazy-load BM25 and jobs only when first request hits
        if bm25 is None or job_index is None or jobs is None:
            print("🕒 BM25 not loaded yet — loading from MongoDB and building model...")
            jobs = load_jobs_from_mongo()
            bm25, job_index = build_or_load_bm25(jobs)
            print("✅ BM25 model built and cached.")
        else:
            print("✅ BM25 model already loaded in memory.")

        # Match jobs
        matches, pickle_path = run_bm25_match(request.students)
        print("🎯 BM25 Matching done. Pickle at:", pickle_path)

        # Analyze matches using LLM
        analysis = analyze_matches(pickle_path, request.students)
        print("🧠 LLM Analysis generated.")

        # Upload result to Supabase
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "intern_name": request.intern_name,
            "student_profile": json.dumps(request.students),
            "bm25_matches": matches,
            "llm_analysis": analysis
        }

        print("📤 Uploading to Supabase...")
        supabase.table("v0001_logs").insert(payload).execute()
        print("✅ Upload complete!")

        return {"success": True, "llm_analysis": analysis}

    except Exception as e:
        print("❌ ERROR OCCURRED:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# AWS Lambda support via Mangum (can be ignored for Railway)
handler = Mangum(app)
