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

port = int(os.environ.get("PORT", 8000))

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProfileRequest(BaseModel):
    intern_name: str
    students: List[Dict[str, Any]]
    interests: str

@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI app is live on Render!"}


@app.post("/match")
def match_students(request: ProfileRequest):
    try:
        # Add interests
        for student in request.students:
            student.setdefault("job_preferences", {})["interests"] = [
                x.strip() for x in request.interests.split("+")
            ]

        # Match jobs
        matches, pickle_path = run_bm25_match(request.students)

        # LLM reasoning
        analysis = analyze_matches(pickle_path, request.students)

        # Upload to Supabase
        supabase.table("v0001_logs").insert({
            "timestamp": datetime.utcnow().isoformat(),
            "intern_name": request.intern_name,
            "student_profile": request.students,
            "bm25_matches": matches,
            "llm_analysis": analysis
        }).execute()

        return {"success": True, "llm_analysis": analysis}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
