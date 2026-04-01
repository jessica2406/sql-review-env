from fastapi import FastAPI, HTTPException
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State
from server.environment import SQLReviewEnvironment

app = FastAPI(
    title="SQL Review Environment",
    description="An OpenEnv-compatible environment where an AI agent learns to find and fix SQL bugs.",
    version="1.0.0"
)

# Single environment instance
env = SQLReviewEnvironment()


@app.post("/reset", response_model=Observation)
def reset():
    """Start a new episode. Always call this first."""
    return env.reset()


@app.post("/step", response_model=Observation)
def step(action: Action):
    """Submit a fixed SQL query. Returns observation with reward."""
    try:
        return env.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def state():
    """Get current episode metadata."""
    try:
        return env.state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    """Health check — used by Hugging Face to verify the Space is alive."""
    return {"status": "ok"}