from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- ACTION ---
# This is what the AI agent SENDS to the environment
class Action(BaseModel):
    fixed_query: str          # The agent's corrected SQL query
    explanation: str          # Why the agent thinks this fixes the bug

# --- OBSERVATION ---
# This is what the environment SENDS BACK to the agent
class Observation(BaseModel):
    task_id: str              # Which task we're on (e.g. "easy_1")
    buggy_query: str          # The broken SQL query to fix
    table_schema: str             # The table structure (so agent knows columns)
    expected_output: str      # What the correct result should look like
    reward: float             # Score for last action (0.0 to 1.0)
    done: bool                # Is the episode over?
    feedback: str             # Human-readable feedback on last attempt
    metadata: Dict[str, Any] = {}

# --- STATE ---
# Internal episode tracking
class State(BaseModel):
    episode_id: str
    step_count: int
    current_task_id: str
    total_score: float
    max_steps: int = 5