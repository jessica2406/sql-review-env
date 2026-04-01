from fastapi import FastAPI, HTTPException
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Action, Observation, State
from server.environment import SQLReviewEnvironment
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import json

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for HF Spaces compatibility."""
    await websocket.accept()
    # Fresh environment instance per connection
    ws_env = SQLReviewEnvironment()
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            
            if msg_type == "reset":
                result = ws_env.reset()
                await websocket.send_json(result.model_dump())
                
            elif msg_type == "step":
                action = Action(**data.get("action", {}))
                result = ws_env.step(action)
                await websocket.send_json(result.model_dump())
                
            elif msg_type == "state":
                result = ws_env.state()
                await websocket.send_json(result.model_dump())
                
            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})
                
    except WebSocketDisconnect:
        pass

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()