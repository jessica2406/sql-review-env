import os
import requests
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG — reads from environment variables
# ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

SYSTEM_PROMPT = """You are an expert SQL developer. 
You will be given a buggy SQL query and the table schema.
Your job is to find the bug and return the fixed query.

You must respond in this exact JSON format:
{
  "fixed_query": "YOUR FIXED SQL HERE",
  "explanation": "Brief explanation of what was wrong"
}

Return only the JSON. No markdown, no extra text."""


def ask_llm(buggy_query: str, table_schema: str, expected_output: str) -> dict:
    """Send the task to the LLM and get back a fixed query."""
    user_message = f"""Fix this SQL query:

SCHEMA:
{table_schema}

BUGGY QUERY:
{buggy_query}

EXPECTED OUTPUT FORMAT:
{expected_output}

Return only JSON with fixed_query and explanation."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    import json
    text = response.choices[0].message.content.strip()
    # Strip markdown code blocks if model adds them
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def run_episode() -> dict:
    """Run one full episode of the environment with the LLM agent."""
    scores = {}

    # Reset environment
    res = requests.post(f"{API_BASE_URL}/reset")
    res.raise_for_status()
    obs = res.json()
    print(f"Reset response: {obs}")  # Debug line
    
    # Safety check
    if "done" not in obs:
        raise ValueError(f"Unexpected response from /reset: {obs}")

    print(f"\n{'='*60}")    
    print("EPISODE START")
    print(f"{'='*60}")

    max_steps = 10
    step = 0

    while not obs["done"] and step < max_steps:
        task_id = obs["task_id"]
        print(f"\n[Step {step+1}] Task: {task_id}")
        print(f"Buggy query: {obs['buggy_query'].strip()}")

        # Ask LLM to fix the query
        try:
            action = ask_llm(
                buggy_query=obs["buggy_query"],
                table_schema=obs["table_schema"],
                expected_output=obs["expected_output"]
            )
        except Exception as e:
            print(f"LLM error: {e}")
            action = {
                "fixed_query": obs["buggy_query"],
                "explanation": "Could not parse LLM response"
            }

        print(f"LLM fix: {action['fixed_query'].strip()}")
        print(f"Explanation: {action['explanation']}")

        # Submit to environment
        res = requests.post(f"{API_BASE_URL}/step", json=action)
        obs = res.json()

        reward = obs["reward"]
        scores[task_id] = reward
        print(f"Reward: {reward}")
        print(f"Feedback: {obs['feedback']}")

        step += 1

    # Final state
    state_res = requests.get(f"{API_BASE_URL}/state")
    state = state_res.json()

    print(f"\n{'='*60}")
    print("EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {state['step_count']}")
    print(f"Total score: {state['total_score']}")
    print(f"\nScores per task:")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score}")

    return scores


if __name__ == "__main__":
    print("SQL Review Environment — Baseline Inference")
    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")

    # Summary
    try:
        scores = run_episode()
        if scores:
            avg = sum(scores.values()) / len(scores)
            print(f"\nBaseline Score: {avg:.2f} / 1.0")
    except Exception as e:
        print(f"Error during episode: {e}")
        raise