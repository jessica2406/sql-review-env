import os
import requests
import json
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG — exactly as required by checklist
# ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — required by checklist

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

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def run_episode() -> dict:
    """Run one full episode of the environment with the LLM agent."""
    scores = {}

    # Reset environment
    res = requests.post(f"{API_BASE_URL}/reset")
    res.raise_for_status()
    obs = res.json()

    if "done" not in obs:
        raise ValueError(f"Unexpected response from /reset: {obs}")

    # START log — required structured format
    print("START")

    max_steps = 10
    step = 0

    while not obs["done"] and step < max_steps:
        task_id = obs["task_id"]

        # STEP log — required structured format
        print(f"STEP task_id={task_id} step={step+1}")

        try:
            action = ask_llm(
                buggy_query=obs["buggy_query"],
                table_schema=obs["table_schema"],
                expected_output=obs["expected_output"]
            )
        except Exception as e:
            print(f"STEP llm_error={str(e)}")
            action = {
                "fixed_query": obs["buggy_query"],
                "explanation": "Could not parse LLM response"
            }

        # Submit to environment
        res = requests.post(f"{API_BASE_URL}/step", json=action)
        res.raise_for_status()
        obs = res.json()

        reward = obs["reward"]
        scores[task_id] = reward

        print(f"STEP reward={reward} done={obs['done']}")
        step += 1

    # Get final state
    state_res = requests.get(f"{API_BASE_URL}/state")
    state = state_res.json()

    # END log — required structured format
    print("END")

    print(f"total_score={state['total_score']}")
    for task_id, score in scores.items():
        print(f"score task_id={task_id} score={score}")

    return scores


if __name__ == "__main__":
    scores = run_episode()
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"baseline_score={avg:.2f}")