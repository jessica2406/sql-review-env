import os
import json
import requests
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG — exactly as required
# ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
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

TASKS = [
    {
        "task_id": "easy_1",
        "description": "Fix a case-sensitivity bug in a WHERE clause"
    },
    {
        "task_id": "medium_1",
        "description": "Fix a missing GROUP BY clause"
    },
    {
        "task_id": "hard_1",
        "description": "Fix an incorrect JOIN type"
    }
]


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


def run_task(task_id: str) -> dict:
    """Run one task as a full episode."""
    rewards = []
    step = 0
    max_steps = 5
    success = False
    score = 0.0
    last_error = "null"

    # Reset environment
    res = requests.post(f"{API_BASE_URL}/reset")
    res.raise_for_status()
    obs = res.json()

    # Print START line — exact required format
    print(f"[START] task={task_id} env=sql-review-env model={MODEL_NAME}")

    try:
        while not obs["done"] and step < max_steps:
            step += 1
            last_error = "null"

            # Get LLM action
            try:
                action = ask_llm(
                    buggy_query=obs["buggy_query"],
                    table_schema=obs["table_schema"],
                    expected_output=obs["expected_output"]
                )
                action_str = action["fixed_query"].replace("\n", " ").strip()
            except Exception as e:
                last_error = str(e).replace("\n", " ")
                action = {
                    "fixed_query": obs["buggy_query"],
                    "explanation": "LLM error"
                }
                action_str = obs["buggy_query"].replace("\n", " ").strip()

            # Submit to environment
            res = requests.post(f"{API_BASE_URL}/step", json=action)
            res.raise_for_status()
            obs = res.json()

            reward = float(obs["reward"])
            rewards.append(reward)
            done = str(obs["done"]).lower()

            # Print STEP line — exact required format
            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done} error={last_error}")

            if obs["done"] or reward == 1.0:
                score = reward
                success = reward == 1.0
                break

        if rewards:
            score = max(rewards)
            success = score == 1.0

    except Exception as e:
        last_error = str(e).replace("\n", " ")
        if not rewards:
            rewards = [0.0]

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        # Print END line — exact required format
        print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={rewards_str}")

    return {"task_id": task_id, "score": score}


if __name__ == "__main__":
    all_scores = []

    for task in TASKS:
        result = run_task(task["task_id"])
        all_scores.append(result["score"])

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"baseline_score={avg:.2f}")