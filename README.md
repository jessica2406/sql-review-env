---
title: SQL Review Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---


# SQL Review Environment 🔍

An OpenEnv-compatible RL environment where an AI agent learns to identify 
and fix bugs in SQL queries.

## What is this?

Real SQL bugs cost companies millions of dollars. This environment trains 
agents to catch common SQL mistakes before they reach production — from 
simple case-sensitivity errors to complex JOIN bugs.

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| easy_1 | 🟢 Easy | Fix a case-sensitivity bug in a WHERE clause |
| medium_1 | 🟡 Medium | Fix a missing GROUP BY in an aggregation query |
| hard_1 | 🔴 Hard | Fix an incorrect JOIN type causing missing rows |

## Action Space
```json
{
  "fixed_query": "SELECT ... your corrected SQL ...",
  "explanation": "What was wrong and how you fixed it"
}
```

## Observation Space
```json
{
  "task_id": "easy_1",
  "buggy_query": "The broken SQL query to fix",
  "table_schema": "Table structure and sample data",
  "expected_output": "What the correct result should look like",
  "reward": 0.0,
  "done": false,
  "feedback": "Human-readable feedback on your last attempt",
  "metadata": {}
}
```

## Reward Function

| Result | Score |
|--------|-------|
| Exact correct output | 1.0 |
| Partial correct rows | 0.0 – 0.99 |
| Syntax error | 0.0 |
| Wrong results | 0.0 |

## Setup & Usage

### Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker
```bash
docker build -t sql-review-env .
docker run -p 7860:7860 sql-review-env
```

### Run baseline inference
```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=http://127.0.0.1:7860
python inference.py
```

## Baseline Scores

| Task | Baseline Score (gpt-4o-mini) |
|------|------------------------------|
| easy_1 | 1.0 |
| medium_1 | 1.0 |
| hard_1 | 1.0 |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | The API endpoint for the environment |
| `MODEL_NAME` | The model identifier to use for inference |
| `HF_TOKEN` | Your Hugging Face / OpenAI API key |

## Project Structure
```
sql-review-env/
├── app.py            # FastAPI server
├── environment.py    # Core logic, tasks, graders
├── models.py         # Typed Action/Observation/State
├── inference.py      # Baseline inference script
├── openenv.yaml      # OpenEnv metadata
├── Dockerfile        # Container definition
└── README.md         # This file
```