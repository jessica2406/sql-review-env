import sqlite3
import uuid
from typing import Optional
from models import Action, Observation, State

# ─────────────────────────────────────────
# ALL 3 TASKS (Easy → Medium → Hard)
# ─────────────────────────────────────────

TASKS = [
    {
        "task_id": "easy_1",
        "difficulty": "easy",
        "schema": """
            Table: employees
            Columns: id (INTEGER), name (TEXT), department (TEXT), salary (REAL)
            Sample data:
              (1, 'Alice', 'Engineering', 90000)
              (2, 'Bob', 'Marketing', 70000)
              (3, 'Carol', 'Engineering', 85000)
        """,
        "buggy_query": "SELECT name, salary FROM employees WHERE department = 'engineering'",
        "correct_query": "SELECT name, salary FROM employees WHERE department = 'Engineering'",
        "description": "Find all employees in the Engineering department",
        "bug_hint": "SQL string comparisons are case-sensitive in most databases",
        "setup_sql": """
            CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary REAL);
            INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000);
            INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 70000);
            INSERT INTO employees VALUES (3, 'Carol', 'Engineering', 85000);
        """,
        "expected_output": "[('Alice', 90000.0), ('Carol', 85000.0)]"
    },
    {
        "task_id": "medium_1",
        "difficulty": "medium",
        "schema": """
            Table: orders
            Columns: id (INTEGER), customer (TEXT), amount (REAL), status (TEXT)
            Sample data:
              (1, 'Alice', 150.0, 'completed')
              (2, 'Bob', 200.0, 'pending')
              (3, 'Alice', 300.0, 'completed')
              (4, 'Carol', 100.0, 'completed')
        """,
        "buggy_query": """
            SELECT customer, SUM(amount)
            FROM orders
            WHERE status = 'completed'
        """,
        "correct_query": """
            SELECT customer, SUM(amount)
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer
        """,
        "description": "Find the total amount spent by each customer on completed orders",
        "bug_hint": "When using aggregate functions like SUM(), you need to GROUP BY non-aggregated columns",
        "setup_sql": """
            CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL, status TEXT);
            INSERT INTO orders VALUES (1, 'Alice', 150.0, 'completed');
            INSERT INTO orders VALUES (2, 'Bob', 200.0, 'pending');
            INSERT INTO orders VALUES (3, 'Alice', 300.0, 'completed');
            INSERT INTO orders VALUES (4, 'Carol', 100.0, 'completed');
        """,
        "expected_output": "[('Alice', 450.0), ('Carol', 100.0)]"
    },
    {
        "task_id": "hard_1",
        "difficulty": "hard",
        "schema": """
            Table: employees
            Columns: id (INTEGER), name (TEXT), department (TEXT), salary (REAL), manager_id (INTEGER)
            Sample data:
              (1, 'Alice', 'Engineering', 90000, NULL)
              (2, 'Bob', 'Engineering', 70000, 1)
              (3, 'Carol', 'Engineering', 85000, 1)
              (4, 'Dave', 'Marketing', 75000, NULL)
              (5, 'Eve', 'Marketing', 65000, 4)
        """,
        "buggy_query": """
            SELECT e.name, e.salary, m.name as manager_name
            FROM employees e
            JOIN employees m ON e.manager_id = m.id
        """,
        "correct_query": """
            SELECT e.name, e.salary, m.name as manager_name
            FROM employees e
            LEFT JOIN employees m ON e.manager_id = m.id
        """,
        "description": "List all employees with their salary and their manager's name (managers themselves should still appear)",
        "bug_hint": "A regular JOIN excludes rows with no match — managers have NULL manager_id so they disappear",
        "setup_sql": """
            CREATE TABLE employees (id INTEGER, name TEXT, department TEXT, salary REAL, manager_id INTEGER);
            INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 90000, NULL);
            INSERT INTO employees VALUES (2, 'Bob', 'Engineering', 70000, 1);
            INSERT INTO employees VALUES (3, 'Carol', 'Engineering', 85000, 1);
            INSERT INTO employees VALUES (4, 'Dave', 'Marketing', 75000, NULL);
            INSERT INTO employees VALUES (5, 'Eve', 'Marketing', 65000, 4);
        """,
        "expected_output": "[('Alice', 90000.0, None), ('Bob', 70000.0, 'Alice'), ('Carol', 85000.0, 'Alice'), ('Dave', 75000.0, None), ('Eve', 65000.0, 'Dave')]"
    }
]


# ─────────────────────────────────────────
# GRADER — runs the query and scores it
# ─────────────────────────────────────────

def run_query(setup_sql: str, query: str):
    """Run a query on a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        for statement in setup_sql.strip().split(";"):
            s = statement.strip()
            if s:
                cursor.execute(s)
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return result, None
    except Exception as e:
        conn.close()
        return None, str(e)


def grade(task: dict, agent_query: str) -> tuple[float, str]:
    """
    Score the agent's fixed query.
    Returns (score 0.0-1.0, feedback string)
    """
    # Run the correct query to get expected results
    expected_result, _ = run_query(task["setup_sql"], task["correct_query"])

    # Run the agent's query
    agent_result, error = run_query(task["setup_sql"], agent_query)

    if error:
        return 0.0, f"Your query has a syntax error: {error}"

    if agent_result is None:
        return 0.0, "Query returned no result."

    # Exact match = full score
    if set(agent_result) == set(expected_result):
        return 1.0, "Perfect! Your query returns exactly the correct results."

    # Partial score: how many correct rows did they get?
    if len(expected_result) == 0:
        return 0.0, "Expected empty result but got rows."

    correct_rows = len(set(agent_result) & set(expected_result))
    partial = round(correct_rows / len(expected_result), 2)

    if partial > 0:
        return partial, f"Partial credit: {correct_rows}/{len(expected_result)} correct rows returned."

    return 0.0, f"Query ran but returned wrong results. Expected: {expected_result}, Got: {agent_result}"


# ─────────────────────────────────────────
# ENVIRONMENT CLASS
# ─────────────────────────────────────────

class SQLReviewEnvironment:

    def __init__(self):
        self._state: Optional[State] = None
        self._task_index: int = 0

    def reset(self) -> Observation:
        """Start a fresh episode from task 0 (easy)."""
        self._task_index = 0
        self._state = State(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_id=TASKS[0]["task_id"],
            total_score=0.0,
        )
        return self._make_observation(reward=0.0, done=False, feedback="New episode started. Fix the SQL query!")

    def step(self, action: Action) -> Observation:
        """Agent submits a fixed query. Grade it and move to next task if correct."""
        if self._state is None:
            raise ValueError("Call reset() before step()")

        task = TASKS[self._task_index]
        score, feedback = grade(task, action.fixed_query)

        self._state.step_count += 1
        self._state.total_score += score

        # Move to next task if agent got it right OR used all steps
        done = False
        if score == 1.0:
            if self._task_index < len(TASKS) - 1:
                self._task_index += 1
                self._state.current_task_id = TASKS[self._task_index]["task_id"]
                feedback += f" Moving to next task: {TASKS[self._task_index]['difficulty']}!"
            else:
                done = True
                feedback += " All tasks complete! 🎉"
        elif self._state.step_count >= self._state.max_steps:
            done = True
            feedback += " Max steps reached."

        return self._make_observation(reward=score, done=done, feedback=feedback)

    def state(self) -> State:
        if self._state is None:
            raise ValueError("Call reset() first")
        return self._state

    def _make_observation(self, reward: float, done: bool, feedback: str) -> Observation:
        task = TASKS[self._task_index]
        return Observation(
            task_id=task["task_id"],
            buggy_query=task["buggy_query"],
            table_schema=task["schema"],
            expected_output=task["expected_output"],
            reward=reward,
            done=done,
            feedback=feedback,
        )