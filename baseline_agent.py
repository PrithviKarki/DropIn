import sqlite3
import json
import time
import os
import argparse
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

DB_NAME = "dropin_synthetic.db"


def _is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 500, 503)


class BaselineEvaluationOutput(BaseModel):
    is_at_risk: bool = Field(description="True if the student needs immediate intervention, False otherwise")
    primary_risk_factor: str = Field(
        description="Must be strictly one of: 'Low Attendance', 'Poor Grades', 'Missed Assignments', or 'None'"
    )
    recommended_urgency: str = Field(description="Must be strictly 'Low', 'Medium', or 'High'")
    risk_summary: str = Field(description="A concise 2-sentence summary explaining why the student is struggling")
    intervention_plan: str = Field(
        description="A comprehensive, 3-step Markdown intervention plan tailored to the student"
    )


def get_student_data(student_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves raw student data from the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def engineer_prompt(student_data: Dict[str, Any]) -> str:
    """Constructs the unified single-LLM prompt architecture."""
    # Remove the synthetic generation tag 'risk_level' from prompt input so the Baseline LLM
    # must actually reason over GPA, attendance, and missed assignments just like the Multi-Agent system.
    clean_input = {k: v for k, v in student_data.items() if k != "risk_level"}
    raw_data_json = json.dumps(clean_input, indent=2)

    prompt = f"""
You are an expert Academic Advisor AI. Your goal is to review a student's academic profile and output a complete risk evaluation and 3-step intervention plan in a single pass.

As a single-LLM system, you must handle the data ingestion, risk analysis, and intervention planning all in one step.

Here is the raw student data profile:
```json
{raw_data_json}
```

Analyze this student's GPA, attendance_percentage, missed_assignments, and academic_notes:
1. Determine if they are at risk (`is_at_risk`), their `recommended_urgency` ('Low', 'Medium', or 'High'), and their `primary_risk_factor` ('Low Attendance', 'Poor Grades', 'Missed Assignments', or 'None').
2. Draft a complete 3-step Markdown `intervention_plan` tailored to their standing.
"""
    return prompt


@retry(
    retry=retry_if_exception(_is_retryable_api_error),
    wait=wait_exponential(multiplier=2, min=15, max=90),
    stop=stop_after_attempt(6),
    before_sleep=lambda retry_state: print(
        f"⚠️  [API Rate Limit / Error] Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}. "
        f"Waiting {round(retry_state.next_action.sleep, 1)}s before retry..."
    ),
)
def _call_baseline_gemini(prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 2048):
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BaselineEvaluationOutput,
            temperature=0.4,
            max_output_tokens=max_output_tokens,
        ),
    )


def run_baseline_evaluation(student_id: int, model: str = "gemini-2.5-flash") -> Dict[str, Any]:
    """
    Runs the single-LLM baseline architecture with matched experimental controls
    (same model, token tracking, and latency telemetry as the Multi-Agent Orchestrator).
    """
    start_time = time.perf_counter()
    student_data = get_student_data(student_id)

    if not student_data:
        return {
            "student_id": student_id,
            "error": f"Student ID {student_id} not found in database '{DB_NAME}'.",
        }

    prompt = engineer_prompt(student_data)
    response = _call_baseline_gemini(prompt, model=model)
    latency_sec = round(time.perf_counter() - start_time, 3)

    usage = getattr(response, "usage_metadata", None)
    token_usage = {
        "prompt_tokens": (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
        "completion_tokens": (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
        "total_tokens": (getattr(usage, "total_token_count", 0) or 0) if usage else 0,
        "llm_calls": 1,
    }

    parsed = json.loads(response.text)

    return {
        "student_id": student_id,
        "risk_profile": {
            "is_at_risk": parsed.get("is_at_risk"),
            "primary_risk_factor": parsed.get("primary_risk_factor"),
            "recommended_urgency": parsed.get("recommended_urgency"),
            "risk_summary": parsed.get("risk_summary"),
        },
        "intervention_plan": parsed.get("intervention_plan", ""),
        "metrics": {
            "model": model,
            "total_pipeline_latency_sec": latency_sec,
            "token_usage": token_usage,
        },
        "error": None,
    }


def generate_intervention_plan(student_id: int):
    print(f"Fetching data for student ID: {student_id}...")
    result = run_baseline_evaluation(student_id)

    if result.get("error"):
        print(result["error"])
        return result

    print("\n" + "=" * 60)
    print(f"📋 BASELINE SINGLE-LLM OUTPUT (Student ID: {student_id})")
    print("=" * 60)
    print("Risk Evaluation:")
    print(json.dumps(result["risk_profile"], indent=2))
    print("\nIntervention Plan:")
    print(result["intervention_plan"])
    print("=" * 60)
    print(f"📊 Baseline Metrics: {json.dumps(result['metrics'], indent=2)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Single-LLM Intervention Generator")
    parser.add_argument("--id", type=int, default=4, help="Student ID to analyze (1-100)")
    args = parser.parse_args()

    generate_intervention_plan(args.id)
