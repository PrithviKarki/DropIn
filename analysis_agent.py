import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def _is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 500, 503)


# 1. Define the Pydantic Schema
class RiskProfile(BaseModel):
    is_at_risk: bool = Field(description="True if the student needs immediate intervention")
    primary_risk_factor: str = Field(description="The main reason for struggle (e.g., 'Low Attendance', 'Poor Grades', 'Missed Assignments', 'None')")
    risk_summary: str = Field(description="A concise 2-sentence summary explaining exactly why the student is struggling based on the data")
    recommended_urgency: str = Field(description="Must be exactly 'Low', 'Medium', or 'High'")


# 2. Build the Agent Class
class AcademicAnalysisAgent:
    def __init__(self, model: str = "gemini-2.5-flash", max_output_tokens: int = 1024):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }

    @retry(
        retry=retry_if_exception(_is_retryable_api_error),
        wait=wait_exponential(multiplier=2, min=15, max=90),
        stop=stop_after_attempt(6),
        before_sleep=lambda retry_state: print(
            f"⚠️  [API Rate Limit / Error] Attempt {retry_state.attempt_number} failed: {retry_state.outcome.exception()}. "
            f"Waiting {round(retry_state.next_action.sleep, 1)}s before retry..."
        ),
    )
    def analyze_student_data(self, raw_data: str) -> dict:
        """
        Takes raw student data from the database and returns a structured Risk Profile dictionary.
        """
        prompt = f"""
        You are the Academic Analysis Agent. 
        Your job is to act as a strict data evaluator. Do not offer advice or draft intervention plans.
        Look at the raw student data provided below. Analyze their GPA, attendance, and missed assignments.
        Output a structured Risk Profile highlighting the root causes of their academic standing.
        For 'primary_risk_factor', choose strictly from: 'Low Attendance', 'Poor Grades', 'Missed Assignments', or 'None'.
        
        Raw Data:
        {raw_data}
        """

        print("🧠 Analysis Agent is evaluating data...")

        # Call Gemini and enforce structured JSON output
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RiskProfile,
                temperature=0.2,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        self.last_usage = {
            "prompt_tokens": (getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            "completion_tokens": (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0,
            "total_tokens": (getattr(usage, "total_token_count", 0) or 0) if usage else 0,
            "llm_calls": 1,
        }

        try:
            risk_profile_dict = json.loads(response.text)
            return risk_profile_dict
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM: {e}")
            return {}


# 3. Built-in Testing Logic
if __name__ == "__main__":
    from dotenv import load_dotenv
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Test the Analysis Agent")
    parser.add_argument("--scenario", type=str, default="struggling", choices=["struggling", "excelling", "mixed"])
    args = parser.parse_args()

    if args.scenario == "struggling":
        test_data = "Student: James Wilson | GPA: 1.8 | Attendance: 55% | Missed Assignments: 8"
    elif args.scenario == "excelling":
        test_data = "Student: Sarah Connor | GPA: 3.9 | Attendance: 98% | Missed Assignments: 0"
    else:
        test_data = "Student: John Doe | GPA: 2.9 | Attendance: 80% | Missed Assignments: 2"

    print(f"Testing Scenario: {args.scenario.upper()}")
    print(f"Input Data: {test_data}")
    print("-" * 50)

    agent = AcademicAnalysisAgent()
    profile = agent.analyze_student_data(test_data)

    print("\n✅ Final Risk Profile Dictionary:")
    print(json.dumps(profile, indent=2))
    print(f"\n📊 Token Usage: {json.dumps(agent.last_usage, indent=2)}")
