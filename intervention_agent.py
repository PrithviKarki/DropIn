import os
import json
from google import genai
from google.genai import types, errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception


def _is_retryable_api_error(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 500, 503)


# Build the Agent Class
class InterventionPlanningAgent:
    def __init__(self, model: str = "gemini-2.5-flash", max_output_tokens: int = 2048):
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
    def draft_intervention_plan(self, risk_profile: dict) -> str:
        """
        Takes a structured Risk Profile dictionary and drafts an actionable advising plan.
        """
        profile_string = json.dumps(risk_profile, indent=2)

        prompt = f"""
        You are the Intervention Planning Agent, acting as an empathetic academic advisor.
        Your job is to read the academic risk profile provided below and draft a concrete, 3-step 
        action plan tailored specifically to the student's root problems.
        
        IMPORTANT: 
        - Do not analyze raw data (that is the Analysis Agent's job). 
        - Assume the risk profile provided is 100% accurate.
        - If the profile says the student is NOT at risk, your 3 steps should focus on 
          maintaining their success or pursuing excellence (like internships or honors).
        
        Risk Profile:
        {profile_string}
        
        Format your response clearly using Markdown formatting.
        """

        print("📝 Intervention Agent is drafting the plan...")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
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

        return response.text


# Built-in Testing Logic
if __name__ == "__main__":
    from dotenv import load_dotenv
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Test the Intervention Agent")
    parser.add_argument("--scenario", type=str, default="attendance", choices=["attendance", "grades", "perfect"])
    args = parser.parse_args()

    if args.scenario == "attendance":
        mock_profile = {
            "is_at_risk": True,
            "primary_risk_factor": "Low Attendance",
            "risk_summary": "The student is missing over 45% of classes, leading to missed assignments.",
            "recommended_urgency": "High"
        }
    elif args.scenario == "grades":
        mock_profile = {
            "is_at_risk": True,
            "primary_risk_factor": "Poor Grades",
            "risk_summary": "The student attends class regularly but is failing the core exams, suggesting a lack of comprehension.",
            "recommended_urgency": "Medium"
        }
    else:
        mock_profile = {
            "is_at_risk": False,
            "primary_risk_factor": "None",
            "risk_summary": "The student has near perfect attendance and excellent grades.",
            "recommended_urgency": "Low"
        }

    print(f"Testing Scenario: {args.scenario.upper()} RISK")
    print("-" * 50)

    agent = InterventionPlanningAgent()
    plan = agent.draft_intervention_plan(mock_profile)

    print("\n✅ Final Intervention Plan:\n")
    print(plan)
    print("=" * 50)
    print(f"📊 Token Usage: {json.dumps(agent.last_usage, indent=2)}")
