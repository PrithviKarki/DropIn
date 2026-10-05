import os
import json
from google import genai
from google.genai import types

# Build the Agent Class
class InterventionPlanningAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = 'gemini-2.5-flash' # You can upgrade this to pro for deeper reasoning later
        
    def draft_intervention_plan(self, risk_profile: dict) -> str:
        """
        Takes a structured Risk Profile dictionary and drafts an actionable advising plan.
        """
        # Convert the Python dictionary back into a nicely formatted string for the LLM prompt
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
        
        # We don't enforce JSON here because we want a readable Markdown report!
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7 # Higher temperature allows for more creative and empathetic writing
            ),
        )
        
        return response.text

# Built-in Testing Logic
if __name__ == "__main__":
    from dotenv import load_dotenv
    import argparse
    
    load_dotenv()
    
    # Setup argparse so you can test different scenarios from the terminal
    parser = argparse.ArgumentParser(description="Test the Intervention Agent")
    parser.add_argument("--scenario", type=str, default="attendance", choices=["attendance", "grades", "perfect"])
    args = parser.parse_args()
    
    # These represent the exact kind of dictionaries the Analysis Agent outputs!
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
