import os
import json
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Define the Pydantic Schema
class RiskProfile(BaseModel):
    is_at_risk: bool = Field(description="True if the student needs immediate intervention")
    primary_risk_factor: str = Field(description="The main reason for struggle (e.g., 'Low Attendance', 'Poor Grades', 'Missed Assignments', 'None')")
    risk_summary: str = Field(description="A concise 2-sentence summary explaining exactly why the student is struggling based on the data")
    recommended_urgency: str = Field(description="Must be exactly 'Low', 'Medium', or 'High'")

# 2. Build the Agent Class
class AcademicAnalysisAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = 'gemini-2.5-flash'
        
    def analyze_student_data(self, raw_data: str) -> dict:
        """
        Takes raw student data from the database and returns a structured Risk Profile dictionary.
        """
        prompt = f"""
        You are the Academic Analysis Agent. 
        Your job is to act as a strict data evaluator. Do not offer advice or draft intervention plans.
        Look at the raw student data provided below. Analyze their GPA, attendance, and missed assignments.
        Output a structured Risk Profile highlighting the root causes of their academic standing.
        
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
                temperature=0.2 # Low temperature forces the LLM to be highly analytical and factual
            ),
        )
        
        # The response is a pure JSON string. We parse it into a native Python dictionary
        # so that it can be easily passed to the next agent!
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
    
    # Setup argparse so you can test different scenarios from the terminal
    parser = argparse.ArgumentParser(description="Test the Analysis Agent")
    parser.add_argument("--scenario", type=str, default="struggling", choices=["struggling", "excelling", "mixed"])
    args = parser.parse_args()
    
    # Mock data depending on the chosen scenario
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
    # Pretty-print the resulting Python dictionary
    print(json.dumps(profile, indent=2))
