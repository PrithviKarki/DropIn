import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Step 1: Define the Pydantic Model to enforce structured JSON output
class RiskProfile(BaseModel):
    is_at_risk: bool = Field(
        description="True if the student needs immediate intervention, False otherwise"
    )
    primary_risk_factor: str = Field(
        description="The main reason for struggle (e.g., 'Low Attendance', 'Poor Grades', 'Missed Assignments', 'None')"
    )
    risk_summary: str = Field(
        description="A concise 2-sentence summary explaining exactly why the student is struggling based on the data"
    )
    recommended_urgency: str = Field(
        description="Must be exactly 'Low', 'Medium', or 'High'"
    )

def test_analysis_agent_structure():
    """Tests if the LLM strictly adheres to the RiskProfile schema."""
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # This is mock data simulating what the Data Retrieval Agent would pass over
    mock_clean_data = """
    Student Name: James Wilson
    Major: Computer Science
    GPA: 1.8
    Attendance: 55%
    Missed Assignments: 8
    Notes: James has been falling asleep in class and missing critical labs.
    """
    
    prompt = f"""
    You are the Academic Analysis Agent. 
    Analyze this raw student data and output a structured risk profile.
    
    Raw Data:
    {mock_clean_data}
    """
    
    print("🤖 Analysis Agent is evaluating the raw data...")
    
    # Call Gemini and FORCE it to use our Pydantic schema
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            # This tells Gemini to return raw JSON instead of markdown
            response_mime_type="application/json",
            # This tells Gemini exactly what keys and types the JSON must have
            response_schema=RiskProfile, 
        ),
    )
    
    print("\n✅ Structured Output Received (Notice it is perfectly formatted JSON!):")
    print("-" * 50)
    print(response.text)
    print("-" * 50)

if __name__ == "__main__":
    test_analysis_agent_structure()

