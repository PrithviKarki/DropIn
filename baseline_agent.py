import sqlite3
import json
import argparse
from llm_client import generate_response_with_backoff

DB_NAME = "dropin_synthetic.db"

def get_student_data(student_id):
    """Retrieves raw student data from the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def engineer_prompt(student_data):
    """Constructs the unified single-LLM prompt architecture."""
    
    # Serialize the raw dictionary into formatted JSON for the prompt
    raw_data_json = json.dumps(student_data, indent=2)
    
    prompt = f"""
You are an expert Academic Advisor AI. Your goal is to review a student's academic profile and output a comprehensive intervention plan. 

As a single-LLM system, you must handle the data ingestion, analysis, and planning all in one step.

Here is the raw student data profile:
```json
{raw_data_json}
```

Please deeply analyze this data and provide a structured intervention plan that includes:
1. **Student Overview**: A brief summary of their current standing.
2. **Risk Analysis**: Why are they at risk, if at all? Connect their GPA, attendance, missed assignments, and notes.
3. **Recommended Interventions**: Specific, actionable steps the advising office or the student should take.
4. **Follow-up Timeline**: When and how to check back in with the student.

Format your response cleanly in Markdown.
"""
    return prompt

def generate_intervention_plan(student_id):
    print(f"Fetching data for student ID: {student_id}...")
    student_data = get_student_data(student_id)
    
    if not student_data:
        print(f"Error: Student ID {student_id} not found in database '{DB_NAME}'.")
        return
        
    print(f"Data found for {student_data['first_name']} {student_data['last_name']} (Risk: {student_data['risk_level']}). Generating plan...")
    prompt = engineer_prompt(student_data)
    
    try:
        # Utilize the rate-limited client developed in Week 1
        # We can use flash for speed or pro for deeper reasoning
        plan = generate_response_with_backoff(prompt, model="gemini-2.5-flash")
        
        print("\n" + "="*60)
        print(f"📋 BASELINE INTERVENTION PLAN: {student_data['first_name']} {student_data['last_name']}")
        print("="*60)
        print(plan)
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred while generating the plan: {e}")

if __name__ == "__main__":
    # Setup argparse so we can test different student IDs easily from the terminal
    parser = argparse.ArgumentParser(description="Baseline Single-LLM Intervention Generator")
    parser.add_argument("--id", type=int, default=1, help="Student ID to analyze (1-100)")
    args = parser.parse_args()
    
    generate_intervention_plan(args.id)

