import sqlite3
import json
import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

DB_NAME = "dropin_synthetic.db"

def setup_database():
    """Defines the SQLite schema and creates the database file."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop table if it exists to allow clean re-runs
    cursor.execute('DROP TABLE IF EXISTS students')
    
    # Create the schema
    cursor.execute('''
        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            major TEXT,
            gpa REAL,
            attendance_percentage INTEGER,
            missed_assignments INTEGER,
            risk_level TEXT,
            academic_notes TEXT
        )
    ''')
    conn.commit()
    return conn

def generate_batch(batch_num, risk_target):
    """Calls Gemini to generate 20 specific synthetic student profiles."""
    prompt = f"""
    Generate a JSON array of exactly 20 fictional college student profiles.
    Target risk level for this batch: {risk_target}.
    
    The JSON must be a raw array of objects with these exact keys and types:
    - "first_name": (string)
    - "last_name": (string)
    - "major": (string)
    - "gpa": (float between 1.0 and 4.0)
    - "attendance_percentage": (integer between 40 and 100)
    - "missed_assignments": (integer)
    - "risk_level": (string strictly either "Low", "Medium", or "High")
    - "academic_notes": (string: 1-2 sentences of recent professor or advisor notes)
    
    Output ONLY valid JSON. Do not include markdown blocks, backticks, or conversational text.
    """
    
    print(f"Generating batch {batch_num}/5 (Target Risk: {risk_target})...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    
    # Clean response string to ensure JSON parsing works
    raw_text = response.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:-3].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:-3].strip()
        
    return json.loads(raw_text)

def main():
    conn = setup_database()
    cursor = conn.cursor()
    
    # Distributed risk targets to ensure your agents have varied data to analyze
    risk_targets = ["Low", "Low", "Medium", "High", "Mixed"]
    
    for i, risk in enumerate(risk_targets, 1):
        try:
            students = generate_batch(i, risk)
            
            for s in students:
                cursor.execute('''
                    INSERT INTO students 
                    (first_name, last_name, major, gpa, attendance_percentage, missed_assignments, risk_level, academic_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (s['first_name'], s['last_name'], s['major'], float(s['gpa']), 
                      int(s['attendance_percentage']), int(s['missed_assignments']), 
                      s['risk_level'], s['academic_notes']))
                
            conn.commit()
            print(f"Batch {i} inserted successfully.")
            
            # 10-second delay to prevent triggering the Free Tier RPM rate limit
            if i < 5:
                print("Pausing for 10 seconds to respect API rate limits...")
                time.sleep(10)
                
        except json.JSONDecodeError:
            print(f"Error on batch {i}: Gemini did not return valid JSON.")
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            
    print(f"\nDatabase setup complete! A total of 100 profiles were created in {DB_NAME}.")
    conn.close()

if __name__ == "__main__":
    main()