import asyncio
import json
import argparse
from dotenv import load_dotenv

# Import our 3 specialized agents
from data_agent import DataRetrievalAgent
from analysis_agent import AcademicAnalysisAgent
from intervention_agent import InterventionPlanningAgent

async def run_bucket_brigade(student_id: int):
    """
    Tests the sequential "handoff" of data between the three specialized agents.
    This simulates what the LangGraph Orchestrator will do automatically in Week 8.
    """
    print(f"\n🚀 STARTING MULTI-AGENT HANDOFF TEST FOR STUDENT ID: {student_id}")
    print("=" * 70)
    
    # ---------------------------------------------------------
    # AGENT 1: Data Retrieval
    # ---------------------------------------------------------
    print("\n[AGENT 1] Triggering Data Retrieval Agent...")
    data_agent = DataRetrievalAgent()
    query = f"Get the complete student profile for student ID {student_id}."
    
    # The agent autonomously connects to MCP, writes SQL, and fetches the data
    raw_data = await data_agent.run(query)
    
    # ---------------------------------------------------------
    # AGENT 2: Academic Analysis
    # ---------------------------------------------------------
    print("\n[AGENT 2] Triggering Academic Analysis Agent...")
    print("Passing raw data from Data Agent -> Analysis Agent...")
    analysis_agent = AcademicAnalysisAgent()
    
    # The agent reads the raw numbers and outputs a structured Pydantic JSON dictionary
    risk_profile_dict = analysis_agent.analyze_student_data(raw_data)
    
    # ---------------------------------------------------------
    # AGENT 3: Intervention Planning
    # ---------------------------------------------------------
    print("\n[AGENT 3] Triggering Intervention Planning Agent...")
    print("Passing structured Risk Profile from Analysis Agent -> Planning Agent...")
    plan_agent = InterventionPlanningAgent()
    
    # The agent reads the parsed risk profile and drafts a final Markdown plan
    final_plan = plan_agent.draft_intervention_plan(risk_profile_dict)
    
    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("🏆 FINAL SWARM OUTPUT (Intervention Plan):")
    print("=" * 70)
    print(final_plan)
    print("=" * 70)

if __name__ == "__main__":
    load_dotenv()
    
    # Allow testing different student IDs from the terminal
    parser = argparse.ArgumentParser(description="Run the Multi-Agent Handoff Test")
    parser.add_argument("--id", type=int, default=4, help="Student ID to run through the swarm")
    args = parser.parse_args()
    
    asyncio.run(run_bucket_brigade(args.id))
