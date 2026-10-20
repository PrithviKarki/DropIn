import asyncio
import argparse
import json
import time
from typing import TypedDict, Optional, Dict, Any, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import our 3 specialized agents
from data_agent import DataRetrievalAgent
from analysis_agent import AcademicAnalysisAgent
from intervention_agent import InterventionPlanningAgent


# Step 1: Define the Shared Graph State
class DropInState(TypedDict):
    """
    Master shared state for the DropIn LangGraph multi-agent pipeline.
    Each specialized agent node reads from and writes to specific keys in this state.
    """
    # Input
    student_id: int

    # Agent 1 (Data Retrieval Agent via MCP) output
    raw_student_data: Optional[str]

    # Agent 2 (Academic Analysis Agent) structured Pydantic dict output
    risk_profile: Optional[Dict[str, Any]]

    # Agent 3 (Intervention Planning Agent) final Markdown output
    intervention_plan: Optional[str]

    # Pipeline control & observability
    error: Optional[str]
    metrics: Dict[str, Any]


def create_initial_state(student_id: int) -> DropInState:
    """Helper to initialize a clean DropInState before starting the LangGraph run."""
    return {
        "student_id": student_id,
        "raw_student_data": None,
        "risk_profile": None,
        "intervention_plan": None,
        "error": None,
        "metrics": {
            "steps_completed": [],
            "node_latencies_sec": {},
        },
    }


# Step 2: Wrap Agents as Asynchronous Graph Nodes
async def retrieve_data_node(state: DropInState) -> Dict[str, Any]:
    """
    Node 1: Invokes DataRetrievalAgent via MCP to fetch factual student data.
    Reads: state['student_id']
    Writes: state['raw_student_data'], state['metrics'], state['error']
    """
    student_id = state["student_id"]
    print(f"\n[ORCHESTRATOR -> NODE 1: Data Retrieval] Fetching profile for Student ID {student_id}...")
    start_time = time.perf_counter()

    try:
        agent = DataRetrievalAgent()
        query = f"Get the complete student profile for student ID {student_id}."
        raw_data = await agent.run(query)
        elapsed = round(time.perf_counter() - start_time, 3)

        metrics = dict(state.get("metrics", {}))
        metrics.setdefault("steps_completed", []).append("retrieve_data")
        metrics.setdefault("node_latencies_sec", {})["retrieve_data"] = elapsed

        if not raw_data or "no text returned" in raw_data.lower():
            return {
                "raw_student_data": None,
                "error": f"Data Retrieval Agent returned empty data for student ID {student_id}.",
                "metrics": metrics,
            }

        return {
            "raw_student_data": raw_data,
            "metrics": metrics,
        }
    except Exception as e:
        return {
            "error": f"Exception in retrieve_data_node: {str(e)}",
        }


async def analyze_risk_node(state: DropInState) -> Dict[str, Any]:
    """
    Node 2: Invokes AcademicAnalysisAgent to evaluate raw student data into a Pydantic RiskProfile dict.
    Reads: state['raw_student_data']
    Writes: state['risk_profile'], state['metrics'], state['error']
    """
    print("\n[ORCHESTRATOR -> NODE 2: Academic Analysis] Evaluating student risk factors...")
    start_time = time.perf_counter()

    try:
        agent = AcademicAnalysisAgent()
        risk_profile = agent.analyze_student_data(state["raw_student_data"])
        elapsed = round(time.perf_counter() - start_time, 3)

        metrics = dict(state.get("metrics", {}))
        metrics.setdefault("steps_completed", []).append("analyze_risk")
        metrics.setdefault("node_latencies_sec", {})["analyze_risk"] = elapsed

        if not risk_profile:
            return {
                "risk_profile": None,
                "error": "Academic Analysis Agent failed to produce a valid JSON RiskProfile.",
                "metrics": metrics,
            }

        return {
            "risk_profile": risk_profile,
            "metrics": metrics,
        }
    except Exception as e:
        return {
            "error": f"Exception in analyze_risk_node: {str(e)}",
        }


async def draft_plan_node(state: DropInState) -> Dict[str, Any]:
    """
    Node 3: Invokes InterventionPlanningAgent to generate a 3-step advising plan from the RiskProfile.
    Reads: state['risk_profile']
    Writes: state['intervention_plan'], state['metrics'], state['error']
    """
    print("\n[ORCHESTRATOR -> NODE 3: Intervention Planning] Drafting targeted intervention plan...")
    start_time = time.perf_counter()

    try:
        agent = InterventionPlanningAgent()
        plan = agent.draft_intervention_plan(state["risk_profile"])
        elapsed = round(time.perf_counter() - start_time, 3)

        metrics = dict(state.get("metrics", {}))
        metrics.setdefault("steps_completed", []).append("draft_plan")
        metrics.setdefault("node_latencies_sec", {})["draft_plan"] = elapsed

        if not plan:
            return {
                "intervention_plan": None,
                "error": "Intervention Planning Agent returned an empty plan.",
                "metrics": metrics,
            }

        return {
            "intervention_plan": plan,
            "metrics": metrics,
        }
    except Exception as e:
        return {
            "error": f"Exception in draft_plan_node: {str(e)}",
        }


# Step 3: Implement Conditional Edges & State Validation
REQUIRED_RISK_KEYS = {"is_at_risk", "primary_risk_factor", "risk_summary", "recommended_urgency"}


def check_data_retrieval(state: DropInState) -> Literal["analyze_risk", "end"]:
    """
    Conditional Edge 1: Validates that retrieve_data_node succeeded and found a real student record
    before allowing the pipeline to proceed to analyze_risk_node.
    """
    if state.get("error"):
        print(f"🛑 [GUARDRAIL: check_data_retrieval] Halting pipeline due to error: {state['error']}")
        return "end"

    raw_data = (state.get("raw_student_data") or "").strip()
    not_found_phrases = ("not found", "no student", "does not exist", "no record", "0 rows")
    if not raw_data or any(phrase in raw_data.lower() for phrase in not_found_phrases):
        print(f"🛑 [GUARDRAIL: check_data_retrieval] Student ID {state['student_id']} not found in DB. Halting to prevent hallucination.")
        return "end"

    print("✅ [GUARDRAIL: check_data_retrieval] Valid student data verified. Routing -> analyze_risk")
    return "analyze_risk"


def check_risk_analysis(state: DropInState) -> Literal["draft_plan", "end"]:
    """
    Conditional Edge 2: Validates that analyze_risk_node produced a complete Pydantic RiskProfile
    dictionary with all required keys before routing to draft_plan_node.
    """
    if state.get("error"):
        print(f"🛑 [GUARDRAIL: check_risk_analysis] Halting pipeline due to error: {state['error']}")
        return "end"

    profile = state.get("risk_profile")
    if not isinstance(profile, dict) or not REQUIRED_RISK_KEYS.issubset(profile.keys()):
        print(f"🛑 [GUARDRAIL: check_risk_analysis] Invalid RiskProfile schema: {profile}. Halting pipeline.")
        return "end"

    print("✅ [GUARDRAIL: check_risk_analysis] Structured RiskProfile schema verified. Routing -> draft_plan")
    return "draft_plan"


def build_orchestrator_graph():
    """
    Constructs and compiles the Master LangGraph StateGraph with conditional validation edges.
    """
    workflow = StateGraph(DropInState)

    # 1. Register the 3 specialized agent nodes
    workflow.add_node("retrieve_data", retrieve_data_node)
    workflow.add_node("analyze_risk", analyze_risk_node)
    workflow.add_node("draft_plan", draft_plan_node)

    # 2. Set entry point
    workflow.set_entry_point("retrieve_data")

    # 3. Add conditional edge after Data Retrieval
    workflow.add_conditional_edges(
        "retrieve_data",
        check_data_retrieval,
        {
            "analyze_risk": "analyze_risk",
            "end": END,
        },
    )

    # 4. Add conditional edge after Academic Analysis
    workflow.add_conditional_edges(
        "analyze_risk",
        check_risk_analysis,
        {
            "draft_plan": "draft_plan",
            "end": END,
        },
    )

    # 5. Terminal edge after Intervention Planning
    workflow.add_edge("draft_plan", END)

    return workflow.compile()


# Step 4: CLI Runner & Execution Trace Output
async def run_orchestrator(student_id: int, save_trace: Optional[str] = None) -> DropInState:
    """
    Executes the full Master LangGraph pipeline for a given student_id and prints the execution trace.
    """
    print("=" * 75)
    print(f"🌐 STARTING DROPIN LANGGRAPH ORCHESTRATOR | STUDENT ID: {student_id}")
    print("=" * 75)

    app = build_orchestrator_graph()
    initial_state = create_initial_state(student_id)

    pipeline_start = time.perf_counter()
    final_state: DropInState = await app.ainvoke(initial_state)
    total_latency = round(time.perf_counter() - pipeline_start, 3)

    final_state.setdefault("metrics", {})["total_pipeline_latency_sec"] = total_latency

    # Print Execution Trace Summary
    print("\n" + "=" * 75)
    print("📊 LANGGRAPH EXECUTION TRACE & METRICS")
    print("=" * 75)
    print(f"Student ID       : {final_state['student_id']}")
    print(f"Steps Completed  : {' -> '.join(final_state['metrics'].get('steps_completed', [])) or 'None'}")
    print(f"Node Latencies   : {json.dumps(final_state['metrics'].get('node_latencies_sec', {}))}")
    print(f"Total Latency    : {total_latency}s")

    if final_state.get("error"):
        print(f"Pipeline Status  : ❌ HALTED ({final_state['error']})")
    elif final_state.get("intervention_plan"):
        print("Pipeline Status  : ✅ COMPLETED")
        print("\n📋 STRUCTURED RISK PROFILE (Agent 2 Output):")
        print(json.dumps(final_state["risk_profile"], indent=2))
        print("\n🏆 FINAL MULTI-AGENT INTERVENTION PLAN (Agent 3 Output):")
        print("-" * 75)
        print(final_state["intervention_plan"])
        print("-" * 75)
    else:
        print("Pipeline Status  : ⚠️ TERMINATED EARLY BY GUARDRAIL")

    if save_trace:
        with open(save_trace, "w", encoding="utf-8") as f:
            json.dump(final_state, f, indent=2)
        print(f"\n💾 Saved full execution state trace to: {save_trace}")

    return final_state


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the DropIn Master LangGraph Orchestrator")
    parser.add_argument("--id", type=int, default=4, help="Student ID to process through the LangGraph pipeline")
    parser.add_argument("--save-trace", type=str, default=None, help="Optional JSON file path to save the execution trace")
    args = parser.parse_args()

    asyncio.run(run_orchestrator(student_id=args.id, save_trace=args.save_trace))
