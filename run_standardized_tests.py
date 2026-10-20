import asyncio
import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv

from ground_truth import select_standardized_scenarios, SCENARIOS_FILE
from baseline_agent import run_baseline_evaluation
from orchestrator import run_orchestrator

RESULTS_DIR = "results"
DEFAULT_LOG_FILE = os.path.join(RESULTS_DIR, "standardized_test_logs.json")


def load_or_create_scenarios() -> List[Dict[str, Any]]:
    """Loads the 20 ground-truth labeled scenarios from disk, or generates them if missing."""
    if os.path.exists(SCENARIOS_FILE):
        with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    scenarios = select_standardized_scenarios(20)
    with open(SCENARIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
    return scenarios


def load_existing_logs(log_path: str) -> Dict[str, Any]:
    """Loads existing test execution logs so interrupted runs can resume from the last checkpoint."""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "experiment_metadata": {
            "project_title": "DropIn: Evaluating Multi-Agent Systems for Proactive Academic Advising",
            "model_control": "gemini-2.5-flash",
            "total_scenarios_planned": 20,
            "last_updated": None,
        },
        "scenarios_executed": {},
    }


def save_logs(log_path: str, data: Dict[str, Any]):
    """Saves checkpointed test execution logs to disk."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    data["experiment_metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def run_standardized_suite(
    limit: int = 20,
    delay_sec: float = 8.0,
    force_rerun: bool = False,
    log_path: str = DEFAULT_LOG_FILE,
    model: str = "gemini-2.5-flash",
):
    """
    Runs the 20 standardized synthetic scenarios through both:
      1. Baseline Single-LLM Pipeline
      2. DropIn Multi-Agent LangGraph Orchestrator
    Saves raw output logs and execution traces after every scenario (checkpointed).
    """
    load_dotenv()
    scenarios = load_or_create_scenarios()[:limit]
    logs = load_existing_logs(log_path)
    logs["experiment_metadata"]["model_control"] = model

    print("=" * 80)
    print("🧪 DROPIN WEEK 9: STANDARDIZED TESTING HARNESS")
    print("=" * 80)
    print(f"Total Scenarios to Process : {len(scenarios)}")
    print(f"Matched Model Control      : {model}")
    print(f"Output Log File            : {log_path}")
    print(f"Inter-Run Pacing Delay     : {delay_sec}s (protects Free-Tier RPM quota)")
    print("=" * 80)

    for idx, scenario in enumerate(scenarios, start=1):
        sid = scenario["student_id"]
        sid_key = str(sid)

        existing = logs["scenarios_executed"].get(sid_key, {})
        baseline_done = existing.get("baseline", {}).get("error") is None and existing.get("baseline", {}).get("intervention_plan")
        multi_done = existing.get("multi_agent", {}).get("error") is None and existing.get("multi_agent", {}).get("intervention_plan")

        if not force_rerun and baseline_done and multi_done:
            print(f"\n⏭️  [{idx}/{len(scenarios)}] Student ID {sid} ({scenario['first_name']} {scenario['last_name']}) already completed. Skipping (use --force to re-run).")
            continue

        print("\n" + "#" * 80)
        print(
            f"▶️  SCENARIO [{idx}/{len(scenarios)}] | Student ID: {sid} ({scenario['first_name']} {scenario['last_name']}) "
            f"| GT Urgency: {scenario['ground_truth_urgency']} | GT Factor: {scenario['ground_truth_primary_factor']}"
        )
        print("#" * 80)

        scenario_record = existing if not force_rerun else {}
        scenario_record["ground_truth"] = scenario

        # ------------------------------------------------------------------
        # 1. Run Baseline Single-LLM Pipeline
        # ------------------------------------------------------------------
        if force_rerun or not baseline_done:
            print(f"\n--- [1/2] Running Baseline Single-LLM for Student ID {sid} ({model}) ---")
            try:
                baseline_result = run_baseline_evaluation(sid, model=model)
            except Exception as e:
                baseline_result = {"student_id": sid, "error": str(e)}
            scenario_record["baseline"] = baseline_result
            logs["scenarios_executed"][sid_key] = scenario_record
            save_logs(log_path, logs)
            print(f"💾 Checkpointed Baseline result for Student ID {sid}.")

            if delay_sec > 0:
                print(f"⏳ Pausing {delay_sec}s for API rate-limit pacing...")
                await asyncio.sleep(delay_sec)

        # ------------------------------------------------------------------
        # 2. Run DropIn Multi-Agent LangGraph Pipeline
        # ------------------------------------------------------------------
        if force_rerun or not multi_done:
            print(f"\n--- [2/2] Running DropIn Multi-Agent Orchestrator for Student ID {sid} ({model}) ---")
            try:
                multi_result = await run_orchestrator(sid, model=model)
            except Exception as e:
                multi_result = {"student_id": sid, "error": str(e)}
            scenario_record["multi_agent"] = multi_result
            logs["scenarios_executed"][sid_key] = scenario_record
            save_logs(log_path, logs)
            print(f"💾 Checkpointed Multi-Agent result for Student ID {sid}.")

            if idx < len(scenarios) and delay_sec > 0:
                print(f"⏳ Pausing {delay_sec}s before next scenario...")
                await asyncio.sleep(delay_sec)

    completed_count = sum(
        1
        for k, v in logs["scenarios_executed"].items()
        if v.get("baseline", {}).get("intervention_plan") and v.get("multi_agent", {}).get("intervention_plan")
    )
    print("\n" + "=" * 80)
    print(f"🏁 STANDARDIZED TESTING RUN FINISHED: {completed_count}/{len(scenarios)} scenarios fully logged.")
    print(f"📂 Raw logs & execution traces saved to: {log_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DropIn Week 9 Standardized Testing across 20 scenarios")
    parser.add_argument("--limit", type=int, default=20, help="Number of scenarios to process (default: 20, use --limit 2 for a quick smoke test)")
    parser.add_argument("--delay", type=float, default=8.0, help="Seconds to pause between pipeline runs to respect Gemini RPM limits")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Matched Gemini model control across both architectures")
    parser.add_argument("--force", action="store_true", help="Force re-running scenarios even if already checkpointed in logs")
    parser.add_argument("--output", type=str, default=DEFAULT_LOG_FILE, help="Path to output JSON log file")
    args = parser.parse_args()

    asyncio.run(
        run_standardized_suite(
            limit=args.limit,
            delay_sec=args.delay,
            force_rerun=args.force,
            log_path=args.output,
            model=args.model,
        )
    )

