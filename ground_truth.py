import json
import sqlite3
from typing import Dict, Any, List

DB_NAME = "dropin_synthetic.db"
SCENARIOS_FILE = "test_scenarios.json"


def compute_ground_truth_labels(student: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies deterministic ground-truth criteria to a student record to establish
    an objective benchmark for risk classification and primary risk factor identification.

    Predefined Criteria:
    1. Urgency / Risk Level:
       - High   : GPA < 2.0 OR Attendance < 65% OR Missed Assignments >= 6
       - Medium : (2.0 <= GPA < 2.7) OR (65% <= Attendance < 80%) OR (3 <= Missed Assignments <= 5)
       - Low    : GPA >= 2.7 AND Attendance >= 80% AND Missed Assignments < 3
    2. At-Risk Status (is_at_risk):
       - True if High or Medium urgency; False if Low urgency.
    3. Primary Risk Factor:
       - For at-risk students, determined by normalizing the severity of each deficit
         (GPA deficit below 2.7, Attendance deficit below 80%, Missed Assignments above 2)
         and selecting the highest-severity factor.
    """
    gpa = float(student["gpa"])
    attendance = int(student["attendance_percentage"])
    missed = int(student["missed_assignments"])

    # 1. Determine Ground-Truth Urgency Level
    if gpa < 2.0 or attendance < 65 or missed >= 6:
        urgency = "High"
        is_at_risk = True
    elif gpa < 2.7 or attendance < 80 or missed >= 3:
        urgency = "Medium"
        is_at_risk = True
    else:
        urgency = "Low"
        is_at_risk = False

    # 2. Determine Ground-Truth Primary Risk Factor & Acceptable Factors
    if not is_at_risk:
        primary_factor = "None"
        acceptable_factors = ["None"]
    else:
        # Compute normalized severity scores for each deficit domain
        # - GPA deficit below 2.7 (scaled by 1.2 points)
        gpa_severity = max(0.0, (2.7 - gpa) / 1.2)
        # - Attendance deficit below 80% (scaled by 25 percentage points)
        attendance_severity = max(0.0, (80.0 - attendance) / 25.0)
        # - Missed assignments deficit above 2 (scaled by 5 assignments)
        missed_severity = max(0.0, (missed - 2.0) / 5.0)

        severities = {
            "Poor Grades": round(gpa_severity, 3),
            "Low Attendance": round(attendance_severity, 3),
            "Missed Assignments": round(missed_severity, 3),
        }

        # Primary factor is the highest severity score
        primary_factor = max(severities, key=severities.get)

        # Any factor that crosses its individual risk threshold is also recorded as a valid secondary factor
        acceptable_factors = []
        if gpa < 2.7:
            acceptable_factors.append("Poor Grades")
        if attendance < 80:
            acceptable_factors.append("Low Attendance")
        if missed >= 3:
            acceptable_factors.append("Missed Assignments")

    # 3. Required Intervention Themes based on active deficits
    required_themes = []
    if not is_at_risk:
        required_themes.append("enrichment_or_maintenance")
    else:
        if gpa < 2.7:
            required_themes.append("academic_tutoring_or_study_support")
        if attendance < 80:
            required_themes.append("attendance_check_in_or_schedule_review")
        if missed >= 3:
            required_themes.append("assignment_makeup_or_time_management")

    return {
        "ground_truth_is_at_risk": is_at_risk,
        "ground_truth_urgency": urgency,
        "ground_truth_primary_factor": primary_factor,
        "acceptable_risk_factors": acceptable_factors,
        "required_intervention_themes": required_themes,
    }


def select_standardized_scenarios(total_count: int = 20) -> List[Dict[str, Any]]:
    """
    Selects a deterministic, stratified sample of 20 students across High, Medium, and Low
    ground-truth risk levels from dropin_synthetic.db.
    Target stratification: 7 High, 7 Medium, 6 Low (or closest available).
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students ORDER BY student_id ASC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()

    buckets: Dict[str, List[Dict[str, Any]]] = {"High": [], "Medium": [], "Low": []}

    for row in rows:
        gt = compute_ground_truth_labels(row)
        enriched = {
            "student_id": row["student_id"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "major": row["major"],
            "gpa": row["gpa"],
            "attendance_percentage": row["attendance_percentage"],
            "missed_assignments": row["missed_assignments"],
            "db_risk_level": row["risk_level"],
            "academic_notes": row["academic_notes"],
            **gt,
        }
        buckets[gt["ground_truth_urgency"]].append(enriched)

    # Stratified selection: 7 High, 7 Medium, 6 Low
    target_quotas = {"High": 7, "Medium": 7, "Low": 6}
    selected: List[Dict[str, Any]] = []
    selected_ids = set()

    for urgency, quota in target_quotas.items():
        candidates = buckets[urgency][:quota]
        for c in candidates:
            selected.append(c)
            selected_ids.add(c["student_id"])

    # If any bucket had fewer than its quota, backfill from remaining students
    if len(selected) < total_count:
        for urgency in ("High", "Medium", "Low"):
            for c in buckets[urgency]:
                if c["student_id"] not in selected_ids:
                    selected.append(c)
                    selected_ids.add(c["student_id"])
                    if len(selected) == total_count:
                        break
            if len(selected) == total_count:
                break

    selected.sort(key=lambda x: x["student_id"])
    return selected


def main():
    scenarios = select_standardized_scenarios(20)
    with open(SCENARIOS_FILE, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)

    counts = {"High": 0, "Medium": 0, "Low": 0}
    for s in scenarios:
        counts[s["ground_truth_urgency"]] += 1

    print("=" * 75)
    print(f"✅ GENERATED {len(scenarios)} STANDARDIZED GROUND-TRUTH SCENARIOS -> {SCENARIOS_FILE}")
    print("=" * 75)
    print(f"Stratified Distribution: High={counts['High']}, Medium={counts['Medium']}, Low={counts['Low']}\n")
    print(f"{'ID':<4} | {'Name':<20} | {'GPA':<4} | {'Att%':<4} | {'Miss':<4} | {'GT Urgency':<10} | {'GT Primary Factor'}")
    print("-" * 75)
    for s in scenarios:
        full_name = f"{s['first_name']} {s['last_name']}"[:20]
        print(
            f"{s['student_id']:<4} | {full_name:<20} | {s['gpa']:<4.2f} | "
            f"{s['attendance_percentage']:<4} | {s['missed_assignments']:<4} | "
            f"{s['ground_truth_urgency']:<10} | {s['ground_truth_primary_factor']}"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()

