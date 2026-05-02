def grade_easy(step_log):
    """
    Easy task: did the agent always treat the highest-severity patient?

    Each entry in step_log must contain:
        - "treated_severity": int   — severity of the patient who was treated
        - "all_severities": list    — severities of ALL patients present that step

    Returns a score in [0.0, 1.0].
    """
    if not step_log:
        return 0.0

    correct = 0
    for step in step_log:
        if step["treated_severity"] == max(step["all_severities"]):
            correct += 1

    return correct / len(step_log)