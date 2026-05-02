def compute_priority_score(severity, waiting_time, condition,
                           max_severity=10, max_wait=20, max_condition=10):
    """
    Canonical priority score shared by the baseline agent and the medium grader.

    Combines three normalised signals:
        - severity       (weight 0.5) — higher is more urgent
        - waiting_time   (weight 0.3) — longer wait deserves more attention
        - condition_risk (weight 0.2) — lower condition means closer to critical

    Returns a float in [0.0, 1.0].
    """
    def norm(v, mx):
        return max(0.0, min(1.0, v / mx)) if mx > 0 else 0.0

    severity_norm  = norm(severity, max_severity)
    waiting_norm   = norm(waiting_time, max_wait)
    condition_risk = norm(max_condition - condition, max_condition)

    return (
        severity_norm  * 0.5
        + waiting_norm * 0.3
        + condition_risk * 0.2
    )


def grade_medium(step_log, tolerance=0.05):
    """
    Medium task: did the agent pick the patient with the highest priority score?

    Evaluates relative optimality rather than an absolute threshold, because
    priority scores near the start of an episode are inherently low (waiting=0,
    condition=10 for all new patients) and would fail any fixed cutoff.

    Each entry in step_log must contain:
        - "priority_score" : float — score of the patient who WAS treated
        - "best_score"     : float — score of the best available patient that step

    The agent passes a step if its chosen score is within `tolerance` of the best
    available score (default: 5 percentage points).

    Returns a score in [0.0, 1.0].
    """
    if not step_log:
        return 0.0

    scored_steps = [
        s for s in step_log
        if s.get("priority_score") is not None and s.get("best_score") is not None
    ]
    if not scored_steps:
        return 0.0

    good_decisions = sum(
        1 for s in scored_steps
        if s["priority_score"] >= s["best_score"] - tolerance
    )
    return good_decisions / len(scored_steps)