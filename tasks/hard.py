def grade_hard(stats):
    """
    Hard task: overall ER performance.

    Expects a stats dict produced by HospitalEREnv at episode end:
        - "patients_treated": int   — how many patients received treatment
        - "total_patients":   int   — all patients seen (initial + arrivals)
        - "avg_wait_time":    float — mean waiting time of treated patients (in steps)

    Scoring:
        survival_rate    — fraction of patients treated              (range 0–1)
        wait_penalty     — 0.05 per step of average wait             (range 0–1+)

    Final score is clamped to [0.0, 1.0].
    """
    patients_treated = stats.get("patients_treated", 0)
    total_patients   = max(1, stats.get("total_patients", 1))
    avg_wait         = stats.get("avg_wait_time", 0.0)

    survival_rate = patients_treated / total_patients
    wait_penalty  = avg_wait * 0.05

    score = survival_rate - wait_penalty
    return max(0.0, min(1.0, score))