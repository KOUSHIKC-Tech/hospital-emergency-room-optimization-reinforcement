from env import HospitalEREnv
from models import Action
from tasks.easy import grade_easy
from tasks.medium import grade_medium, compute_priority_score
from tasks.hard import grade_hard

CRITICAL_SEVERITY = 8
MAX_SEVERITY = 10
MAX_CONDITION = 10
DEFAULT_MAX_WAIT = 20

# Tunable baseline weights applied to normalised values.
SEVERITY_WEIGHT       = 0.5
WAITING_WEIGHT        = 0.3
CONDITION_RISK_WEIGHT = 0.2
QUEUE_PRESSURE_WEIGHT = 0.1


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize(value, max_value):
    if max_value <= 0:
        return 0.0
    return clamp(value / max_value, 0.0, 1.0)


def pick_best_patient(observation, max_wait=DEFAULT_MAX_WAIT):
    """
    Returns (patient_id, chosen_priority_score, best_available_score).
    Returns (None, None, None) if no treatment should happen this step.

    chosen_priority_score and best_available_score are on the compute_priority_score
    scale so the medium grader can do a relative optimality check.
    """
    patients = observation.patients

    # Pre-compute canonical priority scores for all patients (used by medium grader).
    all_priority_scores = {
        p.id: compute_priority_score(
            p.severity, p.waiting_time, p.condition, max_wait=max_wait
        )
        for p in patients
    }
    best_available_score = max(all_priority_scores.values()) if all_priority_scores else 0.0

    # If no doctor or bed is available, waiting is the only valid option.
    if observation.available_doctors <= 0 or observation.available_beds <= 0:
        return None, None, None

    # Treat critical patients immediately.
    critical_patients = [p for p in patients if p.severity >= CRITICAL_SEVERITY]
    if critical_patients:
        best = max(
            critical_patients,
            key=lambda p: (p.severity, p.waiting_time, -p.condition),
        )
        return best.id, all_priority_scores[best.id], best_available_score

    # Weighted priority score across all patients.
    best_patient_id = None
    best_score = float("-inf")
    queue_pressure = normalize(len(patients), max(observation.available_beds, 1))

    for p in patients:
        severity_norm  = normalize(p.severity, MAX_SEVERITY)
        waiting_norm   = normalize(p.waiting_time, max_wait)
        condition_risk = normalize(MAX_CONDITION - p.condition, MAX_CONDITION)

        score = (
            severity_norm  * SEVERITY_WEIGHT
            + waiting_norm * WAITING_WEIGHT
            + condition_risk * CONDITION_RISK_WEIGHT
            + queue_pressure * QUEUE_PRESSURE_WEIGHT
        )

        if score > best_score:
            best_score = score
            best_patient_id = p.id

    chosen_score = all_priority_scores.get(best_patient_id, 0.0)
    return best_patient_id, chosen_score, best_available_score


def run():
    env = HospitalEREnv()
    obs = env.reset()
    max_wait = env.max_steps

    done = False
    total_reward = 0.0

    while not done:
        if obs.patients:
            patient_id, priority_score, best_score = pick_best_patient(obs, max_wait=max_wait)

            if patient_id is None:
                action = Action(action_type="wait")
                priority_score = None
                best_score = None
            else:
                action = Action(action_type="treat", patient_id=patient_id)
        else:
            action = Action(action_type="wait")
            priority_score = None
            best_score = None

        previous_log_len = len(env.step_log)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.score

        # Inject grader fields only when the current step created a new treatment log entry.
        if priority_score is not None and len(env.step_log) > previous_log_len:
            env.step_log[-1]["priority_score"] = priority_score
            env.step_log[-1]["best_score"]     = best_score

    # -----------------------------------------------
    # Grader evaluation
    # -----------------------------------------------
    easy_score   = grade_easy(env.step_log)
    medium_score = grade_medium(env.step_log)
    hard_score   = grade_hard(env.stats)

    print("=" * 40)
    print(f"  Total Reward  : {total_reward:.3f}")
    print("-" * 40)
    print(f"  Easy Score    : {easy_score:.3f}   (severity-first accuracy)")
    print(f"  Medium Score  : {medium_score:.3f}   (priority-score optimality)")
    print(f"  Hard Score    : {hard_score:.3f}   (survival + wait efficiency)")
    print("=" * 40)

    return {
        "total_reward": total_reward,
        "easy":         easy_score,
        "medium":       medium_score,
        "hard":         hard_score,
    }


if __name__ == "__main__":
    run()
