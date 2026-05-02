import random
from models import Observation, Action, Reward, Patient


class HospitalEREnv:
    def __init__(self):
        self.max_steps = 20
        self.next_patient_id = 0
        self.reset()

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self):
        self.step_count = 0
        self.available_doctors = 2
        self.available_beds = 2
        self.next_patient_id = 0
        self._pending_resource_release = False

        # Episode logging (consumed by graders)
        self.step_log = []       # one entry per step where a patient was treated
        self.stats = {
            "patients_treated": 0,
            "total_patients": 0,   # all patients ever seen (initial + arrivals)
            "total_wait": 0,       # cumulative waiting time of treated patients
            "avg_wait_time": 0.0,
        }

        self.patients = [
            self._create_patient(severity=random.randint(3, 10))
            for _ in range(3)
        ]
        self.stats["total_patients"] += len(self.patients)

        return self._get_obs()

    # -----------------------------
    # STATE
    # -----------------------------
    def state(self):
        return self._get_obs()

    # -----------------------------
    # STEP FUNCTION
    # -----------------------------
    def step(self, action: Action):
        reward = 0.0
        done = False
        release_resources = self._pending_resource_release
        self._pending_resource_release = False
        reward_components = {
            "treatment_reward": 0.0,
            "wait_penalty": 0.0,
            "condition_penalty": 0.0,
        }

        # Apply action
        if action.action_type == "treat":
            # Resource check: need at least one doctor and one bed
            if self.available_doctors <= 0 or self.available_beds <= 0:
                # Treat as a wait — resources unavailable
                reward -= 0.1
                reward_components["wait_penalty"] -= 0.1
            else:
                patient_index = self._resolve_patient_index(action)
                if patient_index is not None:
                    patient = self.patients[patient_index]

                    # Log step data for graders.
                    # priority_score and best_score are injected by the agent after the call.
                    all_severities = [p.severity for p in self.patients]
                    self.step_log.append({
                        "step": self.step_count,
                        "treated_severity": patient.severity,
                        "treated_wait": patient.waiting_time,
                        "all_severities": all_severities,
                        "priority_score": None,   # set by agent post-step
                        "best_score": None,        # set by agent post-step
                    })

                    # Update stats
                    self.stats["patients_treated"] += 1
                    self.stats["total_wait"] += patient.waiting_time

                    # Reward based on severity
                    treatment_reward = patient.severity * 0.1
                    reward += treatment_reward
                    reward_components["treatment_reward"] += treatment_reward

                    # Remove treated patient and free resources
                    self.patients.pop(patient_index)
                    self.available_doctors -= 1
                    self.available_beds -= 1

                    # Resources stay occupied through the next decision step.
                    self._pending_resource_release = True
                else:
                    # Invalid patient id — penalise
                    reward -= 0.1
                    reward_components["wait_penalty"] -= 0.1

        elif action.action_type == "wait":
            reward -= 0.1
            reward_components["wait_penalty"] -= 0.1

        # Release resources that were occupied during this step.
        if release_resources:
            self.available_doctors = min(2, self.available_doctors + 1)
            self.available_beds = min(2, self.available_beds + 1)

        # Update remaining patients
        for p in self.patients:
            p.waiting_time += 1
            p.condition = max(0, p.condition - 1)

            if p.condition <= 3:
                reward -= 0.5
                reward_components["condition_penalty"] -= 0.5

        # Randomly admit a new patient (~30% chance per step)
        if random.random() < 0.3:
            new_patient = self._create_patient(severity=random.randint(1, 10))
            self.patients.append(new_patient)
            self.stats["total_patients"] += 1

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True
            # Finalise avg wait stat
            treated = self.stats["patients_treated"]
            self.stats["avg_wait_time"] = (
                self.stats["total_wait"] / treated if treated > 0 else 0.0
            )

        return self._get_obs(), Reward(score=reward, components=reward_components), done, {}

    # -----------------------------
    # INTERNAL HELPERS
    # -----------------------------
    def _get_obs(self):
        return Observation(
            patients=self.patients,
            available_doctors=self.available_doctors,
            available_beds=self.available_beds,
            step_count=self.step_count,
        )

    def _create_patient(self, severity):
        patient = Patient(
            id=self.next_patient_id,
            severity=severity,
            waiting_time=0,
            condition=10,
        )
        self.next_patient_id += 1
        return patient

    def _resolve_patient_index(self, action: Action):
        if action.patient_id is not None:
            for index, patient in enumerate(self.patients):
                if patient.id == action.patient_id:
                    return index
            return None

        if action.patient_index is not None and 0 <= action.patient_index < len(self.patients):
            return action.patient_index

        return None
