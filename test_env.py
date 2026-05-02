import unittest

from baseline import pick_best_patient
from env import HospitalEREnv
from models import Action, Observation, Patient


class HospitalEREnvTests(unittest.TestCase):
    def setUp(self):
        self.env = HospitalEREnv()
        self.env.reset()

    def test_treatment_holds_resources_for_one_following_step(self):
        self.env.patients = [
            Patient(id=0, severity=9, waiting_time=0, condition=10),
            Patient(id=1, severity=4, waiting_time=0, condition=10),
        ]
        self.env.next_patient_id = 2

        obs, _, _, _ = self.env.step(Action(action_type="treat", patient_id=0))
        self.assertEqual(obs.available_doctors, 1)
        self.assertEqual(obs.available_beds, 1)

        obs, _, _, _ = self.env.step(Action(action_type="wait"))
        self.assertEqual(obs.available_doctors, 2)
        self.assertEqual(obs.available_beds, 2)

    def test_invalid_treatment_does_not_create_step_log_entry(self):
        self.env.patients = [Patient(id=0, severity=6, waiting_time=0, condition=10)]
        self.env.next_patient_id = 1

        _, reward, _, _ = self.env.step(Action(action_type="treat", patient_id=999))
        self.assertEqual(len(self.env.step_log), 0)
        self.assertLess(reward.score, 0)

    def test_pick_best_patient_waits_when_resources_are_unavailable(self):
        observation = Observation(
            patients=[Patient(id=7, severity=10, waiting_time=0, condition=10)],
            available_doctors=0,
            available_beds=1,
            step_count=0,
        )

        self.assertEqual(pick_best_patient(observation), (None, None, None))

    def test_wait_action_rejects_patient_targets(self):
        with self.assertRaises(ValueError):
            Action(action_type="wait", patient_id=0)


if __name__ == "__main__":
    unittest.main()
