from env import HospitalEREnv
from models import Action
from baseline import pick_best_patient

import random

# Q-table
Q = {}

# -----------------------------
# STATE REPRESENTATION
# -----------------------------
def get_state(obs):
    if not obs.patients:
        return ("empty",)

    max_severity = max(p.severity for p in obs.patients)
    num_patients = len(obs.patients)
    doctors = obs.available_doctors
    beds = obs.available_beds

    return (max_severity, num_patients, doctors, beds)


# -----------------------------
# ACTION SPACE
# -----------------------------
def get_possible_actions(obs):
    if not obs.patients:
        return ["wait"]
    if obs.available_doctors <= 0 or obs.available_beds <= 0:
        return ["wait"]
    return ["treat", "wait"]


# -----------------------------
# EPSILON-GREEDY POLICY
# -----------------------------
def choose_action(state, actions, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(actions)

    q_values = [Q.get((state, a), 0) for a in actions]
    return actions[q_values.index(max(q_values))]


# -----------------------------
# Q-LEARNING UPDATE
# -----------------------------
def update_q(state, action, reward, next_state, next_actions, alpha=0.1, gamma=0.9):
    old_q = Q.get((state, action), 0)
    future_q = max([Q.get((next_state, a), 0) for a in next_actions], default=0)

    Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)


def build_action(obs, action_type):
    if action_type != "treat":
        return Action(action_type="wait")

    patient_id, _, _ = pick_best_patient(obs)
    if patient_id is None:
        return Action(action_type="wait")

    return Action(action_type="treat", patient_id=patient_id)


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_rl(episodes=50):
    env = HospitalEREnv()

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            state = get_state(obs)
            actions = get_possible_actions(obs)

            action_type = choose_action(state, actions)

            # Convert to environment action
            action = build_action(obs, action_type)

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward.score

            next_state = get_state(next_obs)
            next_actions = get_possible_actions(next_obs)

            update_q(state, action_type, reward.score, next_state, next_actions)

            obs = next_obs

        print(f"Episode {episode+1}: Reward = {total_reward:.3f}")

    print("\nTraining complete.\n")


# -----------------------------
# TESTING LOOP (NO EXPLORATION)
# -----------------------------
def test_rl():
    env = HospitalEREnv()
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = get_state(obs)
        actions = get_possible_actions(obs)

        # No exploration (epsilon = 0)
        q_values = [Q.get((state, a), 0) for a in actions]
        action_type = actions[q_values.index(max(q_values))]

        action = build_action(obs, action_type)

        obs, reward, done, _ = env.step(action)
        total_reward += reward.score

    print(f"Test Reward: {total_reward:.3f}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_rl(episodes=50)
    test_rl()
