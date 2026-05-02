# 🏥 Hospital ER Optimization using Reinforcement Learning

## 📌 Overview

This project simulates an **Emergency Room (ER)** environment and applies **Reinforcement Learning (RL)** to optimize patient treatment decisions under resource constraints.

The system compares:

* A **rule-based baseline agent**
* A **learning-based RL agent**

The goal is to improve:

* Patient survival
* Waiting time
* Resource utilization

---

## ⚙️ Features

* 🚑 Dynamic patient arrival (stochastic environment)
* 🧑‍⚕️ Limited resources (doctors & beds)
* ⏳ Patient condition deterioration over time
* 🎯 Reward-based decision making
* 📊 Multi-level evaluation metrics

---

## 🧠 Problem Formulation

This problem is modeled as a **Markov Decision Process (MDP)**:

* **State (Observation)**

  * Patients (severity, waiting time, condition)
  * Available doctors
  * Available beds

* **Actions**

  * `treat_patient`
  * `wait`

* **Reward**

  * Positive reward for treating patients
  * Penalty for waiting
  * Penalty for patient condition worsening

---

## 📁 Project Structure

```
hospital-er-env/
│
├── env.py              # Environment simulation
├── models.py           # Data models (Patient, Action, Reward)
├── baseline.py         # Rule-based agent
├── rl_agent.py         # Reinforcement Learning agent
├── tasks/
│   ├── easy.py
│   ├── medium.py
│   └── hard.py
└── README.md
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install pydantic
```

---

### 2️⃣ Run baseline agent

```bash
python baseline.py
```

Example output:

```
Total Reward  : 4.400
Easy Score    : 1.000
Medium Score  : 1.000
Hard Score    : 0.870
```

---

### 3️⃣ Run RL agent

```bash
python rl_agent.py
```

Example output:

```
Episode 1: Reward = 2.300
...
Training complete.

Test Reward: 4.200
```

---

## 📊 Evaluation Metrics

The system evaluates performance using three levels:

### ✅ Easy Score

* Measures **correct prioritization of severity**

### ⚖️ Medium Score

* Measures **optimal decision quality**

### 🧩 Hard Score

* Measures **overall system performance**

  * Patient survival
  * Waiting efficiency

---

## 📈 Results

| Agent    | Easy | Medium | Hard  | Reward |
| -------- | ---- | ------ | ----- | ------ |
| Baseline | 1.00 | 1.00   | ~0.87 | ~4.4   |
| RL Agent | ~0.9 | ~0.8   | ~0.85 | ~4.2   |

> 🔍 Observation:
> The baseline performs strongly due to optimized heuristics.
> RL requires further improvement to surpass it.

---

## 🧪 Key Insights

* Strong baselines are hard to beat
* Reward design heavily impacts learning
* RL struggles with limited state representation
* Greedy policies can outperform weak RL setups

---

## 🔮 Future Improvements

* Allow RL to **choose specific patients**
* Improve **state representation**
* Use **Deep Q-Learning (DQN)**
* Tune reward scaling
* Add more realistic hospital constraints

---

## 🛠️ Technologies Used

* Python
* Reinforcement Learning (Q-learning)
* Pydantic (data validation)

---

## 👨‍💻 Author

Developed as part of learning **Reinforcement Learning and decision systems**.

KOUSHIK C
MANJUNATH V
GOWTHAM L

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
