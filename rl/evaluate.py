"""
evaluate.py

Measures how good the TRAINED model really is.

The numbers printed during training (train.py) are measured while the agent
is still taking random exploration actions, so they only show training
progress. This script loads the saved model and runs it with exploration
completely switched off (greedy=True) on every scenario. It also runs a
purely random policy on the same scenarios, so you can see whether the
learned policy actually beats "no intelligence at all".

Run from inside the rl/ folder:
    python evaluate.py

Output: ../results/evaluation_results.csv  (same metric columns as the
training log, so Person 4 can put it next to the fixed-time baseline).

To evaluate the ORIGINAL model instead, change MODEL_PATH to
"../models/dqn_traffic_model.pt".
"""

import csv
import os
import random

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

# ---------------- SETTINGS ----------------
SCENARIOS = [
    "S1_normal",
    "S2_equal_opposite",
    "S3_one_side_heavy",
    "S4_all_sides_heavy",
    "S5_changing_traffic",
    "S6_sudden_surge",
]

MODEL_PATH = "../models/dqn_traffic_model_v2.pt"
OUTPUT_CSV = "../results/evaluation_results.csv"

MAX_STEPS_PER_EPISODE = 500   # keep identical to train.py so numbers are comparable
GUI = False

# The SUMO scenarios run the same way each time, so a greedy policy gives
# the same result on every run: one run per scenario is enough for the DQN.
# The random policy varies, so it gets a few runs.
DQN_RUNS = 1
RANDOM_RUNS = 3
# -------------------------------------------


def run_episode(env, scenario, choose_action):
    """Plays one full episode with the given action-chooser, no learning."""
    state = env.reset(scenario_name=scenario)

    total_reward = 0.0
    total_waiting = 0.0
    total_queue = 0.0
    total_throughput = 0
    steps = 0
    done = False

    while not done:
        action = choose_action(state)
        state, reward, done, info = env.step(action)

        total_reward += reward
        total_waiting += info["waiting_after"]
        total_queue += info["total_queue_length"]
        total_throughput += info["throughput"]
        steps += 1

    return {
        "total_reward": total_reward,
        "avg_waiting_time": total_waiting / max(steps, 1),
        "avg_queue_length": total_queue / max(steps, 1),
        "throughput": total_throughput,
    }


def run_evaluation():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        print("Train first (python train.py) or point MODEL_PATH at an existing model.")
        return

    agent = DQNAgent()
    agent.load(MODEL_PATH)

    env = TrafficEnv(scenario_name=SCENARIOS[0], gui=GUI, max_steps=MAX_STEPS_PER_EPISODE)

    policies = [
        # greedy=True -> always the network's best action, no random exploration
        ("dqn_greedy", lambda s: agent.select_action(s, greedy=True), DQN_RUNS),
        ("random", lambda s: random.randrange(agent.action_size), RANDOM_RUNS),
    ]

    rows = []

    for policy_name, choose_action, runs in policies:
        for scenario in SCENARIOS:
            for run in range(1, runs + 1):
                result = run_episode(env, scenario, choose_action)
                result.update({"policy": policy_name, "scenario": scenario, "run": run})
                rows.append(result)
                print(
                    f"{policy_name:10s} | {scenario:20s} | run {run} | "
                    f"Avg waiting: {result['avg_waiting_time']:.2f} | "
                    f"Avg queue: {result['avg_queue_length']:.2f} | "
                    f"Throughput: {result['throughput']}"
                )

    env.close()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["policy", "scenario", "run", "total_reward",
                  "avg_waiting_time", "avg_queue_length", "throughput"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Summary: learned policy vs random policy, per scenario ----
    print("\n=========== SUMMARY (average waiting time, lower is better) ===========")
    print(f"{'Scenario':22s} {'Random':>10s} {'DQN':>10s} {'Change':>10s}")
    for scenario in SCENARIOS:
        random_vals = [r["avg_waiting_time"] for r in rows
                       if r["policy"] == "random" and r["scenario"] == scenario]
        dqn_vals = [r["avg_waiting_time"] for r in rows
                    if r["policy"] == "dqn_greedy" and r["scenario"] == scenario]
        random_avg = sum(random_vals) / len(random_vals)
        dqn_avg = sum(dqn_vals) / len(dqn_vals)
        change = (dqn_avg - random_avg) / random_avg * 100 if random_avg else 0.0
        print(f"{scenario:22s} {random_avg:10.2f} {dqn_avg:10.2f} {change:+9.1f}%")

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_evaluation()