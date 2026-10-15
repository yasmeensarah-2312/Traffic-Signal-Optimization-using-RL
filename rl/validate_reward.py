"""
validate_reward.py

Checks that the reward is aligned with real traffic performance.

A good reward must rank policies the same way the measured traffic metrics
do. So this script plays several policies of clearly different quality on
every scenario and, for each episode, records:

  * return_new      total reward with the NEW queue-based reward (what the
                    environment gives now)
  * return_old      what the OLD reward would have given (change in total
                    waiting time between the start and end of each green
                    phase), rebuilt from info["waiting_before"/"waiting_after"]
  * avg_waiting_time, avg_queue_length, throughput   the measured metrics

Then it checks how strongly each return agrees with the measured metrics.
A reward aligned with the goal should be:
    NEGATIVELY correlated with avg waiting time and avg queue length
    POSITIVELY correlated with throughput

Run from inside the rl/ folder:
    python validate_reward.py

Output: ../results/reward_validation.csv
"""

import csv
import os
import random

import numpy as np

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

MODEL_PATH = "../models/dqn_traffic_model_v2.pt"   # optional: used only if the file exists
OUTPUT_CSV = "../results/reward_validation.csv"
MAX_STEPS_PER_EPISODE = 500
GUI = False
RANDOM_RUNS = 2        # the random policy varies, so it gets a couple of runs
# -------------------------------------------


def play(env, scenario, choose_action):
    """Plays one full episode, no learning, and returns its summary numbers."""
    state = env.reset(scenario_name=scenario)

    return_new = 0.0
    return_old = 0.0
    total_waiting = 0.0
    total_queue = 0.0
    total_throughput = 0
    steps = 0
    done = False

    while not done:
        action = choose_action(state, steps)
        state, reward, done, info = env.step(action)

        return_new += reward
        return_old += info["waiting_before"] - info["waiting_after"]
        total_waiting += info["waiting_after"]
        total_queue += info["total_queue_length"]
        total_throughput += info["throughput"]
        steps += 1

    return {
        "return_new": return_new,
        "return_old": return_old,
        "avg_waiting_time": total_waiting / max(steps, 1),
        "avg_queue_length": total_queue / max(steps, 1),
        "throughput": total_throughput,
    }


def zscore_by_scenario(rows, key):
    """
    Scenarios have very different scales (S4 is far busier than S6), so we
    standardise each metric inside its own scenario before pooling.
    """
    values = np.zeros(len(rows))
    for scenario in SCENARIOS:
        idx = [i for i, r in enumerate(rows) if r["scenario"] == scenario]
        if not idx:
            continue
        v = np.array([rows[i][key] for i in idx], dtype=float)
        std = v.std()
        values[idx] = (v - v.mean()) / std if std > 0 else 0.0
    return values


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    env = TrafficEnv(scenario_name=SCENARIOS[0], gui=GUI, max_steps=MAX_STEPS_PER_EPISODE)

    # Policies of clearly different quality, so the reward has a range to rank.
    policies = [
        ("stuck_on_one_phase", lambda s, t: 0, 1),                    # starves 3 directions: bad
        ("random", lambda s, t: random.randrange(4), RANDOM_RUNS),    # no intelligence
        ("round_robin", lambda s, t: t % 4, 1),                       # cycles all phases: fair
    ]

    if os.path.exists(MODEL_PATH):
        agent = DQNAgent()
        agent.load(MODEL_PATH)
        policies.append(("dqn_greedy", lambda s, t: agent.select_action(s, greedy=True), 1))
    else:
        print(f"(No model at {MODEL_PATH} - skipping the DQN policy.)")

    rows = []
    for scenario in SCENARIOS:
        for policy_name, choose_action, runs in policies:
            for run in range(1, runs + 1):
                result = play(env, scenario, choose_action)
                result.update({"scenario": scenario, "policy": policy_name, "run": run})
                rows.append(result)
                print(
                    f"{scenario:20s} | {policy_name:18s} | "
                    f"wait {result['avg_waiting_time']:8.2f} | "
                    f"queue {result['avg_queue_length']:6.2f} | "
                    f"thru {result['throughput']:4d} | "
                    f"new return {result['return_new']:9.2f} | "
                    f"old return {result['return_old']:9.2f}"
                )

    env.close()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["scenario", "policy", "run", "return_new", "return_old",
                  "avg_waiting_time", "avg_queue_length", "throughput"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Does each return agree with the measured traffic metrics? ----
    z = {k: zscore_by_scenario(rows, k) for k in
         ["return_new", "return_old", "avg_waiting_time", "avg_queue_length", "throughput"]}

    print("\n=========== REWARD VALIDATION ===========")
    print("Correlation of episode return with measured performance")
    print("(pooled over all scenarios, each scenario standardised first)")
    print("Aligned reward  ->  NEGATIVE vs waiting/queue,  POSITIVE vs throughput\n")
    print(f"{'Reward':14s} {'vs waiting':>12s} {'vs queue':>12s} {'vs throughput':>15s}")
    for label, key in [("NEW (queue)", "return_new"), ("OLD (delta wait)", "return_old")]:
        print(
            f"{label:14s} "
            f"{correlation(z[key], z['avg_waiting_time']):12.2f} "
            f"{correlation(z[key], z['avg_queue_length']):12.2f} "
            f"{correlation(z[key], z['throughput']):15.2f}"
        )

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
