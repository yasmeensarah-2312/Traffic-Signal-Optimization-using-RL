"""
train.py

This is the file that actually runs training. It repeats the core loop

    Get state from SUMO -> DQN chooses action -> apply signal action ->
    advance SUMO -> calculate reward -> store experience -> train DQN -> repeat

Run this from inside the rl/ folder:
    python train.py


"""

import csv
import os

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

# ---------------- SETTINGS ----------------
# Rotate through all 6 non-ambulance traffic conditions so the agent
# learns a general policy instead of memorizing one traffic pattern.
# S7_single_ambulance and S8_two_ambulances are intentionally excluded --
# those are reserved for Person 3's ambulance-priority integration later.
SCENARIOS = [
    "S1_normal",
    "S2_equal_opposite",
    "S3_one_side_heavy",
    "S4_all_sides_heavy",
    "S5_changing_traffic",
    "S6_sudden_surge",
]

NUM_EPISODES = 150         # 25 full passes through all 6 scenarios
MAX_STEPS_PER_EPISODE = 500
GUI = False                 # keep the SUMO window off during training -- much faster

# Exploration schedule. Epsilon is computed from NUM_EPISODES so that it
# always reaches EPSILON_END after 80% of the episodes. The last 20% then
# train with almost no random actions. (The old fixed 0.995 decay only
# reached ~0.74 after 60 episodes, so the agent was still ~74% random.)
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_END / EPSILON_START) ** (1.0 / (0.8 * NUM_EPISODES))

# v2 filenames so the original model/log on person2-dqn stay untouched.
MODEL_SAVE_PATH = "../models/dqn_traffic_model_v2.pt"
LOG_SAVE_PATH = "../results/training_log_v2.csv"
# -------------------------------------------


def run_training():
    env = TrafficEnv(scenario_name=SCENARIOS[0], gui=GUI, max_steps=MAX_STEPS_PER_EPISODE)
    agent = DQNAgent(
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_SAVE_PATH), exist_ok=True)

    log_rows = []

    for episode in range(1, NUM_EPISODES + 1):
        # Cycle through scenarios in order so each one gets equal exposure
        # (e.g. with 150 episodes and 6 scenarios, each scenario is used 25 times).
        # NOTE: the numbers logged below are measured WHILE exploring (random
        # actions mixed in), so they show training progress only. To measure
        # how good the learned policy really is, run evaluate.py.
        scenario = SCENARIOS[(episode - 1) % len(SCENARIOS)]
        state = env.reset(scenario_name=scenario)

        total_reward = 0.0
        total_waiting = 0.0
        total_queue = 0.0
        total_throughput = 0
        step_count = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward
            total_waiting += info["waiting_after"]
            total_queue += info["total_queue_length"]
            total_throughput += info["throughput"]
            step_count += 1

        agent.decay_epsilon()

        avg_waiting = total_waiting / max(step_count, 1)
        avg_queue = total_queue / max(step_count, 1)

        print(
            f"Episode {episode}/{NUM_EPISODES} | "
            f"Scenario: {scenario} | "
            f"Reward: {total_reward:.2f} | "
            f"Avg waiting: {avg_waiting:.2f} | "
            f"Avg queue: {avg_queue:.2f} | "
            f"Throughput: {total_throughput} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

        log_rows.append({
            "episode": episode,
            "scenario": scenario,
            "total_reward": total_reward,
            "avg_waiting_time": avg_waiting,
            "avg_queue_length": avg_queue,
            "throughput": total_throughput,
            "epsilon": agent.epsilon,
        })

    env.close()

    # Save the training log -- this CSV is exactly what your PDF says
    # Person 2 should hand to Person 4 (page 7): episode number, reward,
    # avg waiting time, avg queue length, throughput.
    fieldnames = ["episode", "scenario", "total_reward", "avg_waiting_time", "avg_queue_length", "throughput", "epsilon"]
    with open(LOG_SAVE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # Save the trained model
    agent.save(MODEL_SAVE_PATH)

    print(f"\nTraining complete.")
    print(f"Log saved to:   {LOG_SAVE_PATH}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    run_training()
