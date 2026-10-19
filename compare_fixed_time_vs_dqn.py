"""Compare SUMO's fixed-time signal program with trained DQN V2.

Run from the project root:
    python compare_fixed_time_vs_dqn.py

Writes a new CSV and graphs under results/; does not modify existing results.
Both policies receive a 20-step warm-up (excluded from metrics) and are measured
for up to 500 simulation steps after warm-up, using the same step-wise metrics.
"""
import csv
import os
import sys
import traci

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMO_INTERFACE_DIR = os.path.join(PROJECT_ROOT, "simulation", "sumo")
RL_DIR = os.path.join(PROJECT_ROOT, "rl")
for path in (SUMO_INTERFACE_DIR, RL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import traffic_interface as ti
from traffic_env import TrafficEnv
from dqn_agent import DQNAgent

SCENARIOS = [
    "S1_normal", "S2_equal_opposite", "S3_one_side_heavy",
    "S4_all_sides_heavy", "S5_changing_traffic", "S6_sudden_surge",
]
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dqn_traffic_model_v2.pt")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "fixed_vs_dqn_v2.csv")
WARMUP_STEPS = 20
MAX_MEASURE_STEPS = 500
GUI = False
GREEN_PHASES = [0, 2, 4, 6]
YELLOW_FOR_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}
YELLOW_DURATION = 4
GREEN_DURATION = 10
LANES = {
    "north": ["n_t_0", "n_t_1"],
    "south": ["s_t_0", "s_t_1"],
    "east": ["e_t_0", "e_t_1"],
    "west": ["w_t_0", "w_t_1"],
}


def close_traci():
    try:
        traci.close()
    except Exception:
        pass


def step_metrics():
    """One simulation-step snapshot using the original baseline definitions."""
    stopped_vehicles = 0.0
    delay_sum = 0.0
    for vehicle_id in traci.vehicle.getIDList():
        speed = traci.vehicle.getSpeed(vehicle_id)
        if speed < 0.1:
            stopped_vehicles += 1.0
        allowed_speed = traci.vehicle.getAllowedSpeed(vehicle_id)
        if allowed_speed > 0:
            delay_sum += max(0.0, 1.0 - speed / allowed_speed)

    queue_length = 0
    for direction_lanes in LANES.values():
        for lane_id in direction_lanes:
            try:
                queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                pass

    return stopped_vehicles, queue_length, delay_sum, traci.simulation.getArrivedNumber()


def add_snapshot(totals):
    stopped, queue, delay, arrived = step_metrics()
    totals["stopped_vehicle_steps"] += stopped
    totals["queue_sum"] += queue
    totals["delay_sum"] += delay
    totals["throughput"] += arrived
    totals["steps"] += 1


def make_result(policy, scenario, totals):
    steps = totals["steps"]
    return {
        "policy": policy,
        "scenario": scenario,
        # This is stopped-vehicle count accumulated over steps, not seconds.
        "total_stopped_vehicle_steps": round(totals["stopped_vehicle_steps"], 2),
        "average_stopped_vehicles_per_step": round(totals["stopped_vehicle_steps"] / steps, 4) if steps else 0.0,
        "average_queue_length": round(totals["queue_sum"] / steps, 4) if steps else 0.0,
        "throughput": int(totals["throughput"]),
        "average_delay": round(totals["delay_sum"] / steps, 4) if steps else 0.0,
        "simulation_steps": steps,
    }


def empty_totals():
    return {"stopped_vehicle_steps": 0.0, "queue_sum": 0.0,
            "delay_sum": 0.0, "throughput": 0, "steps": 0}


def run_fixed_time(scenario):
    """Run SUMO's configured signal program without changing its phases."""
    close_traci()
    ti.start_simulation(scenario, gui=GUI)
    totals = empty_totals()
    try:
        for _ in range(WARMUP_STEPS):
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
        while totals["steps"] < MAX_MEASURE_STEPS and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            add_snapshot(totals)
        return make_result("fixed_time", scenario, totals)
    finally:
        close_traci()


def run_dqn(env, agent, scenario):
    """Run DQN actions while sampling metrics after every SUMO simulation step."""
    close_traci()
    state = env.reset(scenario_name=scenario)  # performs the 20-step warm-up
    totals = empty_totals()
    max_total_steps = WARMUP_STEPS + MAX_MEASURE_STEPS

    def advance_and_measure(number_of_steps):
        advanced = 0
        for _ in range(number_of_steps):
            if totals["steps"] >= MAX_MEASURE_STEPS:
                break
            if traci.simulation.getMinExpectedNumber() <= 0:
                break
            traci.simulationStep()
            env.current_step += 1
            add_snapshot(totals)
            advanced += 1
        return advanced

    try:
        done = traci.simulation.getMinExpectedNumber() <= 0
        while not done and totals["steps"] < MAX_MEASURE_STEPS:
            action = agent.select_action(state, greedy=True)
            target_phase = GREEN_PHASES[action]
            current_phase = ti.get_current_phase()

            if current_phase in GREEN_PHASES and current_phase != target_phase:
                ti.change_traffic_light_phase(YELLOW_FOR_GREEN[current_phase])
                advanced = advance_and_measure(YELLOW_DURATION)
                if advanced == 0 or totals["steps"] >= MAX_MEASURE_STEPS:
                    break
                if traci.simulation.getMinExpectedNumber() <= 0:
                    break

            ti.change_traffic_light_phase(target_phase)
            advanced = advance_and_measure(GREEN_DURATION)
            if advanced == 0:
                break

            done = (
                traci.simulation.getMinExpectedNumber() <= 0
                or env.current_step >= max_total_steps
                or totals["steps"] >= MAX_MEASURE_STEPS
            )
            if not done:
                state = env._get_state_vector()
        return make_result("dqn_greedy", scenario, totals)
    finally:
        env.close()


def write_csv(rows):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fields = ["policy", "scenario", "total_stopped_vehicle_steps",
              "average_stopped_vehicles_per_step", "average_queue_length",
              "throughput", "average_delay", "simulation_steps"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_graphs(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is missing; CSV saved, graphs skipped. Install with: python -m pip install matplotlib")
        return

    metrics = [
        ("average_queue_length", "Average queue length", "Lower is better", "average_queue"),
        ("average_stopped_vehicles_per_step", "Average stopped vehicles per step", "Lower is better", "average_stopped_vehicles"),
        ("throughput", "Throughput (vehicles arrived)", "Higher is generally better", "throughput"),
        ("average_delay", "Approximate delay score", "Lower is better", "average_delay"),
    ]
    x = list(range(len(SCENARIOS)))
    width = 0.36
    for field, title, note, suffix in metrics:
        fixed_values, dqn_values = [], []
        for scenario in SCENARIOS:
            fixed = next((r for r in rows if r["scenario"] == scenario and r["policy"] == "fixed_time"), None)
            dqn = next((r for r in rows if r["scenario"] == scenario and r["policy"] == "dqn_greedy"), None)
            fixed_values.append(float(fixed[field]) if fixed else 0.0)
            dqn_values.append(float(dqn[field]) if dqn else 0.0)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar([i - width / 2 for i in x], fixed_values, width, label="Fixed-time")
        ax.bar([i + width / 2 for i in x], dqn_values, width, label="DQN V2")
        ax.set_title(f"{title}\n{note}")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, rotation=25, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        graph_path = os.path.join(PROJECT_ROOT, "results", f"fixed_vs_dqn_v2_{suffix}.png")
        fig.savefig(graph_path, dpi=150)
        plt.close(fig)
        print("Graph saved:", graph_path)


def main():
    if not os.path.isfile(MODEL_PATH):
        print("ERROR: DQN model not found:", MODEL_PATH)
        return
    agent = DQNAgent()
    agent.load(MODEL_PATH)
    env = TrafficEnv(SCENARIOS[0], gui=GUI, max_steps=WARMUP_STEPS + MAX_MEASURE_STEPS)
    rows = []
    try:
        for scenario in SCENARIOS:
            print(f"\n=== {scenario}: fixed-time ===")
            result = run_fixed_time(scenario)
            rows.append(result)
            print(result)
            print(f"=== {scenario}: DQN V2 ===")
            result = run_dqn(env, agent, scenario)
            rows.append(result)
            print(result)
    finally:
        env.close()
        close_traci()

    write_csv(rows)
    print("\nCSV saved:", OUTPUT_CSV)
    print("\n=== Summary ===")
    print("Scenario               Policy       Avg queue   Avg stopped   Throughput   Avg delay")
    for scenario in SCENARIOS:
        for policy in ("fixed_time", "dqn_greedy"):
            row = next(r for r in rows if r["scenario"] == scenario and r["policy"] == policy)
            print(f"{scenario:22s} {policy:11s} {row['average_queue_length']:10.2f} "
                  f"{row['average_stopped_vehicles_per_step']:13.2f} {row['throughput']:12d} "
                  f"{row['average_delay']:10.2f}")
    make_graphs(rows)
    print("\nFinished. Existing evaluation_results.csv was not changed.")


if __name__ == "__main__":
    main()
