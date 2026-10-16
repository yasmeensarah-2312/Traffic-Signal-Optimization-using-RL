import os
import csv
import sys

import traci


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMO_BINARY = r"C:\Program Files (x86)\SUMO\bin\sumo.exe"

SCENARIO_DIR = os.path.join(
    PROJECT_ROOT, "simulation", "sumo", "scenarios"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT, "results", "fixed_time_evaluation_v2.csv"
)

WARMUP_STEPS = 20
MAX_STEPS = 10000

LANES = {
    "north": ["n_t_0", "n_t_1"],
    "south": ["s_t_0", "s_t_1"],
    "east": ["e_t_0", "e_t_1"],
    "west": ["w_t_0", "w_t_1"],
}

SCENARIOS = {
    "S1_normal": (
        "S1_normal",
        "normal.sumocfg",
    ),
    "S2_equal_opposite": (
        "S2_equal_opposite",
        "equal_opposite.sumocfg",
    ),
    "S3_one_side_heavy": (
        "S3_one_side_heavy",
        "one-side-heavy.sumocfg",
    ),
    "S4_all_sides_heavy": (
        "S4_all_sides_heavy",
        "all_sides_heavy.sumocfg",
    ),
    "S5_changing_traffic": (
        "S5_changing_traffic",
        "changing_traffic.sumocfg",
    ),
    "S6_sudden_surge": (
        "S6_sudden_surge",
        "sudden_surge.sumocfg",
    ),
}


# ============================================================
# SCENARIO CONFIGURATION
# ============================================================

def get_config_path(scenario_name):
    folder_name, config_name = SCENARIOS[scenario_name]

    config_path = os.path.join(
        SCENARIO_DIR,
        folder_name,
        config_name,
    )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Configuration not found: {config_path}"
        )

    return config_path


# ============================================================
# METRIC COLLECTION
# ============================================================

def collect_step_metrics():
    """
    Collect the same core per-step metrics used by the DQN
    evaluator: stopped vehicles, lane queue, and speed-based delay.
    """

    current_waiting = 0.0
    current_queue = 0
    current_delay = 0.0

    vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in vehicle_ids:
        speed = traci.vehicle.getSpeed(vehicle_id)

        if speed < 0.1:
            current_waiting += 1.0

        allowed_speed = traci.vehicle.getAllowedSpeed(vehicle_id)

        if allowed_speed > 0:
            current_delay += max(
                0.0,
                1.0 - (speed / allowed_speed),
            )

    for direction_lanes in LANES.values():
        for lane_id in direction_lanes:
            try:
                current_queue += (
                    traci.lane.getLastStepHaltingNumber(lane_id)
                )
            except traci.TraCIException:
                pass

    return current_waiting, current_queue, current_delay


# ============================================================
# SINGLE SCENARIO EVALUATION
# ============================================================

def evaluate_scenario(scenario_name):
    config_path = get_config_path(scenario_name)

    print("\n" + "=" * 65)
    print(f"Evaluating fixed-time baseline: {scenario_name}")
    print(f"Configuration: {config_path}")
    print("=" * 65)

    command = [
        SUMO_BINARY,
        "-c",
        config_path,
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    total_waiting = 0.0
    total_queue = 0.0
    total_delay = 0.0
    total_arrived = 0

    metric_steps = 0
    simulation_steps = 0
    finished = False

    try:
        traci.start(command)

        while (
            traci.simulation.getMinExpectedNumber() > 0
            and simulation_steps < MAX_STEPS
        ):
            traci.simulationStep()
            simulation_steps += 1

            # Count arrivals across the complete simulation,
            # including the warm-up, as in the DQN evaluator.
            total_arrived += traci.simulation.getArrivedNumber()

            # Exclude warm-up steps from metric accumulation.
            if simulation_steps <= WARMUP_STEPS:
                continue

            waiting, queue, delay = collect_step_metrics()

            total_waiting += waiting
            total_queue += queue
            total_delay += delay
            metric_steps += 1

        finished = (
            traci.simulation.getMinExpectedNumber() <= 0
        )

    finally:
        try:
            traci.close()
        except Exception:
            pass

    if metric_steps == 0:
        raise RuntimeError(
            f"{scenario_name}: no metric steps were collected."
        )

    result = {
        "scenario": scenario_name,
        "total_waiting_time": round(total_waiting, 2),
        "average_waiting": round(
            total_waiting / metric_steps, 2
        ),
        "average_queue_length": round(
            total_queue / metric_steps, 2
        ),
        "throughput": total_arrived,
        "average_delay": round(
            total_delay / metric_steps, 2
        ),
        "simulation_steps": simulation_steps,
        "metric_steps": metric_steps,
        "finished": finished,
    }

    print("\nResults:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if not finished:
        print(
            "WARNING: Simulation reached the step limit "
            "before all vehicles completed."
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.isfile(SUMO_BINARY):
        print(f"ERROR: SUMO executable not found: {SUMO_BINARY}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    results = []

    for scenario_name in SCENARIOS:
        try:
            result = evaluate_scenario(scenario_name)
            results.append(result)
        except Exception as exc:
            print(
                f"\nERROR evaluating {scenario_name}: {exc}"
            )
            raise

    fieldnames = [
        "scenario",
        "total_waiting_time",
        "average_waiting",
        "average_queue_length",
        "throughput",
        "average_delay",
        "simulation_steps",
        "metric_steps",
        "finished",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("FIXED-TIME EVALUATION COMPLETE")
    print(f"Results saved to:\n{OUTPUT_CSV}")
    print("=" * 65)

    if len(results) != len(SCENARIOS):
        print("WARNING: Not all scenarios produced results.")
    elif not all(row["finished"] for row in results):
        print(
            "WARNING: One or more scenarios did not finish "
            "within the step limit."
        )
    else:
        print("All six scenarios finished successfully.")


if __name__ == "__main__":
    main()