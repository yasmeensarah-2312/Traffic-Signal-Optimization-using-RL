import csv
import os
import sys

import traci

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.dqn_agent import DQNAgent
from simulation.sumo import traffic_interface as ti


SUMO_BINARY = r"C:\Program Files (x86)\SUMO\bin\sumo.exe"

SCENARIO_DIR = os.path.join(
    PROJECT_ROOT,
    "simulation",
    "sumo",
    "scenarios",
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "dqn_traffic_model_v2.pt",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

OUTPUT_CSV = os.path.join(
    RESULTS_DIR,
    "dqn_evaluation.csv",
)

SCENARIOS = [
    "S1_normal",
    "S2_equal_opposite",
    "S3_one_side_heavy",
    "S4_all_sides_heavy",
    "S5_changing_traffic",
    "S6_sudden_surge",
]

MAX_SIMULATION_STEPS = 10000
WARMUP_STEPS = 20
GREEN_DURATION = 10
YELLOW_DURATION = 4

DIRECTIONS = ["north", "south", "east", "west"]

GREEN_PHASES = [0, 2, 4, 6]

YELLOW_PHASES = {
    0: 1,
    2: 3,
    4: 5,
    6: 7,
}


# ---------------------------------------------------------
# FIND SCENARIO CONFIGURATION
# ---------------------------------------------------------

def find_scenario_config(scenario_name):
    """
    Locate a scenario's SUMO configuration file.
    """

    # First, use the project's existing scenario resolver.
    try:
        config_path = ti.get_scenario_config(scenario_name)

        if os.path.isfile(config_path):
            return config_path

    except (FileNotFoundError, IndexError, OSError):
        pass

    # Fallback: search the scenario directory.
    scenario_root = os.path.join(
        SCENARIO_DIR,
        scenario_name,
    )

    if os.path.isdir(scenario_root):
        for root, _, files in os.walk(scenario_root):
            for filename in files:
                if filename.endswith(".sumocfg"):
                    return os.path.join(root, filename)

    direct_path = os.path.join(
        SCENARIO_DIR,
        f"{scenario_name}.sumocfg",
    )

    if os.path.isfile(direct_path):
        return direct_path

    raise FileNotFoundError(
        f"SUMO configuration not found for {scenario_name}"
    )


# ---------------------------------------------------------
# COLLECT ONE STEP OF METRICS
# ---------------------------------------------------------

def collect_step_metrics():
    """
    Use the same metric definitions as the fixed-time
    baseline evaluator.
    """

    current_waiting = 0.0
    current_delay = 0.0
    current_queue = 0

    for vehicle_id in traci.vehicle.getIDList():

        speed = traci.vehicle.getSpeed(vehicle_id)

        if speed < 0.1:
            current_waiting += 1.0

        allowed_speed = traci.vehicle.getAllowedSpeed(
            vehicle_id
        )

        if allowed_speed > 0:
            current_delay += max(
                0.0,
                1.0 - speed / allowed_speed,
            )

    for lane_list in ti.LANES.values():

        for lane in lane_list:

            try:
                current_queue += (
                    traci.lane.getLastStepHaltingNumber(lane)
                )

            except traci.TraCIException:
                continue

    return current_waiting, current_queue, current_delay


# ---------------------------------------------------------
# BUILD DQN STATE
# ---------------------------------------------------------

def build_state():

    state = ti.get_traffic_state()

    vector = []

    # 1. Vehicle counts: four values.
    for direction in DIRECTIONS:
        vector.append(
            state["vehicle_counts"][direction] / 20.0
        )

    # 2. Queue lengths: four values.
    for direction in DIRECTIONS:
        vector.append(
            state["queue_lengths"][direction] / 20.0
        )

    # 3. Waiting times: four values.
    for direction in DIRECTIONS:
        vector.append(
            state["waiting_times"][direction] / 300.0
        )

    # 4. Traffic-light phase: four one-hot values.
    phase_onehot = [0, 0, 0, 0]

    current_phase = state["traffic_light_phase"]

    if current_phase in GREEN_PHASES:
        phase_onehot[
            GREEN_PHASES.index(current_phase)
        ] = 1

    vector.extend(phase_onehot)

    if len(vector) != 16:
        raise ValueError(
            f"Expected 16 state values, received {len(vector)}"
        )

    return vector


# ---------------------------------------------------------
# EVALUATE ONE SCENARIO
# ---------------------------------------------------------

def evaluate_scenario(scenario_name, agent):

    print("\n" + "=" * 60)
    print(f"Evaluating: {scenario_name}")
    print("=" * 60)

    config_path = find_scenario_config(scenario_name)

    command = [
        SUMO_BINARY,
        "-c",
        config_path,
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]

    traci.start(command)

    try:

        total_waiting_time = 0.0
        total_queue_length = 0.0
        total_delay = 0.0

        total_arrived = 0
        total_steps = 0
        metric_steps = 0
        total_decisions = 0

        # ---------------------------------------------
        # WARM-UP
        # ---------------------------------------------
        # Warm-up contributes to simulation duration and
        # throughput, but not to measured averages.
        for _ in range(WARMUP_STEPS):

            if (
                total_steps >= MAX_SIMULATION_STEPS
                or traci.simulation.getMinExpectedNumber() <= 0
            ):
                break

            traci.simulationStep()

            total_steps += 1
            total_arrived += traci.simulation.getArrivedNumber()

        # ---------------------------------------------
        # DQN CONTROL LOOP
        # ---------------------------------------------
        while (
            total_steps < MAX_SIMULATION_STEPS
            and traci.simulation.getMinExpectedNumber() > 0
        ):

            state_vector = build_state()

            action = agent.select_action(
                state_vector,
                greedy=True,
            )

            target_phase = GREEN_PHASES[action]
            current_phase = ti.get_current_phase()

            # -----------------------------------------
            # YELLOW TRANSITION
            # -----------------------------------------
            if (
                current_phase in GREEN_PHASES
                and current_phase != target_phase
            ):

                yellow_phase = YELLOW_PHASES[current_phase]

                ti.change_traffic_light_phase(
                    yellow_phase
                )

                for _ in range(YELLOW_DURATION):

                    if (
                        total_steps >= MAX_SIMULATION_STEPS
                        or traci.simulation.getMinExpectedNumber() <= 0
                    ):
                        break

                    traci.simulationStep()

                    total_steps += 1
                    total_arrived += (
                        traci.simulation.getArrivedNumber()
                    )

                    waiting, queue, delay = (
                        collect_step_metrics()
                    )

                    total_waiting_time += waiting
                    total_queue_length += queue
                    total_delay += delay

                    metric_steps += 1

            if (
                total_steps >= MAX_SIMULATION_STEPS
                or traci.simulation.getMinExpectedNumber() <= 0
            ):
                break

            # -----------------------------------------
            # GREEN PHASE
            # -----------------------------------------
            ti.change_traffic_light_phase(
                target_phase
            )

            total_decisions += 1

            for _ in range(GREEN_DURATION):

                if (
                    total_steps >= MAX_SIMULATION_STEPS
                    or traci.simulation.getMinExpectedNumber() <= 0
                ):
                    break

                traci.simulationStep()

                total_steps += 1
                total_arrived += (
                    traci.simulation.getArrivedNumber()
                )

                waiting, queue, delay = (
                    collect_step_metrics()
                )

                total_waiting_time += waiting
                total_queue_length += queue
                total_delay += delay

                metric_steps += 1

        # ---------------------------------------------
        # CALCULATE AVERAGES
        # ---------------------------------------------
        # Important: divide by metric_steps, not total_steps,
        # because warm-up steps are not included in metrics.
        if metric_steps > 0:

            average_waiting = (
                total_waiting_time / metric_steps
            )

            average_queue = (
                total_queue_length / metric_steps
            )

            average_delay = (
                total_delay / metric_steps
            )

        else:
            average_waiting = 0.0
            average_queue = 0.0
            average_delay = 0.0

        finished = (
            traci.simulation.getMinExpectedNumber() <= 0
        )

        result = {
            "scenario": scenario_name,
            "total_waiting_time": round(
                total_waiting_time, 2
            ),
            "average_waiting": round(
                average_waiting, 2
            ),
            "average_queue_length": round(
                average_queue, 2
            ),
            "throughput": total_arrived,
            "average_delay": round(
                average_delay, 2
            ),
            "simulation_steps": total_steps,
            "metric_steps": metric_steps,
            "dqn_decisions": total_decisions,
            "finished": finished,
        }

        print("\nScenario results:")

        for key, value in result.items():
            print(f"{key}: {value}")

        return result

    finally:
        traci.close()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not os.path.isfile(SUMO_BINARY):
        raise FileNotFoundError(
            f"SUMO executable not found: {SUMO_BINARY}"
        )

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"DQN model not found: {MODEL_PATH}"
        )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print("Loading existing DQN model:")
    print(MODEL_PATH)

    # Load the existing model. No training occurs here.
    agent = DQNAgent(
        state_size=16,
        action_size=4,
    )

    agent.load(MODEL_PATH)

    results = []

    for scenario_name in SCENARIOS:

        result = evaluate_scenario(
            scenario_name,
            agent,
        )

        results.append(result)

    # ---------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------
    fieldnames = [
        "scenario",
        "total_waiting_time",
        "average_waiting",
        "average_queue_length",
        "throughput",
        "average_delay",
        "simulation_steps",
        "metric_steps",
        "dqn_decisions",
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

    print("\n" + "=" * 60)
    print("DQN EVALUATION COMPLETE")
    print("=" * 60)

    for result in results:
        print(
            f"{result['scenario']}: "
            f"throughput={result['throughput']}, "
            f"waiting={result['average_waiting']}, "
            f"queue={result['average_queue_length']}, "
            f"delay={result['average_delay']}, "
            f"finished={result['finished']}"
        )

    print(f"\nResults saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()