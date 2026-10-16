import os
import sys
import random
import numpy as np
import torch
import traci

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_DIR = os.path.join(ROOT_DIR, "simulation", "sumo")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SUMO_DIR)

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent
import traffic_interface as ti
import emergency_controller as emergency

SCENARIO = "S7_single_ambulance"
MAX_STEPS = 1000
DECISION_INTERVAL = 10

GREEN_PHASES = [0, 2, 4, 6]
YELLOW_FOR_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}
YELLOW_PHASES = {1, 3, 5, 7}
YELLOW_DURATION = 2

MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_traffic_model.pt")


def run_experiment(priority_enabled):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = TrafficEnv(
        scenario_name=SCENARIO,
        gui=False,
        max_steps=MAX_STEPS,
    )
    agent = DQNAgent()

    if os.path.isfile(MODEL_PATH):
        agent.load(MODEL_PATH)
        model_status = "saved DQN model loaded"
    else:
        model_status = "UNTRAINED DQN"

    simulation_steps = 0
    dqn_decisions = 0
    ambulance_start_step = None
    ambulance_arrival_step = None
    active_ambulance = None
    emergency_green = None
    detections = 0
    releases = 0
    throughput = 0
    waiting_total = 0.0
    waiting_samples = 0
    next_decision = 20
    started = False

    def advance_step():
        nonlocal simulation_steps, throughput
        nonlocal waiting_total, waiting_samples
        nonlocal ambulance_start_step, ambulance_arrival_step

        traci.simulationStep()
        simulation_steps += 1

        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:
            waiting_total += traci.vehicle.getWaitingTime(vehicle_id)
            waiting_samples += 1

        if (
            ambulance_start_step is None
            and "ambulance_1" in vehicle_ids
        ):
            ambulance_start_step = simulation_steps

        arrived = traci.simulation.getArrivedIDList()
        throughput += len(arrived)

        if (
            "ambulance_1" in arrived
            and ambulance_start_step is not None
        ):
            ambulance_arrival_step = simulation_steps

    def change_green_safely(target_green):
        nonlocal simulation_steps

        tl_id = ti.get_traffic_light_id()
        current_phase = traci.trafficlight.getPhase(tl_id)

        if current_phase == target_green:
            return

        # If leaving a green, display its corresponding yellow.
        if current_phase in GREEN_PHASES:
            yellow_phase = YELLOW_FOR_GREEN[current_phase]
            traci.trafficlight.setPhase(tl_id, yellow_phase)

            for _ in range(YELLOW_DURATION):
                advance_step()

        # If already in yellow, allow the yellow interval to finish.
        elif current_phase in YELLOW_PHASES:
            for _ in range(YELLOW_DURATION):
                advance_step()

        traci.trafficlight.setPhase(tl_id, target_green)

    try:
        label = "PRIORITY ON" if priority_enabled else "PRIORITY OFF"
        print(f"\n{'=' * 18} {label} {'=' * 18}")
        print(f"Model status: {model_status}")

        ti.start_simulation(SCENARIO, gui=False)
        started = True

        # Same 20-step warm-up for both experiments.
        for _ in range(20):
            advance_step()

        while simulation_steps < MAX_STEPS:
            advance_step()

            # Release priority only after SUMO confirms arrival.
            if (
                active_ambulance is not None
                and ambulance_arrival_step is not None
            ):
                print(
                    f"[EMERGENCY] {active_ambulance} completed "
                    f"its route at step {ambulance_arrival_step}."
                )
                active_ambulance = None
                emergency_green = None
                releases += 1
                next_decision = simulation_steps

            # While the ambulance is active, it owns the signal.
            if active_ambulance is not None:
                if active_ambulance in traci.vehicle.getIDList():
                    change_green_safely(emergency_green)
                continue

            # Detect and prioritise an ambulance only in the ON experiment.
            if priority_enabled:
                ambulances = emergency.get_ambulances()

                if ambulances:
                    ambulance_id = ambulances[0]
                    target = emergency.get_required_green_phase(
                        ambulance_id
                    )

                    if target is not None:
                        active_ambulance = ambulance_id
                        emergency_green = target
                        detections += 1

                        print(
                            f"[EMERGENCY] Detected {ambulance_id} "
                            f"at step {simulation_steps}; "
                            f"requesting phase {target}."
                        )

                        change_green_safely(target)
                        continue

            # Normal DQN control.
            if simulation_steps >= next_decision:
                state = env._get_state_vector()
                action = agent.select_action(state)
                target_green = GREEN_PHASES[action]

                change_green_safely(target_green)

                dqn_decisions += 1
                next_decision = simulation_steps + DECISION_INTERVAL

            if traci.simulation.getMinExpectedNumber() <= 0:
                break

        travel_time = None
        if (
            ambulance_start_step is not None
            and ambulance_arrival_step is not None
        ):
            travel_time = ambulance_arrival_step - ambulance_start_step

        mean_wait = (
            waiting_total / waiting_samples
            if waiting_samples else 0.0
        )

        results = {
            "mode": label,
            "steps": simulation_steps,
            "ambulance_travel_time": travel_time,
            "mean_vehicle_waiting": mean_wait,
            "throughput": throughput,
            "dqn_decisions": dqn_decisions,
            "detections": detections,
            "releases": releases,
        }

        print(f"Steps: {simulation_steps}")
        print(f"Ambulance travel time: {travel_time}")
        print(f"Mean instantaneous vehicle waiting: {mean_wait:.3f}")
        print(f"Vehicles arrived (throughput): {throughput}")
        print(f"DQN decisions: {dqn_decisions}")
        print(f"Emergency detections/releases: {detections}/{releases}")

        return results

    finally:
        if started:
            try:
                traci.close()
            except Exception:
                pass


def main():
    baseline = run_experiment(priority_enabled=False)
    priority = run_experiment(priority_enabled=True)

    print("\n\n========== COMPARISON ==========")
    print(f"{'Metric':<34}{'Priority OFF':>16}{'Priority ON':>16}")
    print("-" * 66)

    metrics = [
        ("Ambulance travel time (steps)", "ambulance_travel_time"),
        ("Mean vehicle waiting", "mean_vehicle_waiting"),
        ("Vehicles arrived", "throughput"),
        ("DQN decisions", "dqn_decisions"),
    ]

    for label, key in metrics:
        off = baseline[key]
        on = priority[key]

        off_text = "NOT COMPLETED" if off is None else f"{off:.3f}" if isinstance(off, float) else str(off)
        on_text = "NOT COMPLETED" if on is None else f"{on:.3f}" if isinstance(on, float) else str(on)

        print(f"{label:<34}{off_text:>16}{on_text:>16}")

    if (
        baseline["ambulance_travel_time"] is not None
        and priority["ambulance_travel_time"] is not None
    ):
        improvement = (
            baseline["ambulance_travel_time"]
            - priority["ambulance_travel_time"]
        )
        print(f"\nAmbulance travel-time improvement: {improvement} steps")
        if improvement > 0:
            print("Emergency priority reduced ambulance travel time in this run.")
        elif improvement < 0:
            print("Emergency priority did not improve ambulance travel time in this run.")
        else:
            print("Ambulance travel time was unchanged in this run.")

    print("\nNote: waiting is an instantaneous per-vehicle measure averaged")
    print("over observed vehicle-steps, not average trip delay.")
    print("One run is preliminary evidence, not statistical validation.")


if __name__ == "__main__":
    main()
