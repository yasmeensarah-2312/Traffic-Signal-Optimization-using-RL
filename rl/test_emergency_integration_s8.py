
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


SCENARIO = "S8_two_ambulances"
MAX_STEPS = 1000
DECISION_INTERVAL = 10

GREEN_PHASES = [0, 2, 4, 6]
YELLOW_FOR_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}
YELLOW_PHASES = {1, 3, 5, 7}
YELLOW_DURATION = 2

MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_traffic_model_v2.pt")


def run_experiment(priority_enabled):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = TrafficEnv(
        scenario_name=SCENARIO,
        gui=True,
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

    # Track each ambulance independently.
    ambulance_start_steps = {}
    ambulance_arrival_steps = {}

    # An ambulance is handled once per experiment.
    handled_ambulances = set()

    active_ambulance = None
    emergency_green = None

    detections = 0
    releases = 0
    throughput = 0

    # S8 validation metrics
    safe_transition_count = 0
    unsafe_transition_count = 0
    dqn_decisions_after_release = 0
    release_steps = []
    dqn_decision_steps = []
    transition_log = []

    waiting_total = 0.0
    waiting_samples = 0

    next_decision = 20
    started = False

    def advance_step():
        nonlocal simulation_steps
        nonlocal throughput
        nonlocal waiting_total, waiting_samples

        traci.simulationStep()
        simulation_steps += 1

        vehicle_ids = traci.vehicle.getIDList()

        # Record instantaneous waiting for active vehicles.
        for vehicle_id in vehicle_ids:
            waiting_total += traci.vehicle.getWaitingTime(vehicle_id)
            waiting_samples += 1

        # Record the first observed step for each ambulance.
        for vehicle_id in vehicle_ids:
            if (
                vehicle_id.startswith("ambulance_")
                and vehicle_id not in ambulance_start_steps
            ):
                ambulance_start_steps[vehicle_id] = simulation_steps

        # SUMO reports vehicles that completed their routes.
        arrived = traci.simulation.getArrivedIDList()
        throughput += len(arrived)

        # Record each ambulance's arrival independently.
        for vehicle_id in arrived:
            if (
                vehicle_id.startswith("ambulance_")
                and vehicle_id in ambulance_start_steps
            ):
                ambulance_arrival_steps[vehicle_id] = simulation_steps

    def change_green_safely(target_green):
        nonlocal safe_transition_count, unsafe_transition_count

        tl_id = ti.get_traffic_light_id()
        current_phase = traci.trafficlight.getPhase(tl_id)

        if current_phase == target_green:
            return

        original_phase = current_phase
        yellow_used = None

        # Transition from green through its corresponding yellow phase.
        if current_phase in GREEN_PHASES:
            yellow_phase = YELLOW_FOR_GREEN[current_phase]
            yellow_used = yellow_phase

            traci.trafficlight.setPhase(tl_id, yellow_phase)

            for _ in range(YELLOW_DURATION):
                advance_step()

        # Finish the yellow interval if already in a yellow phase.
        elif current_phase in YELLOW_PHASES:
            yellow_used = current_phase

            for _ in range(YELLOW_DURATION):
                advance_step()

        traci.trafficlight.setPhase(tl_id, target_green)

        # Record the transition for later validation.
        safe = yellow_used is not None

        if safe:
            safe_transition_count += 1
        else:
            unsafe_transition_count += 1

        transition_log.append({
            "from": original_phase,
            "yellow": yellow_used,
            "to": target_green,
            "safe": safe,
        })

    try:
        label = "PRIORITY ON" if priority_enabled else "PRIORITY OFF"

        print(f"\n{'=' * 18} {label} {'=' * 18}")
        print(f"Model status: {model_status}")

        ti.start_simulation(SCENARIO, gui=True)
        started = True

        # Identical warm-up for both experiments.
        for _ in range(20):
            advance_step()

        while simulation_steps < MAX_STEPS:
            advance_step()

            # Release priority only after SUMO confirms that the
            # currently active ambulance has arrived.
            if (
                active_ambulance is not None
                and active_ambulance in ambulance_arrival_steps
            ):
                arrival_step = ambulance_arrival_steps[active_ambulance]

                print(
                    f"[EMERGENCY] {active_ambulance} completed "
                    f"its route at step {arrival_step}."
                )

                active_ambulance = None
                emergency_green = None
                releases += 1
                release_steps.append(simulation_steps)
                next_decision = simulation_steps

            # While an ambulance is still active, maintain its priority.
            if active_ambulance is not None:
                active_ids = traci.vehicle.getIDList()

                if active_ambulance in active_ids:
                    change_green_safely(emergency_green)

                # If the vehicle disappeared without an arrival event,
                # do not silently count it as a successful completion.
                continue

            # Detect and prioritise ambulances in the priority-enabled run.
            if priority_enabled:
                ambulances = emergency.get_ambulances()

                # Choose the first ambulance that has not already been
                # handled and has not completed its route.
                ambulance_id = next(
                    (
                        vehicle_id
                        for vehicle_id in ambulances
                        if vehicle_id not in handled_ambulances
                        and vehicle_id not in ambulance_arrival_steps
                    ),
                    None,
                )

                if ambulance_id is not None:
                    target = emergency.get_required_green_phase(
                        ambulance_id
                    )

                    if target is not None:
                        active_ambulance = ambulance_id
                        emergency_green = target

                        handled_ambulances.add(ambulance_id)
                        detections += 1

                        print(
                            f"[EMERGENCY] Detected {ambulance_id} "
                            f"at step {simulation_steps}; "
                            f"requesting phase {target}."
                        )

                        change_green_safely(target)
                        continue

            # Normal DQN control resumes when no ambulance has priority.
            if simulation_steps >= next_decision:
                state = env._get_state_vector()
                action = agent.select_action(state, greedy=True)
                target_green = GREEN_PHASES[action]

                change_green_safely(target_green)

                dqn_decisions += 1
                dqn_decision_steps.append(simulation_steps)

                if release_steps and simulation_steps >= release_steps[-1]:
                    dqn_decisions_after_release += 1

                next_decision = simulation_steps + DECISION_INTERVAL

        # Calculate travel time separately for each completed ambulance.
        ambulance_travel_times = {
            vehicle_id: ambulance_arrival_steps[vehicle_id] - start_step
            for vehicle_id, start_step in ambulance_start_steps.items()
            if vehicle_id in ambulance_arrival_steps
        }

        if ambulance_travel_times:
            average_travel_time = (
                sum(ambulance_travel_times.values())
                / len(ambulance_travel_times)
            )
        else:
            average_travel_time = None

        mean_wait = (
            waiting_total / waiting_samples
            if waiting_samples
            else 0.0
        )

        results = {
            "mode": label,
            "steps": simulation_steps,
            "ambulance_travel_time": average_travel_time,
            "ambulance_travel_times": ambulance_travel_times,
            "mean_vehicle_waiting": mean_wait,
            "throughput": throughput,
            "dqn_decisions": dqn_decisions,
            "detections": detections,
            "releases": releases,
        }

        print(f"\nSteps: {simulation_steps}")
        print("Individual ambulance travel times:")

        for vehicle_id in sorted(
            ambulance_travel_times,
            key=lambda value: ambulance_start_steps[value],
        ):
            print(
                f"  {vehicle_id}: "
                f"{ambulance_travel_times[vehicle_id]} steps"
            )

        print(
            f"Ambulances completed: "
            f"{len(ambulance_travel_times)}/"
            f"{len(ambulance_start_steps)} observed"
        )
        print(f"Mean ambulance travel time: {average_travel_time}")
        print(f"Mean instantaneous vehicle waiting: {mean_wait:.3f}")
        print(f"Vehicles arrived (throughput): {throughput}")
        print(f"DQN decisions: {dqn_decisions}")
        print(f"Emergency detections/releases: {detections}/{releases}")
        print(
            f"Safe transitions: {safe_transition_count}; "
            f"unsafe transitions: {unsafe_transition_count}"
        )
        print(f"DQN decisions after priority release: {dqn_decisions_after_release}")
        print(f"Priority release steps: {release_steps}")

        all_ambulances_completed = (
            len(ambulance_travel_times) >= 2
            if priority_enabled
            else len(ambulance_travel_times) >= 2
        )

        if priority_enabled:
            print(
                "Ambulance lifecycle:",
                "PASS" if detections == 2 and releases == 2
                and all_ambulances_completed else "FAIL"
            )

        print(
            "Transition-path check:",
            "PASS" if unsafe_transition_count == 0 else "FAIL"
        )

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
        ("Mean ambulance travel time", "ambulance_travel_time"),
        ("Mean vehicle waiting", "mean_vehicle_waiting"),
        ("Vehicles arrived", "throughput"),
        ("DQN decisions", "dqn_decisions"),
        ("Emergency detections", "detections"),
        ("Emergency releases", "releases"),
    ]

    for label, key in metrics:
        off = baseline[key]
        on = priority[key]

        if off is None:
            off_text = "NOT COMPLETED"
        elif isinstance(off, float):
            off_text = f"{off:.3f}"
        else:
            off_text = str(off)

        if on is None:
            on_text = "NOT COMPLETED"
        elif isinstance(on, float):
            on_text = f"{on:.3f}"
        else:
            on_text = str(on)

        print(f"{label:<34}{off_text:>16}{on_text:>16}")

    off_times = baseline["ambulance_travel_times"]
    on_times = priority["ambulance_travel_times"]

    print("\nIndividual ambulance travel-time comparison:")

    for vehicle_id in ("ambulance_1", "ambulance_2"):
        off_time = off_times.get(vehicle_id)
        on_time = on_times.get(vehicle_id)

        off_text = "NOT COMPLETED" if off_time is None else str(off_time)
        on_text = "NOT COMPLETED" if on_time is None else str(on_time)

        print(
            f"  {vehicle_id}: "
            f"OFF = {off_text}, ON = {on_text}"
        )

    if (
        baseline["ambulance_travel_time"] is not None
        and priority["ambulance_travel_time"] is not None
    ):
        improvement = (
            baseline["ambulance_travel_time"]
            - priority["ambulance_travel_time"]
        )

        print(
            f"\nMean ambulance travel-time improvement: "
            f"{improvement:.3f} steps"
        )

        if improvement > 0:
            print(
                "Emergency priority reduced mean ambulance "
                "travel time in this run."
            )
        elif improvement < 0:
            print(
                "Emergency priority increased mean ambulance "
                "travel time in this run."
            )
        else:
            print(
                "Mean ambulance travel time was unchanged "
                "in this run."
            )

    print("\nNote:")
    print("- Only completed ambulance routes contribute to travel times.")
    print("- Waiting is an instantaneous per-vehicle measure averaged")
    print("  over observed vehicle-steps, not average trip delay.")
    print("- One run is preliminary evidence, not statistical validation.")


if __name__ == "__main__":
    main()

