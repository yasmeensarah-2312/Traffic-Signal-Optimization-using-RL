"""
Adaptive traffic control with ambulance priority.
This uses a queue-based baseline, NOT a trained DQN.
"""

import os
import sys
import traci

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMO_DIR = os.path.join(PROJECT_ROOT, "simulation", "sumo")

for path in (PROJECT_ROOT, SUMO_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import traffic_interface as ti
import emergency_controller as ec
from rl.adaptive_controller import AdaptiveController

GREEN_PHASES = {0, 2, 4, 6}
YELLOW_FOR_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}
YELLOW_DURATION = 2
DIRECTIONS = ("north", "south", "east", "west")


def change_green_safely(target_green, traffic_light_id):
    current = traci.trafficlight.getPhase(traffic_light_id)
    if current == target_green:
        return

    if current in GREEN_PHASES:
        traci.trafficlight.setPhase(
            traffic_light_id, YELLOW_FOR_GREEN[current]
        )
        for _ in range(YELLOW_DURATION):
            traci.simulationStep()

    traci.trafficlight.setPhase(traffic_light_id, target_green)


def run(scenario_name="S7_single_ambulance", max_steps=1000, gui=True):
    controller = AdaptiveController(min_green_steps=5)
    active_ambulance = None
    active_green = None
    last_decision_step = -999
    total_arrived = 0
    emergency_detected = 0
    emergency_completed = 0
    stop_reason = "step limit reached"

    ti.start_simulation(scenario_name, gui=gui)

    try:
        traffic_light_id = ti.get_traffic_light_id()
        print("\nAdaptive baseline + emergency-priority test")
        print("This is not a trained DQN evaluation.")

        for step in range(max_steps):
            traci.simulationStep()
            total_arrived += len(traci.simulation.getArrivedIDList())

            expected = traci.simulation.getMinExpectedNumber()
            if expected <= 0:
                stop_reason = "SUMO reports no expected vehicles"
                break

            ambulance_ids = ec.get_ambulances()

            if active_ambulance is None and ambulance_ids:
                candidate = ambulance_ids[0]
                required_green = ec.get_required_green_phase(candidate)

                if required_green is not None:
                    active_ambulance = candidate
                    active_green = required_green
                    emergency_detected += 1
                    print(
                        f"[EMERGENCY] Detected {candidate}; "
                        f"target phase={required_green}"
                    )
                    ec.set_green_safely(required_green)

            if active_ambulance is not None:
                arrived_ids = traci.simulation.getArrivedIDList()
                active_ids = traci.vehicle.getIDList()

                if active_ambulance in arrived_ids:
                    print(f"[EMERGENCY] {active_ambulance} completed.")
                    active_ambulance = None
                    active_green = None
                    emergency_completed += 1
                    last_decision_step = -999

                elif active_ambulance in active_ids:
                    if traci.trafficlight.getPhase(traffic_light_id) != active_green:
                        ec.set_green_safely(active_green)

                if active_ambulance is not None:
                    if step % 50 == 0:
                        print(
                            f"[STEP {step}] emergency={active_ambulance}; "
                            f"phase={traci.trafficlight.getPhase(traffic_light_id)}; "
                            f"active vehicles={traci.vehicle.getIDCount()}"
                        )
                    continue

            phase = traci.trafficlight.getPhase(traffic_light_id)
            if phase not in GREEN_PHASES:
                continue

            if step - last_decision_step >= 5:
                state = ti.get_traffic_state()
                queues = [state["queue_lengths"][d] for d in DIRECTIONS]
                waiting = [state["waiting_times"][d] for d in DIRECTIONS]

                phase_to_action = {0: 0, 2: 1, 4: 2, 6: 3}
                decision = controller.select_action(
                    queues,
                    waiting,
                    current_action=phase_to_action[phase],
                )

                target = decision["target_phase"]
                last_decision_step = step

                if target != phase:
                    change_green_safely(target, traffic_light_id)

            if step % 100 == 0:
                print(
                    f"[STEP {step}] active={traci.vehicle.getIDCount()}; "
                    f"arrived={total_arrived}; "
                    f"phase={traci.trafficlight.getPhase(traffic_light_id)}"
                )

        final_time = traci.simulation.getTime()
        final_active = traci.vehicle.getIDCount()

        print("\n========== FINAL SUMMARY ==========")
        print(f"Stop reason: {stop_reason}")
        print(f"Simulation time: {final_time}")
        print(f"Vehicles arrived: {total_arrived}")
        print(f"Vehicles still active: {final_active}")
        print(f"Ambulances detected: {emergency_detected}")
        print(f"Ambulances completed: {emergency_completed}")

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped manually.")

    finally:
        try:
            traci.close()
        except traci.TraCIException:
            pass
        print("[INFO] SUMO connection closed.")


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "S7_single_ambulance"
    run(scenario_name=scenario, max_steps=1000, gui=True)
