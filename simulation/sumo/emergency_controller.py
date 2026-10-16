
"""
emergency_controller.py

Standalone emergency-priority controller for the SUMO intersection.

Initial test:
    S7_single_ambulance

This module is intentionally separate from traffic_env.py so we can
validate emergency handling before integrating it with the DQN.
"""

import os
import sys
import time

import traci


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

SUMO_INTERFACE_DIR = os.path.join(
    CURRENT_DIR
)

if SUMO_INTERFACE_DIR not in sys.path:
    sys.path.insert(0, SUMO_INTERFACE_DIR)

import traffic_interface as ti


# Green phases and their corresponding yellow transitions.
GREEN_PHASES = {0, 2, 4, 6}
YELLOW_FOR_GREEN = {
    0: 1,
    2: 3,
    4: 5,
    6: 7,
}

YELLOW_DURATION = 2

# Only routes used in the current single-intersection network.
# Values are target green phases for the ambulance's movement.
ROUTE_TO_GREEN_PHASE = {
    ("n_t", "t_s"): 0,  # North -> South
    ("s_t", "t_n"): 0,  # South -> North

    ("e_t", "t_w"): 4,  # East -> West
    ("w_t", "t_e"): 4,  # West -> East
    ("e_t", "t_n"): 4,  # East -> North
    ("s_t", "t_w"): 4,  # South -> West
    ("w_t", "t_s"): 4,  # West -> South

    ("n_t", "t_w"): 0,  # North -> West

    ("n_t", "t_e"): 2,  # North -> East

    ("e_t", "t_s"): 6,  # East -> South
    ("s_t", "t_e"): 6,  # South -> East
    ("w_t", "t_n"): 6,  # West -> North
}



def get_ambulances():
    """Return IDs of simulated vehicles whose type is ambulance."""
    ambulances = []

    for vehicle_id in traci.vehicle.getIDList():
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)

            if (
                vehicle_type.lower() == "ambulance"
                or vehicle_class == "emergency"
            ):
                ambulances.append(vehicle_id)

        except traci.TraCIException:
            continue

    return ambulances


def get_required_green_phase(vehicle_id):
    """
    Determine the required phase from the ambulance's route.

    The first two route edges identify its movement through the
    intersection. Return None for unsupported routes.
    """
    route = tuple(traci.vehicle.getRoute(vehicle_id))

    if len(route) < 2:
        return None

    return ROUTE_TO_GREEN_PHASE.get((route[0], route[1]))


def set_green_safely(target_green):
    """
    Change to a target green phase.

    If the current phase is a different green, pass through its
    configured yellow phase before setting the target green.
    """
    traffic_light_id = ti.get_traffic_light_id()
    current_phase = traci.trafficlight.getPhase(traffic_light_id)

    if current_phase == target_green:
        return

    if current_phase in GREEN_PHASES:
        yellow_phase = YELLOW_FOR_GREEN[current_phase]
        traci.trafficlight.setPhase(traffic_light_id, yellow_phase)

        for _ in range(YELLOW_DURATION):
            traci.simulationStep()

    traci.trafficlight.setPhase(traffic_light_id, target_green)

    print(
        f"[EMERGENCY] Signal set to green phase {target_green}"
    )


def run(scenario_name="S7_single_ambulance", max_steps=1000, gui=True):
    """
    Run a standalone SUMO simulation with emergency-priority handling.

    This is a first integration test, not yet the final DQN controller.
    """
    ti.start_simulation(scenario_name, gui=gui)

    traffic_light_id = ti.get_traffic_light_id()
    active_ambulance = None
    active_green = None
    cleared_steps = 0

    try:
        for step in range(max_steps):
            traci.simulationStep()

            ambulances = get_ambulances()

            if active_ambulance is None and ambulances:
                # Prioritize the first ambulance detected.
                candidate = ambulances[0]
                target_phase = get_required_green_phase(candidate)

                if target_phase is None:
                    print(
                        f"[WARNING] Unsupported ambulance route: "
                        f"{candidate}, route="
                        f"{traci.vehicle.getRoute(candidate)}"
                    )
                    continue

                active_ambulance = candidate
                active_green = target_phase
                cleared_steps = 0

                print(
                    f"[EMERGENCY] Ambulance detected: "
                    f"{active_ambulance}; "
                    f"target green phase={active_green}"
                )

                set_green_safely(active_green)

        
            elif active_ambulance is not None:
                current_ids = traci.vehicle.getIDList()
                arrived_ids = traci.simulation.getArrivedIDList()

                if active_ambulance in arrived_ids:
                    print(
                        f"[EMERGENCY] {active_ambulance} completed "
                        "its route; releasing emergency priority."
                    )
                    active_ambulance = None
                    active_green = None
                    cleared_steps = 0

                elif active_ambulance not in current_ids:
                    # The ambulance disappeared, but SUMO has not
                    # confirmed route completion. Keep priority active.
                    cleared_steps += 1

                    if cleared_steps == 1:
                        print(
                            f"[WARNING] {active_ambulance} is no longer "
                            "active, but route completion is unconfirmed. "
                            "Keeping emergency priority."
                        )

                else:
                    cleared_steps = 0

                    # Keep the required green active while the
                    # ambulance remains in the simulation.
                    current_phase = traci.trafficlight.getPhase(
                        traffic_light_id
                    )

                    if current_phase != active_green:
                        set_green_safely(active_green)


            if step % 50 == 0:
                print(
                    f"Step={step}; "
                    f"vehicles={traci.vehicle.getIDCount()}; "
                    f"phase={traci.trafficlight.getPhase(traffic_light_id)}; "
                    f"ambulance={active_ambulance}"
                )

            if traci.simulation.getMinExpectedNumber() <= 0:
                print("[INFO] No vehicles remain in the simulation.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")

    finally:
        try:
            traci.close()
        except traci.TraCIException:
            pass

        print("[INFO] SUMO connection closed.")


if __name__ == "__main__":
    run()

