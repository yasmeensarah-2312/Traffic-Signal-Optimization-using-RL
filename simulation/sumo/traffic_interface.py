import os
import sys
import glob
import traci


# ============================================================
# PATH SETUP
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIOS_DIR = os.path.join(
    CURRENT_DIR,
    "scenarios"
)


# ============================================================
# FIND SCENARIO CONFIGURATION FILE AUTOMATICALLY
# ============================================================

def get_scenario_config(scenario_name):

    scenario_path = os.path.join(
        SCENARIOS_DIR,
        scenario_name
    )

    if not os.path.exists(scenario_path):
        raise FileNotFoundError(
            f"Scenario folder not found: {scenario_path}"
        )

    config_files = glob.glob(
        os.path.join(
            scenario_path,
            "*.sumocfg"
        )
    )

    if len(config_files) == 0:
        raise FileNotFoundError(
            f"No .sumocfg file found inside {scenario_path}"
        )

    return config_files[0]


# ============================================================
# START SUMO
# ============================================================

def start_simulation(scenario_name, gui=True):

    config_file = get_scenario_config(
        scenario_name
    )

    if gui:
        sumo_binary = "sumo-gui"
    else:
        sumo_binary = "sumo"

    command = [
        sumo_binary,
        "-c",
        config_file
    ]

    print("\nStarting SUMO...")
    print("Scenario:", scenario_name)
    print("Config:", config_file)

    traci.start(command)


# ============================================================
# GET TRAFFIC LIGHT ID
# ============================================================

def get_traffic_light_id():

    traffic_lights = traci.trafficlight.getIDList()

    if len(traffic_lights) == 0:
        raise Exception(
            "No traffic light found in the SUMO network."
        )

    return traffic_lights[0]


# ============================================================
# LANE DEFINITIONS
#
# These are based on your current network output:
#
# n_t = North
# s_t = South
# e_t = East
# w_t = West
# ============================================================

LANES = {

    "north": [
        "n_t_0",
        "n_t_1"
    ],

    "south": [
        "s_t_0",
        "s_t_1"
    ],

    "east": [
        "e_t_0",
        "e_t_1"
    ],

    "west": [
        "w_t_0",
        "w_t_1"
    ]

}


# ============================================================
# VEHICLE COUNT
# ============================================================

def get_vehicle_counts():

    counts = {}

    for direction, lane_list in LANES.items():

        total = 0

        for lane in lane_list:

            total += traci.lane.getLastStepVehicleNumber(
                lane
            )

        counts[direction] = total

    return counts


# ============================================================
# QUEUE LENGTH
#
# Number of vehicles moving slower than 0.1 m/s
# ============================================================

def get_queue_lengths():

    queues = {}

    for direction, lane_list in LANES.items():

        total_queue = 0

        for lane in lane_list:

            total_queue += traci.lane.getLastStepHaltingNumber(
                lane
            )

        queues[direction] = total_queue

    return queues


# ============================================================
# WAITING TIME
#
# Total waiting time of vehicles currently in each direction
# ============================================================

def get_waiting_times():

    waiting_times = {}

    for direction, lane_list in LANES.items():

        total_waiting = 0.0

        for lane in lane_list:

            vehicles = traci.lane.getLastStepVehicleIDs(
                lane
            )

            for vehicle_id in vehicles:

                total_waiting += traci.vehicle.getWaitingTime(
                    vehicle_id
                )

        waiting_times[direction] = total_waiting

    return waiting_times


# ============================================================
# CURRENT TRAFFIC LIGHT PHASE
# ============================================================

def get_current_phase():

    traffic_light_id = get_traffic_light_id()

    phase = traci.trafficlight.getPhase(
        traffic_light_id
    )

    return phase


# ============================================================
# CHANGE TRAFFIC LIGHT PHASE
#
# Person 2 can use this function later.
# ============================================================

def change_traffic_light_phase(phase_number):

    traffic_light_id = get_traffic_light_id()

    traci.trafficlight.setPhase(
        traffic_light_id,
        phase_number
    )

    print(
        f"Traffic light changed to phase {phase_number}"
    )


# ============================================================
# GET COMPLETE TRAFFIC STATE
#
# This is the MAIN function Person 2 will need.
# ============================================================

def get_traffic_state():

    state = {

        "vehicle_counts":
            get_vehicle_counts(),

        "queue_lengths":
            get_queue_lengths(),

        "waiting_times":
            get_waiting_times(),

        "traffic_light_phase":
            get_current_phase()

    }

    return state


# ============================================================
# PRINT TRAFFIC STATE
# ============================================================

def print_traffic_state():

    state = get_traffic_state()

    print("\nTRAFFIC STATE")

    print(
        "Vehicle Counts:",
        state["vehicle_counts"]
    )

    print(
        "Queue Lengths:",
        state["queue_lengths"]
    )

    print(
        "Waiting Times:",
        state["waiting_times"]
    )

    print(
        "Traffic Light Phase:",
        state["traffic_light_phase"]
    )


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(scenario_name):

    start_simulation(
        scenario_name,
        gui=True
    )

    step = 0

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            # Print information every 10 steps
            if step % 10 == 0:

                print(
                    f"\n========== STEP {step} =========="
                )

                print_traffic_state()

            step += 1

    except KeyboardInterrupt:

        print("\nSimulation stopped manually.")

    finally:

        traci.close()

        print("\nSimulation closed.")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            "python traffic_interface.py "
            "S1_normal"
        )

        print("\nAvailable scenarios:")

        if os.path.exists(SCENARIOS_DIR):

            for folder in os.listdir(
                SCENARIOS_DIR
            ):

                folder_path = os.path.join(
                    SCENARIOS_DIR,
                    folder
                )

                if os.path.isdir(folder_path):

                    print("-", folder)

        sys.exit()

    scenario_name = sys.argv[1]

    run_simulation(
        scenario_name
    )