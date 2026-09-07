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
# SCENARIOS
# ============================================================

SCENARIOS = {
    "S1_normal": "S1_normal",
    "S2_equal_opposite": "S2_equal_opposite",
    "S3_one_side_heavy": "S3_one_side_heavy",
    "S4_all_sides_heavy": "S4_all_sides_heavy",
    "S5_changing_traffic": "S5_changing_traffic",
    "S6_sudden_surge": "S6_sudden_surge",
    "S7_single_ambulance": "S7_single_ambulance",
    "S8_two_ambulances": "S8_two_ambulances"
}


# ============================================================
# FIND SUMO CONFIGURATION FILE AUTOMATICALLY
# ============================================================

def get_config_file(scenario_name):

    scenario_folder = os.path.join(
        SCENARIOS_DIR,
        SCENARIOS[scenario_name]
    )

    config_files = glob.glob(
        os.path.join(scenario_folder, "*.sumocfg")
    )

    if not config_files:
        raise FileNotFoundError(
            f"No .sumocfg file found inside {scenario_folder}"
        )

    return config_files[0]


# ============================================================
# START SUMO
# ============================================================

def start_simulation(scenario_name, gui=True):

    config_file = get_config_file(scenario_name)

    if gui:
        sumo_binary = "sumo-gui"
    else:
        sumo_binary = "sumo"

    print("\n====================================")
    print("STARTING SCENARIO:", scenario_name)
    print("CONFIG FILE:", config_file)
    print("====================================\n")

    traci.start([
        sumo_binary,
        "-c",
        config_file,
        "--start"
    ])


# ============================================================
# STOP SUMO
# ============================================================

def stop_simulation():

    traci.close()

    print("\nSimulation closed.")


# ============================================================
# GET TRAFFIC LIGHT ID
# ============================================================

def get_traffic_light_id():

    traffic_lights = traci.trafficlight.getIDList()

    if len(traffic_lights) == 0:
        raise Exception("No traffic light found!")

    return traffic_lights[0]


# ============================================================
# GET INCOMING LANES
# ============================================================

def get_incoming_lanes():

    all_lanes = traci.lane.getIDList()

    incoming_lanes = []

    for lane in all_lanes:

        # Ignore internal SUMO lanes
        if lane.startswith(":"):
            continue

        # Take approach lanes
        if (
            lane.startswith("n_t")
            or lane.startswith("s_t")
            or lane.startswith("e_t")
            or lane.startswith("w_t")
        ):
            incoming_lanes.append(lane)

    return incoming_lanes


# ============================================================
# GET VEHICLE COUNT FOR EACH DIRECTION
# ============================================================

def get_vehicle_counts():

    counts = {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    }

    lanes = get_incoming_lanes()

    for lane in lanes:

        vehicle_count = traci.lane.getLastStepVehicleNumber(
            lane
        )

        if lane.startswith("n_t"):
            counts["north"] += vehicle_count

        elif lane.startswith("s_t"):
            counts["south"] += vehicle_count

        elif lane.startswith("e_t"):
            counts["east"] += vehicle_count

        elif lane.startswith("w_t"):
            counts["west"] += vehicle_count

    return counts


# ============================================================
# GET QUEUE LENGTH
# ============================================================

def get_queue_lengths():

    queues = {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    }

    lanes = get_incoming_lanes()

    for lane in lanes:

        queue = traci.lane.getLastStepHaltingNumber(
            lane
        )

        if lane.startswith("n_t"):
            queues["north"] += queue

        elif lane.startswith("s_t"):
            queues["south"] += queue

        elif lane.startswith("e_t"):
            queues["east"] += queue

        elif lane.startswith("w_t"):
            queues["west"] += queue

    return queues


# ============================================================
# GET WAITING TIME
# ============================================================

def get_waiting_times():

    waiting = {
        "north": 0.0,
        "south": 0.0,
        "east": 0.0,
        "west": 0.0
    }

    lanes = get_incoming_lanes()

    for lane in lanes:

        vehicle_ids = traci.lane.getLastStepVehicleIDs(
            lane
        )

        for vehicle_id in vehicle_ids:

            waiting_time = traci.vehicle.getWaitingTime(
                vehicle_id
            )

            if lane.startswith("n_t"):
                waiting["north"] += waiting_time

            elif lane.startswith("s_t"):
                waiting["south"] += waiting_time

            elif lane.startswith("e_t"):
                waiting["east"] += waiting_time

            elif lane.startswith("w_t"):
                waiting["west"] += waiting_time

    return waiting


# ============================================================
# GET CURRENT TRAFFIC LIGHT PHASE
# ============================================================

def get_current_phase():

    traffic_light_id = get_traffic_light_id()

    phase = traci.trafficlight.getPhase(
        traffic_light_id
    )

    return phase


# ============================================================
# CHANGE TRAFFIC LIGHT PHASE
# ============================================================

def set_traffic_light_phase(phase):

    traffic_light_id = get_traffic_light_id()

    traci.trafficlight.setPhase(
        traffic_light_id,
        phase
    )


# ============================================================
# GET COMPLETE STATE
# THIS IS WHAT PERSON 2 WILL USE FOR RL
# ============================================================

def get_state():

    vehicle_counts = get_vehicle_counts()

    queue_lengths = get_queue_lengths()

    waiting_times = get_waiting_times()

    current_phase = get_current_phase()

    state = {

        "vehicle_counts": vehicle_counts,

        "queue_lengths": queue_lengths,

        "waiting_times": waiting_times,

        "current_phase": current_phase

    }

    return state


# ============================================================
# PRINT VEHICLE INFORMATION
# ============================================================

def print_vehicle_information(step):

    vehicle_ids = traci.vehicle.getIDList()

    print("\nStep:", step)

    for vehicle_id in vehicle_ids:

        vehicle_type = traci.vehicle.getTypeID(
            vehicle_id
        )

        lane = traci.vehicle.getLaneID(
            vehicle_id
        )

        speed = traci.vehicle.getSpeed(
            vehicle_id
        )

        waiting = traci.vehicle.getWaitingTime(
            vehicle_id
        )

        print(
            f"ID: {vehicle_id} | "
            f"Type: {vehicle_type} | "
            f"Lane: {lane} | "
            f"Speed: {speed} | "
            f"Waiting: {waiting}"
        )


# ============================================================
# RUN SIMULATION
# ============================================================

def run_simulation(scenario_name, max_steps=1000):

    start_simulation(
        scenario_name,
        gui=True
    )

    step = 0

    try:

        while (
            traci.simulation.getMinExpectedNumber() > 0
            and step < max_steps
        ):

            # Advance simulation
            traci.simulationStep()

            # Print individual vehicle information
            print_vehicle_information(step)

            # Get traffic state
            state = get_state()

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
                state["current_phase"]
            )

            step += 1

    finally:

        stop_simulation()


# ============================================================
# TEST ALL SCENARIOS
# ============================================================

if __name__ == "__main__":

    print("\nAVAILABLE SCENARIOS:")

    for scenario in SCENARIOS:
        print("-", scenario)

    print("\nStarting S1_normal...\n")

    run_simulation(
        "S1_normal",
        max_steps=1000
    )