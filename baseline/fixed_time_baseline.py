import os
import sys
import traci

# --------------------------------------------------
# SUMO SETTINGS
# --------------------------------------------------

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SUMO_DIR = os.path.join(
    PROJECT_ROOT,
    "simulation",
    "sumo"
)

SCENARIOS_DIR = os.path.join(
    SUMO_DIR,
    "scenarios"
)

# --------------------------------------------------
# START SUMO
# --------------------------------------------------

def start_sumo(scenario_name):

    scenario_dir = os.path.join(
        SCENARIOS_DIR,
        scenario_name
    )

    # Find the SUMO configuration file automatically
    config_files = [
        file
        for file in os.listdir(scenario_dir)
        if file.endswith(".sumocfg")
    ]

    if not config_files:
        raise FileNotFoundError(
            f"SUMO configuration not found:\n{scenario_dir}"
        )

    config_file = os.path.join(
        scenario_dir,
        config_files[0]
    )

    command = [
        SUMO_BINARY,
        "-c",
        config_file
    ]

    print("\nStarting SUMO...")
    print("Scenario:", scenario_name)
    print("Config:", config_file)

    traci.start(command)


# --------------------------------------------------
# TRAFFIC LANES
# --------------------------------------------------

LANES = {
    "north": ["n_t_0", "n_t_1"],
    "south": ["s_t_0", "s_t_1"],
    "east": ["e_t_0", "e_t_1"],
    "west": ["w_t_0", "w_t_1"]
}


# --------------------------------------------------
# COLLECT METRICS
# --------------------------------------------------

def collect_metrics():

    total_waiting_time = 0.0
    total_queue_length = 0.0
    total_delay = 0.0

    total_steps = 0
    total_arrived = 0

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()

        total_steps += 1

        # ------------------------------------------
        # THROUGHPUT
        # ------------------------------------------

        arrived = traci.simulation.getArrivedNumber()
        total_arrived += arrived

        # ------------------------------------------
        # QUEUE + WAITING TIME + DELAY
        # ------------------------------------------

        current_queue = 0
        current_waiting = 0.0
        current_delay = 0.0

        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            speed = traci.vehicle.getSpeed(
                vehicle_id
            )

            # Waiting vehicle
            if speed < 0.1:
                current_waiting += 1.0

            # Delay approximation
            allowed_speed = traci.vehicle.getAllowedSpeed(
                vehicle_id
            )

            if allowed_speed > 0:

                delay = max(
                    0.0,
                    1.0 - (speed / allowed_speed)
                )

                current_delay += delay

        # ------------------------------------------
        # QUEUE LENGTH
        # ------------------------------------------

        for direction in LANES:

            for lane in LANES[direction]:

                try:

                    current_queue += (
                        traci.lane.getLastStepHaltingNumber(
                            lane
                        )
                    )

                except Exception:

                    pass

        total_waiting_time += current_waiting
        total_queue_length += current_queue
        total_delay += current_delay

        # ------------------------------------------
        # PRINT PROGRESS
        # ------------------------------------------

        if total_steps % 100 == 0:

            print(
                f"Step {total_steps} | "
                f"Queue: {current_queue} | "
                f"Waiting: {current_waiting:.0f} | "
                f"Arrived: {total_arrived}"
            )

    # --------------------------------------------------
    # AVERAGES
    # --------------------------------------------------

    if total_steps > 0:

        average_queue = (
            total_queue_length / total_steps
        )

        average_waiting = (
            total_waiting_time / total_steps
        )

        average_delay = (
            total_delay / total_steps
        )

    else:

        average_queue = 0
        average_waiting = 0
        average_delay = 0

    return {
        "total_waiting_time": total_waiting_time,
        "average_queue_length": average_queue,
        "throughput": total_arrived,
        "average_delay": average_delay,
        "simulation_steps": total_steps
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    # Keep S1 for the first test
    scenario = "S6_sudden_surge"

    try:

        start_sumo(scenario)

        metrics = collect_metrics()

        print("\n")
        print("=" * 50)
        print("FIXED-TIME BASELINE RESULTS")
        print("=" * 50)

        print(
            f"Total Waiting Time : "
            f"{metrics['total_waiting_time']:.2f}"
        )

        print(
            f"Average Queue Length : "
            f"{metrics['average_queue_length']:.2f}"
        )

        print(
            f"Throughput : "
            f"{metrics['throughput']}"
        )

        print(
            f"Average Delay : "
            f"{metrics['average_delay']:.2f}"
        )

        print(
            f"Simulation Steps : "
            f"{metrics['simulation_steps']}"
        )

        print("=" * 50)

    except Exception as e:

        print("\nERROR:")
        print(e)

    finally:

        try:

            traci.close()
            print("\nSUMO simulation closed.")

        except Exception:

            pass