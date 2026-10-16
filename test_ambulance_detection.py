import os
import sys
import traci

sys.path.insert(
    0,
    os.path.abspath("simulation/sumo")
)

import traffic_interface as ti

try:
    ti.start_simulation("S7_single_ambulance", gui=False)

    print("\nMonitoring SUMO for ambulance_1...")
    print("Press Ctrl+C to stop.\n")

    for step in range(500):
        traci.simulationStep()

        vehicle_ids = traci.vehicle.getIDList()

        if "ambulance_1" in vehicle_ids:
            route = traci.vehicle.getRoute("ambulance_1")
            road = traci.vehicle.getRoadID("ambulance_1")
            lane = traci.vehicle.getLaneID("ambulance_1")

            print(
                f"Step={step} | "
                f"ID=ambulance_1 | "
                f"Road={road} | "
                f"Lane={lane} | "
                f"Route={route}"
            )

            break
    else:
        print("Ambulance was not found in the first 500 steps.")

finally:
    try:
        traci.close()
    except Exception:
        pass
