import traci
import os

sumo_config = os.path.join(
    os.path.dirname(__file__),
    "scenarios",
    "S1_normal",
    "normal.sumocfg"
)

sumo_cmd = ["sumo-gui", "-c", sumo_config]

traci.start(sumo_cmd)

print("\nVEHICLE INFORMATION:\n")

for step in range(20):

    traci.simulationStep()

    print("\nStep:", step)

    vehicle_ids = traci.vehicle.getIDList()

    for vehicle_id in vehicle_ids:

        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        speed = traci.vehicle.getSpeed(vehicle_id)
        waiting_time = traci.vehicle.getWaitingTime(vehicle_id)

        print(
            "ID:", vehicle_id,
            "| Type:", vehicle_type,
            "| Lane:", lane_id,
            "| Speed:", speed,
            "| Waiting:", waiting_time
        )

traci.close()