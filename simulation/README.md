# Traffic Signal Optimization Using Reinforcement Learning

## Project Overview

This project implements a smart traffic signal optimization system using **SUMO (Simulation of Urban Mobility)**, **Python**, and **TraCI**.

The project simulates different traffic conditions at an intersection and provides an interface for accessing real-time traffic information such as:

- Vehicle counts
- Queue lengths
- Vehicle waiting times
- Current traffic signal phase

The simulation environment is designed to support multiple traffic scenarios and can later be integrated with:

- Reinforcement Learning
- Traffic signal optimization
- Emergency vehicle priority
- Ambulance detection
- Performance evaluation

---

# Person 1 Work

Person 1 is responsible for setting up the SUMO simulation environment and creating the Python interface between SUMO and the rest of the project.

The completed work includes:

- Creation of the SUMO road network
- Creation of multiple traffic scenarios
- SUMO configuration files for each scenario
- Traffic route files for each scenario
- Python TraCI interface
- Real-time vehicle count extraction
- Queue length extraction
- Waiting time extraction
- Traffic light phase extraction
- Traffic light phase control function

The main interface file is:

```text
traffic_interface.py

This file acts as the connection between the SUMO simulation and the other project components.

Project Structure
Traffic-Signal-Optimization-using-RL/
│
└── simulation/
    │
    └── sumo/
        │
        ├── scenarios/
        │   │
        │   ├── S1_normal/
        │   │   ├── S1_normal.rou.xml
        │   │   └── S1_normal.sumocfg
        │   │
        │   ├── S2_equal_opposite/
        │   │   ├── S2_equal_opposite.rou.xml
        │   │   └── S2_equal_opposite.sumocfg
        │   │
        │   ├── S3_one_side_heavy/
        │   │   ├── S3_one_side_heavy.rou.xml
        │   │   └── S3_one_side_heavy.sumocfg
        │   │
        │   ├── S4_all_sides_heavy/
        │   │   ├── S4_all_sides_heavy.rou.xml
        │   │   └── S4_all_sides_heavy.sumocfg
        │   │
        │   ├── S5_changing_traffic/
        │   │   ├── S5_changing_traffic.rou.xml
        │   │   └── S5_changing_traffic.sumocfg
        │   │
        │   ├── S6_sudden_surge/
        │   │   ├── S6_sudden_surge.rou.xml
        │   │   └── S6_sudden_surge.sumocfg
        │   │
        │   ├── S7_single_ambulance/
        │   │   ├── S7_single_ambulance.rou.xml
        │   │   └── S7_single_ambulance.sumocfg
        │   │
        │   └── S8_two_ambulances/
        │       ├── S8_two_ambulances.rou.xml
        │       └── S8_two_ambulances.sumocfg
        │
        ├── single-intersection.net.xml
        │
        ├── single-intersection-vhvh.rou.xml
        │
        ├── traffic_interface.py
        │
        ├── sumo_runner.py
        │
        └── test_traci.py
Common SUMO Network

All traffic scenarios use the same SUMO road network.

The common network file is:

single-intersection.net.xml

Each scenario has its own:

Route file (.rou.xml)
SUMO configuration file (.sumocfg)

The scenario configuration file should reference the common network file and the route file belonging to that particular scenario.

Example:

S1_normal.sumocfg

uses:

single-intersection.net.xml

and:

S1_normal.rou.xml
Available Traffic Scenarios

The following scenarios are currently available:

S1_normal

Normal traffic conditions.

S1_normal

Files:

S1_normal.rou.xml
S1_normal.sumocfg
S2_equal_opposite

Traffic is distributed equally in opposite directions.

S2_equal_opposite

Files:

S2_equal_opposite.rou.xml
S2_equal_opposite.sumocfg
S3_one_side_heavy

One side of the intersection has heavy traffic.

S3_one_side_heavy

Files:

S3_one_side_heavy.rou.xml
S3_one_side_heavy.sumocfg
S4_all_sides_heavy

All directions have heavy traffic.

S4_all_sides_heavy

Files:

S4_all_sides_heavy.rou.xml
S4_all_sides_heavy.sumocfg
S5_changing_traffic

Traffic density changes during the simulation.

S5_changing_traffic

Files:

S5_changing_traffic.rou.xml
S5_changing_traffic.sumocfg
S6_sudden_surge

A sudden increase in traffic occurs during the simulation.

S6_sudden_surge

Files:

S6_sudden_surge.rou.xml
S6_sudden_surge.sumocfg
S7_single_ambulance

A traffic scenario containing one ambulance.

S7_single_ambulance

Files:

S7_single_ambulance.rou.xml
S7_single_ambulance.sumocfg
S8_two_ambulances

A traffic scenario containing two ambulances.

S8_two_ambulances

Files:

S8_two_ambulances.rou.xml
S8_two_ambulances.sumocfg
Requirements

The following software and Python packages are required.

SUMO

Install SUMO and make sure the SUMO commands are available from the terminal.

The project uses:

sumo

and:

sumo-gui
Python

Python is required to run the TraCI interface.

The following package is used:

traci
Installing Required Python Package

Install TraCI using:

pip install traci

If SUMO is already installed correctly, its Python tools may also provide TraCI.

Main Interface

The main Python interface is:

traffic_interface.py

This file automatically:

Accepts a scenario name.
Finds the scenario folder.
Finds the .sumocfg file inside that scenario folder.
Starts SUMO.
Connects Python with SUMO using TraCI.
Reads the current traffic state.
Provides traffic information to other project components.
Running a Scenario

Open the terminal inside:

simulation/sumo

Run the program using:

python traffic_interface.py S1_normal
Running Other Scenarios

To run Scenario 2:

python traffic_interface.py S2_equal_opposite

To run Scenario 3:

python traffic_interface.py S3_one_side_heavy

To run Scenario 4:

python traffic_interface.py S4_all_sides_heavy

To run Scenario 5:

python traffic_interface.py S5_changing_traffic

To run Scenario 6:

python traffic_interface.py S6_sudden_surge

To run Scenario 7:

python traffic_interface.py S7_single_ambulance

To run Scenario 8:

python traffic_interface.py S8_two_ambulances
Checking Available Scenarios

If the program is run without specifying a scenario:

python traffic_interface.py

The program displays the available scenario folders.

Example:

Usage:
python traffic_interface.py S1_normal

Available scenarios:

- S1_normal
- S2_equal_opposite
- S3_one_side_heavy
- S4_all_sides_heavy
- S5_changing_traffic
- S6_sudden_surge
- S7_single_ambulance
- S8_two_ambulances
Traffic State Information

The interface provides the following information.

1. Vehicle Counts

The number of vehicles currently present in each direction.

Example:

{
    "north": 5,
    "south": 3,
    "east": 7,
    "west": 2
}

The function used is:

get_vehicle_counts()
2. Queue Lengths

The number of vehicles that are currently stopped or nearly stopped in each direction.

The function used is:

get_queue_lengths()

Example:

{
    "north": 2,
    "south": 1,
    "east": 5,
    "west": 1
}
3. Waiting Times

The total waiting time of vehicles currently present in each direction.

The function used is:

get_waiting_times()

Example:

{
    "north": 20.5,
    "south": 10.0,
    "east": 45.2,
    "west": 8.0
}
4. Current Traffic Light Phase

The current traffic signal phase is obtained using:

get_current_phase()

Example output:

Traffic Light Phase: 0
Complete Traffic State

The main function that combines all traffic information is:

get_traffic_state()

It returns:

{
    "vehicle_counts": {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    },

    "queue_lengths": {
        "north": 0,
        "south": 0,
        "east": 0,
        "west": 0
    },

    "waiting_times": {
        "north": 0.0,
        "south": 0.0,
        "east": 0.0,
        "west": 0.0
    },

    "traffic_light_phase": 0
}

This function is the main interface that can be used by the reinforcement learning and optimization components.

Traffic Light Control

The interface also contains a function for changing the traffic light phase.

change_traffic_light_phase(phase_number)

Example:

change_traffic_light_phase(1)

This function can be used by the reinforcement learning agent to select a traffic signal action.

Lane Definitions

The current interface reads traffic information from the following lanes:

North:
n_t_0
n_t_1

South:
s_t_0
s_t_1

East:
e_t_0
e_t_1

West:
w_t_0
w_t_1

These lane IDs are based on the current SUMO network.

Traffic Light ID

The traffic light ID is obtained automatically using:

get_traffic_light_id()

The interface retrieves the traffic light ID from the SUMO network using TraCI.

Therefore, other team members do not need to manually enter the traffic light ID in their code when using the provided functions.

Simulation Output

During execution, the interface prints traffic information periodically.

Example:

========== STEP 100 ==========

TRAFFIC STATE

Vehicle Counts:
{'north': 4, 'south': 2, 'east': 6, 'west': 3}

Queue Lengths:
{'north': 2, 'south': 1, 'east': 4, 'west': 1}

Waiting Times:
{'north': 15.0, 'south': 5.0, 'east': 30.0, 'west': 8.0}

Traffic Light Phase:
0

The step number represents the current simulation step.

The simulation continues until all expected vehicles have completed the simulation.

Stopping the Simulation

If the simulation needs to be stopped manually, press:

Ctrl + C

The program handles the interruption and closes the TraCI connection.

How Person 2 Can Use This Work

Person 2 can use:

get_traffic_state()

to obtain the current traffic state.

Example:

state = get_traffic_state()

Person 2 can use:

state["vehicle_counts"]
state["queue_lengths"]
state["waiting_times"]
state["traffic_light_phase"]

These values can be used as the state or observation for a reinforcement learning agent.

Person 2 can also use:

change_traffic_light_phase(phase_number)

to apply an action selected by the reinforcement learning model.

Therefore, Person 2 can build the training and decision-making logic on top of the SUMO-TraCI interface without recreating the simulation environment.

How Person 3 Can Use This Work

Person 3 can use the existing SUMO scenarios and interface while working on additional project functionality such as emergency vehicle handling or ambulance priority.

The emergency scenarios are:

S7_single_ambulance

and:

S8_two_ambulances

Person 3 can run these scenarios using:

python traffic_interface.py S7_single_ambulance

or:

python traffic_interface.py S8_two_ambulances

Person 3 can access the SUMO simulation through TraCI and implement logic for identifying emergency vehicles and giving them priority.

Person 3 can also integrate their logic with the trained model produced by Person 2.

The SUMO environment and scenarios created by Person 1 serve as the common simulation base.

How Person 4 Can Use This Work

Person 4 can use the simulation environment for testing and evaluating the complete system.

Person 4 can run different scenarios and compare system performance using metrics such as:

Vehicle waiting time
Queue length
Traffic congestion
Number of vehicles
Traffic signal behavior
Emergency vehicle priority performance

The available scenarios allow testing the system under different traffic conditions.

Team Workflow

The overall workflow of the project is:

Person 1
↓
Creates SUMO Network and Traffic Scenarios
↓
Provides TraCI Interface
↓
Person 2
↓
Uses Traffic State as RL Environment Input
↓
Trains Reinforcement Learning Model
↓
Person 3
↓
Adds Emergency Vehicle / Ambulance Priority Logic
↓
Integrates with Trained Model
↓
Person 4
↓
Tests, Evaluates and Integrates the Complete System
Important Functions for Other Team Members

The following functions are available in:

traffic_interface.py
Start Simulation
start_simulation(scenario_name)
Get Traffic Light ID
get_traffic_light_id()
Get Vehicle Counts
get_vehicle_counts()
Get Queue Lengths
get_queue_lengths()
Get Waiting Times
get_waiting_times()
Get Current Traffic Light Phase
get_current_phase()
Change Traffic Light Phase
change_traffic_light_phase(phase_number)
Get Complete Traffic State
get_traffic_state()

This is the main function recommended for use by other project components.

Important Notes for Developers
Do not move the scenario folders without updating the project structure.
Each scenario folder must contain a .sumocfg file.
The SUMO configuration file must correctly reference the required network and route files.
The common network file should remain available to all scenarios.
The lane IDs used in traffic_interface.py are based on the current SUMO network.
If the network is modified, the lane IDs should be checked again.
The traffic light phases depend on the traffic light program defined in the SUMO network.
Person 2, Person 3, and Person 4 can build their modules on top of the existing simulation interface.
Current Status
Completed
SUMO network setup
Multiple traffic scenarios
Route files for all scenarios
SUMO configuration files
Python TraCI interface
Vehicle count extraction
Queue length extraction
Waiting time extraction
Traffic light phase extraction
Traffic light control function
Scenario selection through command-line argument
Future Work
Reinforcement learning model training
Traffic signal optimization logic
Emergency vehicle detection
Ambulance priority logic
Integration of trained model with SUMO
Performance evaluation
Final system testing
Main Entry Point

The main simulation interface is:

traffic_interface.py

Example:

python traffic_interface.py S1_normal

The scenario can be changed by replacing:

S1_normal

with any available scenario name.

Contribution

Person 1 has completed the SUMO simulation environment and TraCI-based traffic interface.

The remaining team members can use this simulation environment as the common base for reinforcement learning, emergency vehicle handling, optimization, and evaluation.