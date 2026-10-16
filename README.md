# Intelligent Traffic Signal Optimization Using Deep Reinforcement Learning

An intelligent traffic signal optimization system using Deep Q-Networks (DQN), SUMO, and YOLO-based ambulance detection to support adaptive traffic signal control and emergency vehicle priority.

## 1. Project Overview

Conventional traffic signals often use fixed timings that cannot adapt effectively to changing traffic conditions. This project explores adaptive signal control using Deep Reinforcement Learning, with an additional mechanism to prioritize emergency ambulances.

The system combines traffic simulation, reinforcement learning, vehicle perception, and emergency signal control.

### Objectives

* Adapt traffic signal decisions to changing traffic conditions.
* Use a DQN agent to select traffic signal actions.
* Evaluate adaptive control against a fixed-time traffic signal baseline.
* Detect ambulances using a YOLO-based perception model.
* Prioritize detected ambulances while maintaining safe signal transitions.
* Resume normal DQN-controlled traffic management after emergency priority is released.

## 2. System Architecture

The system consists of the following components:

1. **Traffic simulation:** SUMO models the road network, vehicles, traffic signals, and vehicle movement.
2. **Traffic state collection:** Simulation information is used to represent traffic conditions.
3. **Reinforcement learning:** A DQN agent selects traffic signal actions based on the observed state.
4. **Vehicle perception:** YOLO-based detection supports ambulance identification.
5. **Emergency priority:** The emergency controller identifies the required signal phase and manages priority for approaching ambulances.
6. **Safe transitions:** Signal changes use transition logic to avoid directly switching between conflicting green phases.
7. **Evaluation:** DQN and fixed-time control are evaluated using traffic performance metrics.

### Emergency priority workflow

Ambulance detection → Determine the required signal phase → Safely transition the signal → Grant emergency priority → Confirm ambulance arrival → Release priority → Resume DQN control.

## 3. Technologies Used

* Python
* Deep Q-Networks (DQN)
* Reinforcement Learning
* SUMO — Simulation of Urban MObility
* TraCI — Traffic Control Interface
* YOLO-based object detection
* PyTorch
* OpenCV
* Pandas and Matplotlib for evaluation and visualization

## 4. Prerequisites

Before running the project, ensure that you have:

* Python installed.
* SUMO installed and its executable paths configured.
* A Python virtual environment with the project's required packages installed.
* The repository's simulation configuration files and required assets.
* The trained model files required by the selected controller.
* The ambulance detection model and associated configuration/data files for perception tests.

**Windows configuration:** The current traffic interface uses the following SUMO executable paths:

* GUI: `C:\Program Files (x86)\SUMO\bin\sumo-gui.exe`
* Command line: `C:\Program Files (x86)\SUMO\bin\sumo.exe`

If SUMO is installed elsewhere, update the paths in `simulation/sumo/traffic_interface.py`.

Use the Python environment and dependencies configured for this repository. Do not retrain or overwrite existing trained models merely to run evaluation.

## 5. Running the Project

Run commands from the repository root directory.

Activate the project's virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### A. Evaluate the DQN controller

```powershell
python evaluation/evaluate_dqn.py
```

The evaluation script runs the DQN controller across the configured traffic scenarios.

### B. Evaluate the fixed-time baseline

```powershell
python evaluation/evaluate_fixed_time.py
```

Use the baseline results to compare traffic performance against the adaptive controller.

### C. Test standalone emergency priority (S7)

```powershell
python simulation/sumo/emergency_controller.py
```

This test exercises the standalone emergency controller, including ambulance detection, signal priority, and priority release.

### D. Test two-ambulance DQN integration (S8)

```powershell
python rl/test_emergency_integration_s8.py
```

This integration test compares operation with emergency priority disabled and enabled. It checks ambulance completion, priority activation and release, safe signal transitions, and the resumption of DQN decisions.

### E. Test ambulance detection

```powershell
python test_ambulance_detection.py
```

Ensure the required detection model, video assets, and configuration files are available before running this test.

**Note:** The commands above identify the evaluation and test entry points. Successful execution depends on the local SUMO installation, model files, configuration files, and Python dependencies.

## 6. Evaluation Metrics

The project records or examines traffic and emergency-response measures, including:

* Average queue length.
* Vehicle waiting time or delay.
* Vehicle throughput.
* Simulation duration.
* Ambulance travel time.
* Emergency detections and priority releases.
* Safe and unsafe signal transitions.
* DQN decisions after emergency priority is released.

Metric definitions and aggregation methods should be considered when comparing results from different controllers.

## 7. Preliminary Evaluation

The recorded S8 integration test produced the following results:

| Metric                                        | Priority Disabled | Priority Enabled |
| --------------------------------------------- | ----------------: | ---------------: |
| Ambulances completed                          |               2/2 |              2/2 |
| Mean ambulance travel time (simulation steps) |                38 |               28 |
| Emergency detections                          |                 0 |                2 |
| Priority releases                             |                 0 |                2 |
| Unsafe transitions reported                   |                 0 |                0 |

In this recorded run, mean ambulance travel time decreased by 10 simulation steps, approximately 26.3%, when emergency priority was enabled.

These results are preliminary and come from a single recorded run. They do not establish statistical significance or guarantee the same improvement across different traffic conditions.

The DQN and fixed-time evaluations were also run on scenarios S1–S6. Their recorded outputs are available in the `results/` directory.

Because simulation durations and other evaluation conditions can differ, comparisons should be interpreted carefully. Multiple controlled runs with consistent metric definitions and comparable conditions are needed for stronger conclusions.

## 8. Results and Evaluation Files

Relevant tracked outputs include:

* `results/dqn_evaluation.csv` — DQN evaluation results.
* `results/fixed_time_evaluation_v2.csv` — Fixed-time baseline evaluation results.
* `results/dqn_evaluation_log.txt` — DQN evaluation log.
* `results/baseline_code_review.txt` — Baseline code review notes.

Additional local datasets, backups, and training artifacts may be present in the working directory.

## 9. Important Directories

| Directory or file  | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `rl/`              | Reinforcement learning controllers and integration tests |
| `simulation/sumo/` | SUMO traffic simulation and emergency control            |
| `perception/`      | Ambulance detection and tracking components              |
| `evaluation/`      | DQN and fixed-time evaluation scripts                    |
| `results/`         | Evaluation outputs, logs, and review notes               |
| `models/`          | Trained model files, where present                       |
| `data/`            | Local datasets and testing assets, where present         |
| `runs/`            | Local model training outputs, where present              |

The exact contents of these directories depend on the checked-out repository version and the files available locally.

## 10. Reproducibility and Limitations

For reproducible evaluation:

1. Use the same SUMO network, traffic scenarios, and simulation configuration.
2. Preserve the trained model files used for the experiment.
3. Use consistent evaluation settings and metric definitions.
4. Record the configuration, software versions, and random seeds where applicable.
5. Repeat experiments across multiple runs.
6. Report both improvements and any trade-offs in traffic performance.
7. Keep dataset licensing and model provenance documented.

The current emergency-priority evaluation is an initial integration test rather than a comprehensive statistical experiment. Further repeated evaluations are needed to assess robustness across traffic scenarios.

## 11. Future Work

Potential extensions include:

* Repeated evaluation across multiple traffic-demand conditions.
* Statistical comparison of DQN and fixed-time controllers.
* Further analysis of emergency response time and traffic-flow trade-offs.
* Improved configuration portability and automated experiment execution.
* Additional validation of ambulance detection under varied visual conditions.

## 12. Project Status

The project has completed initial DQN and fixed-time evaluations on scenarios S1–S6, a standalone emergency-priority test, and a two-ambulance emergency integration test.

The final reproducibility, documentation, and repository audit remains necessary before treating the project as fully complete.
