"""
traffic_env.py

This is Person 2's wrapper around Person 1's traffic_interface.py.

WHY THIS FILE EXISTS:
traffic_interface.py gives us raw information (dictionaries) and one
raw action (change_traffic_light_phase). A DQN needs something simpler
and more structured:

    state  = env.reset()              -> a list of numbers
    state, reward, done = env.step(action)   -> repeat this loop

This file provides exactly that, by wrapping the functions Person 1
already built. It does not duplicate any of Person 1's SUMO/TraCI code.

Put this file inside your project's rl/ folder:
    Traffic-Signal-Optimization-using-RL-main/rl/traffic_env.py
"""

import os
import sys
import numpy as np

# ============================================================
# LOCATE AND IMPORT PERSON 1'S INTERFACE
#
# traffic_interface.py lives in simulation/sumo, which is a
# SIBLING folder to rl/. We need to tell Python where to find it.
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_INTERFACE_DIR = os.path.join(CURRENT_DIR, "..", "simulation", "sumo")
sys.path.append(SUMO_INTERFACE_DIR)

import traffic_interface as ti   # Person 1's file
import traci                     # needed directly for simulationStep(), close()


class TrafficEnv:
    """
    A simplified, DQN-friendly wrapper around the SUMO intersection.

    Usage:
        env = TrafficEnv(scenario_name="S1_normal", gui=False)
        state = env.reset()
        next_state, reward, done, info = env.step(action)
        env.close()
    """

    # The traffic light has 8 phases (0-7). Only these 4 are real
    # "green" phases that we allow the DQN to choose as an action.
    # The others (1,3,5,7) are yellow transition phases only.
    GREEN_PHASES = [0, 2, 4, 6]

    # Which yellow phase must play BEFORE leaving each green phase.
    # e.g. if we're currently on phase 0 (NS green) and want to
    # switch to a different green, we must pass through phase 1 (yellow) first.
    YELLOW_FOR_GREEN = {0: 1, 2: 3, 4: 5, 6: 7}

    # How many simulation steps to hold the yellow phase.
    YELLOW_DURATION = 4

    # How many simulation steps to hold the chosen green phase
    # before asking the DQN to decide again. This is one "decision interval".
    GREEN_DURATION = 10

    # How many steps to let traffic build up before the first decision,
    # so the very first state isn't just an empty, freshly-started road.
    WARMUP_STEPS = 20

    # Rough normalization constants so state values stay in a
    # reasonable range for the neural network (roughly 0-1 range).
    # These are estimates -- tune later if needed once you see real values.
    NORM_VEHICLE_COUNT = 20.0
    NORM_QUEUE_LENGTH = 20.0
    NORM_WAITING_TIME = 300.0

    DIRECTIONS = ["north", "south", "east", "west"]

    def __init__(self, scenario_name, gui=False, max_steps=3000):
        self.scenario_name = scenario_name
        self.gui = gui
        self.max_steps = max_steps
        self.current_step = 0

    # ------------------------------------------------------------
    # RESET: start a fresh simulation, return the first state
    # ------------------------------------------------------------
    def reset(self, scenario_name=None):
        """
        scenario_name: optional. If given, switches to that scenario for
        this episode (used for multi-scenario training). If omitted, reuses
        whatever scenario the environment was created with.
        """
        if scenario_name is not None:
            self.scenario_name = scenario_name

        # If a previous simulation is still connected, close it first.
        # (traci only allows one active connection at a time.)
        try:
            traci.close()
        except Exception:
            pass

        ti.start_simulation(self.scenario_name, gui=self.gui)

        self.current_step = 0

        # Let some vehicles enter the roads before the first decision.
        for _ in range(self.WARMUP_STEPS):
            traci.simulationStep()
            self.current_step += 1

        return self._get_state_vector()

    # ------------------------------------------------------------
    # STEP: apply one action, advance simulation, return the result
    # ------------------------------------------------------------
    def step(self, action):
        """
        action: an integer 0-3, selecting an index into GREEN_PHASES.
        """
        target_phase = self.GREEN_PHASES[action]
        current_phase = ti.get_current_phase()

        # Only insert a yellow transition if we are actually
        # switching to a DIFFERENT green phase.
        if current_phase in self.GREEN_PHASES and current_phase != target_phase:
            yellow_phase = self.YELLOW_FOR_GREEN[current_phase]
            ti.change_traffic_light_phase(yellow_phase)

            for _ in range(self.YELLOW_DURATION):
                traci.simulationStep()
                self.current_step += 1

        # Now switch to (or stay on) the chosen green phase.
        ti.change_traffic_light_phase(target_phase)

        waiting_before = self._total_waiting_time()
        arrived_count = 0

        for _ in range(self.GREEN_DURATION):
            traci.simulationStep()
            self.current_step += 1
            # Vehicles that completed their route and left the simulation
            # this step -- this is our "throughput" metric for Person 4.
            arrived_count += traci.simulation.getArrivedNumber()

        waiting_after = self._total_waiting_time()
        queue_after = sum(ti.get_queue_lengths().values())

        # Reward design (simple, beginner-friendly version):
        # If total waiting time went DOWN, reward is positive.
        # If it went UP, reward is negative.
        # This directly teaches the agent: "reduce waiting time."
        reward = waiting_before - waiting_after

        done = (
            traci.simulation.getMinExpectedNumber() <= 0
            or self.current_step >= self.max_steps
        )

        next_state = self._get_state_vector()

        info = {
            "waiting_before": waiting_before,
            "waiting_after": waiting_after,
            "total_queue_length": queue_after,
            "throughput": arrived_count,
            "step": self.current_step,
        }

        return next_state, reward, done, info

    # ------------------------------------------------------------
    # CLOSE: shut down the SUMO connection cleanly
    # ------------------------------------------------------------
    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    # ------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------
    def _total_waiting_time(self):
        waiting_times = ti.get_waiting_times()
        return sum(waiting_times.values())

    def _get_state_vector(self):
        """
        Converts Person 1's dictionary-based state into a flat,
        normalized numpy array the neural network can read.

        Final vector layout (16 numbers total):
            [0:4]   vehicle counts   (north, south, east, west)
            [4:8]   queue lengths    (north, south, east, west)
            [8:12]  waiting times    (north, south, east, west)
            [12:16] current green phase, one-hot encoded
        """
        state = ti.get_traffic_state()

        vehicle_counts = state["vehicle_counts"]
        queue_lengths = state["queue_lengths"]
        waiting_times = state["waiting_times"]
        phase = state["traffic_light_phase"]

        vector = []

        for d in self.DIRECTIONS:
            vector.append(vehicle_counts[d] / self.NORM_VEHICLE_COUNT)

        for d in self.DIRECTIONS:
            vector.append(queue_lengths[d] / self.NORM_QUEUE_LENGTH)

        for d in self.DIRECTIONS:
            vector.append(waiting_times[d] / self.NORM_WAITING_TIME)

        # One-hot encode which green phase is currently active.
        # If we're mid-yellow-transition, none of these will be 1 --
        # that's fine, it just means "not currently in a stable green phase".
        phase_onehot = [0, 0, 0, 0]
        if phase in self.GREEN_PHASES:
            phase_onehot[self.GREEN_PHASES.index(phase)] = 1
        vector.extend(phase_onehot)

        return np.array(vector, dtype=np.float32)


# ============================================================
# QUICK SANITY TEST
#
# Run this file directly to confirm the wrapper works BEFORE
# writing any DQN code:
#
#     cd rl
#     python traffic_env.py
# ============================================================
if __name__ == "__main__":

    env = TrafficEnv(scenario_name="S1_normal", gui=True, max_steps=500)

    state = env.reset()
    print("Initial state vector:")
    print(state)
    print("State size:", len(state))

    total_reward = 0

    for i in range(10):
        action = np.random.randint(0, 4)  # random action for testing only
        next_state, reward, done, info = env.step(action)

        total_reward += reward

        print(f"\nStep {i} | Action {action} | Reward {reward:.2f} | Done {done}")
        print("Next state:", next_state)

        if done:
            print("\nEpisode ended early.")
            break

    print("\nTotal reward over test run:", total_reward)

    env.close()
    print("\nEnvironment closed. Test complete.")