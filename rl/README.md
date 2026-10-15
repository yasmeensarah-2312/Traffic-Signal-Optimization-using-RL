# Reinforcement Learning Module (Person 2)

This folder contains the Deep Q-Network (DQN) agent that learns adaptive
traffic signal control on top of Person 1's SUMO simulation.

## Files

| File | Purpose |
|---|---|
| `traffic_env.py` | Wraps Person 1's `traffic_interface.py` into a simple `reset()` / `step()` environment, so the DQN doesn't need to know anything about SUMO or TraCI directly. |
| `dqn_agent.py` | The Q-network (neural network), replay buffer, and training logic. |
| `train.py` | Runs the full training loop and saves the model + logs. |
| `evaluate.py` | Runs a saved model with exploration off on every scenario and compares it with a random policy. |
| `validate_reward.py` | Checks that the reward agrees with measured waiting time, queue length and throughput. |

## How to run training

```
cd rl
python train.py
```

Trains for the number of episodes set in `train.py` (`NUM_EPISODES`), rotating
through all 6 non-ambulance traffic scenarios so the agent generalizes instead
of memorizing one traffic pattern. Set `GUI = True` in `train.py` if you want
to watch the simulation while training (much slower).

Outputs:
- Trained model → `../models/dqn_traffic_model.pt`
- Training log (CSV) → `../results/training_log.csv`

## State representation (16 values)

For each of the 4 directions (north, south, east, west):
- Vehicle count (normalized)
- Queue length (normalized)
- Waiting time (normalized)

Plus a 4-value one-hot encoding of which green phase is currently active.

## Action space (4 actions)

The intersection has 8 signal phases, but only 4 are valid green phases an
agent can choose: phases `0, 2, 4, 6`. The other 4 (`1, 3, 5, 7`) are yellow
transition phases. `traffic_env.py` automatically inserts the correct yellow
phase when switching between two different green phases — the DQN only ever
picks an action `0–3`, and never touches phase numbers or yellow timing
directly.

## Reward function

`reward = -(average queue length over the decision interval) / 10`

- Queue length = vehicles stopped on the incoming lanes (SUMO "halting
  vehicles"), summed over the four directions.
- It is sampled at every simulation step of a decision, yellow steps
  included, then averaged.
- The reward is always <= 0; closer to 0 means shorter queues.
- The divisor 10 (`REWARD_QUEUE_SCALE` in `traffic_env.py`) only keeps reward
  values small enough for stable DQN training.

**Why not the earlier version.** The first version used
`waiting_time_before - waiting_time_after`, where waiting time is summed over
the vehicles currently on the network. That number also changes when vehicles
enter or leave, not only because of the chosen phase, and over an episode it
mostly measures how the total changed between start and end rather than how
low it stayed. It also ignored the yellow steps, so switching phases looked
free. The queue-based reward measures how bad traffic was while the action was
in effect.

**Validation.** `validate_reward.py` plays policies of different quality
(stuck on one phase, random, round-robin, and the DQN if a model exists) on
every scenario and checks that episode return is negatively correlated with
measured average waiting time and queue length and positively with
throughput. It scores the old reward on the same episodes for comparison.
Output: `../results/reward_validation.csv`.

## Training scenarios

Trained on: `S1_normal`, `S2_equal_opposite`, `S3_one_side_heavy`,
`S4_all_sides_heavy`, `S5_changing_traffic`, `S6_sudden_surge`.

`S7_single_ambulance` and `S8_two_ambulances` are intentionally **not** used
for this training — those are reserved for Person 3's ambulance-priority
override integration, where the DQN's normal control gets temporarily
interrupted.

## Results summary (60-episode run, 6 scenarios × 10 passes)

Average waiting time improved across every scenario from the first 5 passes
to the last 5 passes:

| Scenario | Improvement |
|---|---|
| S1_normal | ↓ 27% |
| S2_equal_opposite | ↓ 22% |
| S3_one_side_heavy | ↓ 15% |
| S4_all_sides_heavy | ↓ 6% |
| S5_changing_traffic | ↓ 16% |
| S6_sudden_surge | ↓ 6% |

Full per-episode numbers are in `../results/training_log.csv`.

**Note:** this was trained with epsilon (exploration rate) only decaying to
~0.74 by the end — the agent was still choosing randomly most of the time.
Further training (more episodes, or faster epsilon decay) would likely
improve results further before final submission.

## For Person 4 (baseline comparison)

Everything needed for the fixed-time vs. DQN comparison is in:
- `../results/training_log.csv` — episode, scenario, reward, avg waiting
  time, avg queue length, throughput, epsilon
- `../models/dqn_traffic_model.pt` — trained model weights (load via
  `DQNAgent.load()` in `dqn_agent.py`)