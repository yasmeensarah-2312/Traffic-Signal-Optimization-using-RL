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

Settings used for the final (v2) run: 150 episodes (25 passes through the 6
scenarios). Epsilon (the chance of a random exploratory action) starts at 1.0
and decays so that it reaches 0.05 after 80% of the episodes; the last 20%
train with almost no random actions.

Outputs:
- Trained model → `../models/dqn_traffic_model_v2.pt`
- Training log (CSV) → `../results/training_log_v2.csv`

The training log is measured **while the agent is still exploring**, so it
shows training progress only. Use `evaluate.py` for performance numbers.

Note for contributors: the repo's `.gitignore` ignores `*.pt` files (meant for
YOLO weights), so the model must be added with
`git add -f models/dqn_traffic_model_v2.pt`.

## Using the trained model

```python
from dqn_agent import DQNAgent

agent = DQNAgent()
agent.load("../models/dqn_traffic_model_v2.pt")
action = agent.select_action(state, greedy=True)   # greedy=True: no random actions
```

Always pass `greedy=True` when using the model. A fresh agent starts with
epsilon 1.0, so without it the agent would act randomly.

To measure performance: `python evaluate.py` runs the saved model with
exploration off on every scenario, plus a random policy as a reference, and
writes `../results/evaluation_results.csv`.

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

Result: the new reward correlated at -0.99 with waiting time, -1.00 with
queue length and +0.66 with throughput. The old reward scored almost the same
(-1.00, -0.99, +0.67), so this test does not separate the two. It is a sanity
check that the new reward agrees with measured performance on simple
policies. The case for the change rests on how the rewards are defined (see
above), not on this table.

## Training scenarios

Trained on: `S1_normal`, `S2_equal_opposite`, `S3_one_side_heavy`,
`S4_all_sides_heavy`, `S5_changing_traffic`, `S6_sudden_surge`.

`S7_single_ambulance` and `S8_two_ambulances` are intentionally **not** used
for this training — those are reserved for Person 3's ambulance-priority
override integration, where the DQN's normal control gets temporarily
interrupted.

## Results (final v2 model, exploration off)

Measured with `evaluate.py` on all 6 scenarios. The random policy is the
average of 3 runs per scenario; the DQN is a single greedy run, because the
SUMO scenarios run identically each time. Raw numbers:
`../results/evaluation_results.csv`.

| Scenario | Avg waiting time (random → DQN) | Avg queue length | Throughput |
|---|---|---|---|
| S1_normal | 209.8 → 112.7 (-46%) | 15.5 → 10.9 (-29%) | 239 → 259 (+9%) |
| S2_equal_opposite | 321.9 → 249.9 (-22%) | 21.5 → 20.1 (-6%) | 221 → 273 (+23%) |
| S3_one_side_heavy | 261.2 → 145.9 (-44%) | 15.7 → 14.4 (-8%) | 132 → 159 (+20%) |
| S4_all_sides_heavy | 971.0 → 2214.0 (**+128%**) | 60.6 → 50.9 (-16%) | 296 → 260 (**-12%**) |
| S5_changing_traffic | 593.3 → 271.9 (-54%) | 29.3 → 22.0 (-25%) | 249 → 275 (+11%) |
| S6_sudden_surge | 104.0 → 76.5 (-26%) | 9.2 → 8.3 (-10%) | 113 → 120 (+7%) |

For waiting time, lower is better. For throughput, higher is better.

## Known limitations

- **S4_all_sides_heavy is worse than a random policy on total waiting time**
  (2214 vs 971) and throughput, even though its queue length is lower. The
  reward counts how many vehicles are stopped, not how long each has waited,
  so the agent may keep serving the largest queues while a smaller approach
  waits a long time. This explanation is an inference from the totals and has
  not been confirmed with a per-direction check. A fairness or delay term in
  the reward is the likely fix.
- Random is a weak reference. The comparison that matters for the project is
  against Person 4's fixed-time baseline.
- The DQN numbers come from one run per scenario. Scenarios are deterministic,
  so repeats would be identical.
- The model was trained and evaluated on the same 6 scenarios, so these
  results do not show how it behaves on traffic patterns it has not seen.

## For Person 4 and Person 3

- Model: `../models/dqn_traffic_model_v2.pt` (load as shown above, with
  `greedy=True`).
- DQN results for the comparison with the fixed-time baseline:
  `../results/evaluation_results.csv`. Do not use the training log for this,
  because it was measured while exploring.
- The first model (`dqn_traffic_model.pt`, old waiting-time reward) is
  superseded; its numbers are not comparable with v2.
- Throughput now also counts vehicles that finish during yellow steps, so it
  is slightly higher than in the first training log for the same behaviour.