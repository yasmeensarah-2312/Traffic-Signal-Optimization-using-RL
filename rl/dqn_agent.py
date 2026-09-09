"""
dqn_agent.py

The learning "brain" that decides traffic signal actions.
Builds on traffic_env.py -- this file has no SUMO code in it at all,
it only deals with the neural network and the learning logic.

See the class docstrings below for what each piece does.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    A small feed-forward neural network.
    Input:  state vector (16 numbers, from traffic_env.py)
    Output: one Q-value per action (4 numbers)
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores past (state, action, reward, next_state, done) experiences
    and lets us sample random batches from them for training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Wraps the Q-network, target network, replay buffer, and training
    logic into one object. This is what train.py will create and use.
    """

    def __init__(
        self,
        state_size=16,
        action_size=4,
        hidden_size=64,
        lr=0.001,
        gamma=0.95,               # how much future reward matters vs. immediate reward
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_every=200,  # how often (in training steps) to sync target network
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.train_step_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_capacity)

    def select_action(self, state):
        """Epsilon-greedy: mostly random early on, mostly network-driven later."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def train_step(self):
        """
        One batch of learning.
        Returns the loss value, or None if we don't have enough
        stored experiences yet to fill a batch.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-value the network currently predicts for the action we actually took
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Best possible Q-value for the NEXT state, using the (stable) target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())