import os
import math
import logging
import random
from collections import deque
import torch
import torch.nn as nn


class DDQNAgent:

    def __init__(self, n_actions: int, eps_start: float, eps_final: float, eps_decay: int, memory_size: int,
                 memory_bs: int, discount_factor: float, TAU: float, device: str, logger: logging.Logger):

        self.n_actions = n_actions
        self.device = device
        self.logger = logger
        self.discount_factor = discount_factor

        # DDQN specific parameters
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.eps = eps_start
        self.memory = ReplayBuffer(memory_size)
        self.memory_bs = memory_bs
        self.TAU = TAU
        self.old_epsilon = -1

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        self.checkpoint_epoch = None

    def take_one_action(self, state: torch.Tensor, n_states_total: int) -> int:
        self.eps = self.eps_final + (self.eps_start - self.eps_final) * math.exp(
            -1. * n_states_total / self.eps_decay)
        if random.random() > self.eps:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0))
                action = torch.argmax(q_values, dim=1).squeeze(0).item()
        else:
            action = random.randint(0, self.n_actions - 1)

        return action

    def update_policy_net(self) -> float:
        # Sample a batch of transitions from the replay buffer
        with torch.no_grad():
            states, actions, rewards, next_states, dones = self.memory.sample(self.memory_bs)
            states = torch.stack(states).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
            next_states = torch.stack(next_states).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float, device=self.device)

        # Compute current Q-values using the policy network for the selected actions
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Use the policy network to select the best next action for each next state.
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

        # Use the target network to evaluate the Q-value of those actions.
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(-1).detach()

        # Compute the target Q-values
        targets = rewards + self.discount_factor * next_q_values * (1 - dones)

        # Compute loss using Mean Squared Error
        loss = nn.MSELoss()(current_q_values, targets)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = \
                self.TAU * policy_net_state_dict[key] + (1 - self.TAU) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

        self.target_net.eval()

    def store_transition(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Store a transition in memory"""
        assert not state.requires_grad, "State should not require gradients."
        assert not next_state.requires_grad, "Next_state should not require gradients."

        self.memory.push(state.to('cpu'), action, reward, next_state.to('cpu'), done)

    def save(self, save_dir, epoch, mode='latest'):
        torch.save({
            'epoch': epoch,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(save_dir, f'{mode}_RL_model.pth'))
        memory_data = [(state, action, reward, next_state, done)
                       for state, action, reward, next_state, done in self.memory.buffer]
        torch.save(memory_data, os.path.join(save_dir, f'{mode}_RL_memory.pt'))

    def load(self, save_dir, mode='latest'):
        checkpoint = torch.load(os.path.join(save_dir, f'{mode}_RL_model.pth'))
        self.checkpoint_epoch = checkpoint['epoch']
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        memory_data = torch.load(os.path.join(save_dir, f'{mode}_RL_memory.pt'))
        self.memory = ReplayBuffer(len(memory_data))
        for transition in memory_data:
            self.memory.buffer.append(transition)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
