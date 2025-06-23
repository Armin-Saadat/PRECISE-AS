import logging
import torch.optim as optim
import torch.nn as nn
from afa.agents.DDQN_agent import DDQNAgent


class ASTomDDQNAgent(DDQNAgent):

    def __init__(self, state_shape: tuple, n_actions: int, lr: float, milestones: list, gamma: float, eps_start: float,
                 eps_final: float, eps_decay: int, memory_size: int, memory_bs: int, discount_factor: float,
                 device: str, TAU: float, logger: logging.Logger):
        super().__init__(n_actions, eps_start, eps_final, eps_decay, memory_size, memory_bs, discount_factor, TAU,
                         device, logger)

        self.policy_net = FCN(state_shape, self.n_actions).to(device)
        self.target_net = FCN(state_shape, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=milestones, gamma=gamma)


class FCN(nn.Module):
    def __init__(self, input_dim: tuple, n_actions: int):
        super(FCN, self).__init__()

        h = 1
        for i in input_dim:
            h = h * i

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions)
        )

    def forward(self, x):
        return self.fc(x)
