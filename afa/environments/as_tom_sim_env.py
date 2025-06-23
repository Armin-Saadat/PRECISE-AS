import logging
import torch
from afa.environments.environment import Environment
from afa.datasets import ASTomSimDataset

ACTION_2_VIDEO_IDX = {0: [], 1: [0], 2: [1], 3: [2], 4: [3]}
ACTION_2_COST = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
LABELS_STR_2_INT = {'No AS': 0, 'Early AS': 1, 'Significant AS': 2}


class ASTomSimEnv(Environment):
    def __init__(self, data: ASTomSimDataset, n_actions: int, classifier, lammy: float, max_step: int, device: str,
                 logger: logging.Logger):
        super().__init__(n_actions)
        self.logger = logger
        self.device = device
        self.classifier = classifier
        self.data = data
        self.lammy = lammy
        self.max_step = max_step

        self.current_patient = None
        self.super_state = None
        self.y_true = None
        self.y_pred = None
        self.doppler = None
        self.current_state = None
        self.state_shape = None
        self.action_history = None

        self.reset(0)

    def n_episodes(self):
        return len(self.data)

    def reset(self, idx: int) -> (torch.Tensor, str):
        similarity, label, _, doppler = self.data.__getitem__(idx)

        with torch.no_grad():
            self.super_state = similarity.to(self.device)  # torch.tensor [4, 40], [plax1, plax2, psax1, psax2]
            self.y_true = label
            self.doppler = doppler  # torch.tensor (3,)
            self.current_state = torch.zeros_like(self.super_state)
            self.state_shape = self.current_state.shape

        self.action_history = []

        return self.current_state.clone(), "Environment reset."

    def step(self, action: int) -> (torch.Tensor, float, bool, bool, str):
        assert type(action) == int, f"action type must be int, got type {type(action)}"
        assert action in self.action_space.range, f"action must be in action_space = {self.action_space.range}"

        action_is_available = True
        terminated = False
        reward = 0.0

        with torch.no_grad():
            if action == 0:
                terminated = True
                self.set_y_pred()
                reward = int(self.y_pred == self.y_true) - self.lammy * self.get_cost()
            elif action == 5:
                terminated = True
                reward = int(self.get_y_guideline() == self.y_true) - self.lammy * self.get_cost()
            else:
                if (self.super_state[ACTION_2_VIDEO_IDX[action]] == 0.5).all():
                    action_is_available = False
                else:
                    self.current_state[ACTION_2_VIDEO_IDX[action]] = self.super_state[ACTION_2_VIDEO_IDX[action]]

            if action_is_available:
                self.action_history.append(action)
                info = ""
            else:
                for action in range(1, 5):
                    if (self.super_state[ACTION_2_VIDEO_IDX[action]] == 0.5).all() or action in self.action_history:
                        continue
                    self.current_state[ACTION_2_VIDEO_IDX[action]] = self.super_state[ACTION_2_VIDEO_IDX[action]]
                    self.action_history.append(action)
                terminated = True
                self.set_y_pred()
                reward = int(self.y_pred == self.y_true) - self.lammy * self.get_cost()
                self.action_history.append(0)
                info = "- took all available actions"

        truncated = len(self.action_history) == self.max_step
        if truncated and not terminated:
            self.set_y_pred()
            reward = -self.lammy * self.get_cost()

        return self.current_state.clone(), reward, terminated, truncated, info

    def get_cost(self) -> float:
        cost = 0
        for action in self.action_history:
            cost += ACTION_2_COST[action]

        return cost

    def get_y_guideline(self) -> int:
        if self.doppler[0] >= 1.5 and self.doppler[1] <= 3 and self.doppler[2] <= 20:
            y_guideline = 1  # mild or Early AS
        elif 1.5 > self.doppler[0] > 1 and 4 > self.doppler[1] > 3 and 40 > self.doppler[2] > 20:
            y_guideline = 2  # moderate or Significant AS
        elif self.doppler[0] <= 1 and self.doppler[1] >= 4 and self.doppler[2] >= 40:
            y_guideline = 2  # severe or Significant AS
        else:
            y_guideline = -1

        return y_guideline

    def set_y_pred(self):
        self.y_pred = self.classifier.predict(self.current_state.clone(), self.action_history)

    def render(self):
        pass
