class Environment:
    def __init__(self, n_actions):
        self.action_space = ActionSpace(n_actions)


class ActionSpace:
    def __init__(self, n_actions):
        self.n = n_actions
        self.range = range(0, n_actions)
