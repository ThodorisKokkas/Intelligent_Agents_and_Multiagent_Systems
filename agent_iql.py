import numpy as np
from dataclasses import dataclass

@dataclass
class IQLConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_episodes: int = 20000

class IndependentQLearner:
    def __init__(self, cfg: IQLConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((4, 2))

    def epsilon(self, episode):
        frac = min(1.0, episode / self.cfg.eps_decay_episodes)
        return (1-frac)*self.cfg.eps_start + frac*self.cfg.eps_end

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(0,2))
        return int(np.argmax(self.Q[s]))

    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.cfg.gamma*np.max(self.Q[s_next])
        self.Q[s,a] += self.cfg.alpha*(target - self.Q[s,a])