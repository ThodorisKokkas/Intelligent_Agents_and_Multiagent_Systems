import numpy as np
from dataclasses import dataclass

@dataclass
class QMinimaxConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_episodes: int = 20000

class QMinimaxLearner:
    def __init__(self, cfg: QMinimaxConfig, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((4,2,2))

    def epsilon(self, episode):
        frac = min(1.0, episode/self.cfg.eps_decay_episodes)
        return (1-frac)*self.cfg.eps_start + frac*self.cfg.eps_end

    def _v(self, s):
        mins = np.min(self.Q[s], axis=1)
        return np.max(mins)

    def act(self, s, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(0,2))
        mins = np.min(self.Q[s], axis=1)
        return int(np.argmax(mins))

    def update(self, s, a_self, a_opp, r, s_next, done):
        target = r if done else r + self.cfg.gamma*self._v(s_next)
        self.Q[s,a_self,a_opp] += self.cfg.alpha*(target - self.Q[s,a_self,a_opp])