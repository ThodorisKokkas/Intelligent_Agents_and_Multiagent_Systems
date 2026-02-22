import numpy as np
from dataclasses import dataclass

@dataclass
class IQLConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_episodes: int = 20000  # linear decay

class IndependentQLearner:
    """
    Independent tabular Q-learning: Q(s,a) with s in {0..3}, a in {0,1}.
    Treats the other agent as part of environment (non-stationary).
    """
    def __init__(self, cfg: IQLConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self.Q = np.zeros((4, 2), dtype=np.float64)

    def epsilon(self, episode: int) -> float:
        if self.cfg.eps_decay_episodes <= 0:
            return float(self.cfg.eps_end)
        frac = min(1.0, episode / self.cfg.eps_decay_episodes)
        return float((1 - frac) * self.cfg.eps_start + frac * self.cfg.eps_end)

    def act(self, s: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(0, 2))
        q = self.Q[s]
        mx = np.max(q)
        best = np.flatnonzero(np.isclose(q, mx))
        return int(self.rng.choice(best))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        target = r if done else (r + self.cfg.gamma * np.max(self.Q[s_next]))
        self.Q[s, a] = (1 - self.cfg.alpha) * self.Q[s, a] + self.cfg.alpha * target