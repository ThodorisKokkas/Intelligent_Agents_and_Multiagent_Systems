import numpy as np

C, D = 0, 1
ACTIONS = (C, D)

def state_index(prevA: int, prevB: int) -> int:
    return (prevA << 1) | prevB

def decode_state(s: int):
    return ((s >> 1) & 1), (s & 1)

PAYOFF = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}

class RepeatedPD:
    def __init__(self, horizon=200, reset_random=False, rng=None):
        self.horizon = horizon
        self.reset_random = reset_random
        self.rng = rng or np.random.default_rng()
        self.t = 0
        self.prevA = C
        self.prevB = C

    def reset(self):
        self.t = 0
        if self.reset_random:
            self.prevA = int(self.rng.integers(0, 2))
            self.prevB = int(self.rng.integers(0, 2))
        else:
            self.prevA, self.prevB = C, C
        return state_index(self.prevA, self.prevB)

    def step(self, aA, aB):
        rA, rB = PAYOFF[(aA, aB)]
        self.prevA, self.prevB = aA, aB
        s_next = state_index(aA, aB)
        self.t += 1
        done = self.t >= self.horizon
        return s_next, float(rA), float(rB), done