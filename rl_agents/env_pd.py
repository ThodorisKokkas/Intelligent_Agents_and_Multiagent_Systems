import numpy as np

# Actions
C, D = 0, 1
ACTIONS = (C, D)

def state_index(prevA: int, prevB: int) -> int:
    # CC=0, CD=1, DC=2, DD=3 (second letter is B)
    return (prevA << 1) | prevB

def decode_state(s: int):
    return ((s >> 1) & 1), (s & 1)

# Prisoner's Dilemma payoff matrix: (A_reward, B_reward)
PAYOFF = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}

class RepeatedPD:
    """
    Repeated Prisoner's Dilemma with state = previous joint action (prevA, prevB).
    Episode horizon = T steps.
    """
    def __init__(self, horizon: int = 200, reset_random: bool = False, rng: np.random.Generator | None = None):
        self.horizon = int(horizon)
        self.reset_random = bool(reset_random)
        self.rng = rng or np.random.default_rng()
        self.t = 0
        self.prevA = C
        self.prevB = C

    def reset(self) -> int:
        self.t = 0
        if self.reset_random:
            self.prevA = int(self.rng.integers(0, 2))
            self.prevB = int(self.rng.integers(0, 2))
        else:
            self.prevA, self.prevB = C, C
        return state_index(self.prevA, self.prevB)

    def step(self, aA: int, aB: int):
        rA, rB = PAYOFF[(aA, aB)]
        self.prevA, self.prevB = aA, aB
        s_next = state_index(self.prevA, self.prevB)

        self.t += 1
        done = (self.t >= self.horizon)
        return s_next, float(rA), float(rB), done