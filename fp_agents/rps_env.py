# rps_env.py
import numpy as np

class RPSEnv:
    """
    Rock–Paper–Scissors (zero-sum).
    Actions:
      0 -> Rock
      1 -> Paper
      2 -> Scissors

    Payoffs: row gets +1 for win, -1 for loss, 0 for tie. col gets -row.
    """

    def __init__(self):
        self.n_actions = 3

        # payoff[row_action, col_action] = row reward
        self.payoff_row = np.array([
            [ 0, -1,  1],  # Rock vs (R,P,S)
            [ 1,  0, -1],  # Paper
            [-1,  1,  0],  # Scissors
        ], dtype=float)

    def step(self, a_row: int, a_col: int):
        r_row = float(self.payoff_row[a_row, a_col])
        r_col = -r_row
        return r_row, r_col