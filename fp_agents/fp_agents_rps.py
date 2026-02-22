# fp_agents_rps.py
import numpy as np

def normalize_counts(counts: np.ndarray) -> np.ndarray:
    s = counts.sum()
    if s <= 0:
        return np.ones_like(counts, dtype=float) / len(counts)
    return counts.astype(float) / s

class FictitiousPlayAgentRPS:
    """
    Fictitious Play for RPS.
    Keeps counts of opponent actions (empirical distribution), best-responds to that belief.
    Tie-breaking: uniform random among best actions.
    """

    def __init__(self, payoff_row: np.ndarray, player_id: int, seed: int | None = None):
        """
        payoff_row shape: (3,3) gives ROW payoff.
        player_id: 0 for row, 1 for col.
          - If player_id=1 (col), its utility is -payoff_row (zero-sum).
        """
        self.payoff_row = payoff_row
        self.player_id = player_id
        self.rng = np.random.default_rng(seed)

        self.n_actions = payoff_row.shape[0]
        self.opp_counts = np.zeros(self.n_actions, dtype=int)
        self.my_counts = np.zeros(self.n_actions, dtype=int)

    def belief_about_opponent(self) -> np.ndarray:
        return normalize_counts(self.opp_counts)

    def policy_empirical(self) -> np.ndarray:
        return normalize_counts(self.my_counts)

    def expected_payoff_if_play(self, my_action: int, opp_dist: np.ndarray) -> float:
        """
        Expected payoff for this player if plays my_action vs opponent distribution.
        """
        if self.player_id == 0:
            # row utility
            return float(np.dot(self.payoff_row[my_action, :], opp_dist))
        else:
            # col utility = -row utility, and my_action here is COL action
            # E[u_col] = sum_row opp_dist[row] * (-payoff_row[row, my_action])
            return float(np.dot(-self.payoff_row[:, my_action], opp_dist))

    def best_response(self, opp_dist: np.ndarray) -> int:
        vals = np.array([self.expected_payoff_if_play(a, opp_dist) for a in range(self.n_actions)], dtype=float)
        best = np.max(vals)
        best_actions = np.flatnonzero(np.isclose(vals, best))
        return int(self.rng.choice(best_actions))

    def act(self) -> int:
        opp_dist = self.belief_about_opponent()
        a = self.best_response(opp_dist)
        self.my_counts[a] += 1
        return a

    def observe_opponent(self, opp_action: int):
        self.opp_counts[opp_action] += 1