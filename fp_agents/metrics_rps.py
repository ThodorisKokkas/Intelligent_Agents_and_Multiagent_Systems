# metrics_rps.py
import numpy as np

NASH_UNIFORM = np.array([1/3, 1/3, 1/3], dtype=float)

def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))

def distance_to_nash(row_dist: np.ndarray, col_dist: np.ndarray) -> float:
    return l1_distance(row_dist, NASH_UNIFORM) + l1_distance(col_dist, NASH_UNIFORM)

def expected_row_value(payoff_row: np.ndarray, row_dist: np.ndarray, col_dist: np.ndarray) -> float:
    # v = sum_{i,j} row_dist[i]*col_dist[j]*payoff_row[i,j]
    return float(row_dist @ payoff_row @ col_dist)

def exploitability_zero_sum(payoff_row: np.ndarray, row_dist: np.ndarray, col_dist: np.ndarray) -> float:
    """
    One common measure in zero-sum: total exploitability relative to best responses.
    Row exploitability: BR_value(col_dist) - V
    Col exploitability: V - min_row_value(row_dist) (equivalently BR for col)
    Here we compute both and sum.
    """
    V = expected_row_value(payoff_row, row_dist, col_dist)

    # Row best response value vs col_dist: max_a e_a^T A col_dist
    br_row = float(np.max(payoff_row @ col_dist))

    # Col best response corresponds to minimizing row value: min_b row_dist^T A e_b
    # row_dist^T A is a vector over col actions
    min_for_col = float(np.min(row_dist @ payoff_row))

    # If col best-responds, row value goes down to min_for_col
    e_row = br_row - V
    e_col = V - min_for_col
    return float(e_row + e_col)