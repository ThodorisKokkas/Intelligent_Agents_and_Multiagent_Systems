# plotting_rps.py
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = min(window, len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")

def plot_rps(
    pay_row: np.ndarray,
    row_pol: np.ndarray,
    col_pol: np.ndarray,
    dist_nash: np.ndarray,
    expl: np.ndarray,
    window: int = 300,
):
    # 1) Payoffs (row only, col is -row)
    plt.figure()
    plt.plot(moving_average(pay_row, window), label="Row payoff (MA)")
    plt.title("RPS: average payoff (moving average)")
    plt.xlabel("Episode")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)

    # 2) Empirical policy components
    plt.figure()
    plt.plot(row_pol[:, 0], label="Row P(R)")
    plt.plot(row_pol[:, 1], label="Row P(P)")
    plt.plot(row_pol[:, 2], label="Row P(S)")
    plt.title("Row empirical strategy distribution")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(col_pol[:, 0], label="Col P(R)")
    plt.plot(col_pol[:, 1], label="Col P(P)")
    plt.plot(col_pol[:, 2], label="Col P(S)")
    plt.title("Col empirical strategy distribution")
    plt.xlabel("Episode")
    plt.ylabel("Probability")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(True)

    # 3) Distance to Nash
    plt.figure()
    plt.plot(dist_nash, label="L1 distance to uniform Nash (sum players)")
    plt.title("Deviation from Nash (uniform mixed)")
    plt.xlabel("Episode")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)

    # 4) Exploitability
    plt.figure()
    plt.plot(moving_average(expl, window), label="Exploitability (MA)")
    plt.title("Exploitability (zero-sum)")
    plt.xlabel("Episode")
    plt.ylabel("Exploitability")
    plt.legend()
    plt.grid(True)

    plt.show()