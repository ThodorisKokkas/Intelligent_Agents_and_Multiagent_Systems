# main_rps_fp_vs_fp.py
import numpy as np

from rps_env import RPSEnv
from fp_agents_rps import FictitiousPlayAgentRPS
from metrics_rps import distance_to_nash, exploitability_zero_sum
from plotting_rps import plot_rps

EPISODES = 5000
SEED = 11
MA_WINDOW = 400

def run():
    env = RPSEnv()

    row = FictitiousPlayAgentRPS(payoff_row=env.payoff_row, player_id=0, seed=SEED)
    col = FictitiousPlayAgentRPS(payoff_row=env.payoff_row, player_id=1, seed=SEED + 1)

    pay_row = np.zeros(EPISODES, dtype=float)

    row_pol_hist = np.zeros((EPISODES, 3), dtype=float)
    col_pol_hist = np.zeros((EPISODES, 3), dtype=float)

    dist_nash = np.zeros(EPISODES, dtype=float)
    expl = np.zeros(EPISODES, dtype=float)

    for t in range(EPISODES):
        a_row = row.act()
        a_col = col.act()

        r_row, _ = env.step(a_row, a_col)
        pay_row[t] = r_row

        # after game: each sees opponent action
        row.observe_opponent(a_col)
        col.observe_opponent(a_row)

        # empirical strategies so far
        rp = row.policy_empirical()
        cp = col.policy_empirical()
        row_pol_hist[t] = rp
        col_pol_hist[t] = cp

        dist_nash[t] = distance_to_nash(rp, cp)
        expl[t] = exploitability_zero_sum(env.payoff_row, rp, cp)

    plot_rps(
        pay_row=pay_row,
        row_pol=row_pol_hist,
        col_pol=col_pol_hist,
        dist_nash=dist_nash,
        expl=expl,
        window=MA_WINDOW
    )

if __name__ == "__main__":
    run()