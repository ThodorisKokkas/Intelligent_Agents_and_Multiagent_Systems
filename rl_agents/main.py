from experiment_iql_vs_iql import run_iql_vs_iql
from plotting import plot_all

if __name__ == "__main__":
    res = run_iql_vs_iql(
        episodes=50000,
        horizon=200,
        window=200,
        seed=0,
        reset_random=False,
    )

    print("Convergence time (episode):", res["convergence_time_episode"])
    print("Final rolling P(C) A,B:", res["rolling_pC_A"][-1], res["rolling_pC_B"][-1])
    print("Final rolling occupancy CC,DD,CD,DC:", res["rolling_occ"][-1])

    plot_all(res)