import numpy as np
import matplotlib.pyplot as plt
from metrics import OCC_LABELS, rolling_mean

def plot_all(res: dict):
    ep = np.arange(len(res["ep_return_A"]))
    window = int(res["params"]["window"])

    # 1) Rolling mean rewards
    plt.figure()
    plt.plot(ep, res["rolling_return_A"], label="A rolling return")
    plt.plot(ep, res["rolling_return_B"], label="B rolling return")
    plt.xlabel("Episode")
    plt.ylabel("Return (sum rewards per episode)")
    plt.title("Rolling mean episode return")
    plt.legend()
    plt.grid(True)

    # 2) Rolling P(C)
    plt.figure()
    plt.plot(ep, res["rolling_pC_A"], label="P(C) A (rolling)")
    plt.plot(ep, res["rolling_pC_B"], label="P(C) B (rolling)")
    plt.xlabel("Episode")
    plt.ylabel("P(C)")
    plt.title("Rolling action frequency")
    plt.legend()
    plt.grid(True)

    # 3) Rolling occupancy
    plt.figure()
    for i, label in enumerate(OCC_LABELS):
        plt.plot(ep, res["rolling_occ"][:, i], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Occupancy (rolling)")
    plt.title("Rolling joint-action occupancy")
    plt.legend()
    plt.grid(True)

    # 4) Stability: switch rate (rolling)
    plt.figure()
    plt.plot(ep, rolling_mean(res["switch_A"], window), label="Switch rate A (rolling)")
    plt.plot(ep, rolling_mean(res["switch_B"], window), label="Switch rate B (rolling)")
    plt.xlabel("Episode")
    plt.ylabel("Switch rate")
    plt.title("Stability (action switch-rate)")
    plt.legend()
    plt.grid(True)

    # 5) Convergence signals
    plt.figure()
    plt.plot(ep, res["l1_occ_change"], label="L1 change (rolling occupancy)")
    plt.xlabel("Episode")
    plt.ylabel("L1 change")
    plt.title("Convergence signal: occupancy change")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(ep, res["rolling_var_RA"], label="Var(Return A) in window")
    plt.plot(ep, res["rolling_var_RB"], label="Var(Return B) in window")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.title("Convergence signal: rolling reward variance")
    plt.legend()
    plt.grid(True)

    # 6) Exploitability proxy: rolling reward difference
    plt.figure()
    plt.plot(ep, res["rolling_diff"], label="(A - B) rolling return diff")
    plt.xlabel("Episode")
    plt.ylabel("Return difference")
    plt.title("Exploitability proxy (A - B)")
    plt.legend()
    plt.grid(True)

    # Mark convergence time
    ct = res.get("convergence_time_episode", None)
    if ct is not None:
        for fig_num in plt.get_fignums():
            plt.figure(fig_num)
            plt.axvline(ct, linestyle="--", label="convergence time")
            plt.legend()

    plt.show()