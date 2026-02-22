import numpy as np
import matplotlib.pyplot as plt
from metrics import OCC_LABELS, rolling_mean

def plot_all(res: dict):
    ep = np.arange(len(res["ep_return_A"]))
    window = int(res["params"]["window"])

    # 1) Rolling occupancy
    plt.figure()
    for i, label in enumerate(OCC_LABELS):
        plt.plot(ep, res["rolling_occ"][:, i], label=label)
    plt.xlabel("Episode")
    plt.ylabel("Occupancy (rolling)")
    plt.title("Rolling joint-action occupancy")
    plt.legend()
    plt.grid(True)

    # 2) Reward variance (convergence signal)
    plt.figure()
    plt.plot(ep, res["rolling_var_RA"], label="Var(Return A)")
    plt.plot(ep, res["rolling_var_RB"], label="Var(Return B)")
    plt.xlabel("Episode")
    plt.ylabel("Variance")
    plt.title("Rolling reward variance")
    plt.legend()
    plt.grid(True)

    # 3) L1 occupancy change
    plt.figure()
    plt.plot(ep, res["l1_occ_change"], label="L1 change")
    plt.xlabel("Episode")
    plt.ylabel("L1 change")
    plt.title("Convergence signal: occupancy change")
    plt.legend()
    plt.grid(True)

    # 4) Episode returns (shaped zero-sum)
    plt.figure()
    plt.plot(ep, rolling_mean(res["ep_return_A"], window), label="A rolling return")
    plt.plot(ep, rolling_mean(res["ep_return_B"], window), label="B rolling return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Rolling episode return (zero-sum shaped)")
    plt.legend()
    plt.grid(True)

    # Mark convergence time if exists
    ct = res.get("convergence_time_episode", None)
    if ct is not None:
        for fig_num in plt.get_fignums():
            plt.figure(fig_num)
            plt.axvline(ct, linestyle="--", label="convergence time")
            plt.legend()

    plt.show()