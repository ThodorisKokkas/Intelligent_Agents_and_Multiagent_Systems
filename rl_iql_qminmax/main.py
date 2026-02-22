from experiment import run_experiment
from plotting import plot_all

if __name__ == "__main__":
    res = run_experiment()
    print("Convergence time:", res["convergence_time_episode"])
    plot_all(res)
