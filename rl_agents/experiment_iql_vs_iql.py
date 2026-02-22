import numpy as np

from env_pd import RepeatedPD, C, D
from agent_iql import IndependentQLearner, IQLConfig
from metrics import OCC_KEYS, rolling_mean, rolling_var, detect_convergence

def run_iql_vs_iql(
    episodes: int = 50000,
    horizon: int = 200,
    window: int = 200,
    seed: int = 0,
    cfgA: IQLConfig | None = None,
    cfgB: IQLConfig | None = None,
    reset_random: bool = False,
    conv_delta: float = 0.05,
    conv_tau: float = 5.0,
    conv_K: int = 200,
) -> dict:
    rng = np.random.default_rng(seed)
    env = RepeatedPD(horizon=horizon, reset_random=reset_random, rng=rng)

    agentA = IndependentQLearner(cfgA or IQLConfig(), rng=np.random.default_rng(seed + 1))
    agentB = IndependentQLearner(cfgB or IQLConfig(), rng=np.random.default_rng(seed + 2))

    ep_return_A = np.zeros(episodes, dtype=np.float64)
    ep_return_B = np.zeros(episodes, dtype=np.float64)
    pC_A = np.zeros(episodes, dtype=np.float64)
    pC_B = np.zeros(episodes, dtype=np.float64)
    switch_A = np.zeros(episodes, dtype=np.float64)
    switch_B = np.zeros(episodes, dtype=np.float64)
    occ = np.zeros((episodes, 4), dtype=np.float64)  # CC, DD, CD, DC

    for e in range(episodes):
        if e % 1000 == 0 and e > 0:
            print(f"Episode {e}/{episodes}...")
        s = env.reset()
        epsA = agentA.epsilon(e)
        epsB = agentB.epsilon(e)

        cnt_C_A = cnt_C_B = 0
        cnt_switch_A = cnt_switch_B = 0
        prev_aA = prev_aB = None

        occ_counts = {k: 0 for k in OCC_KEYS}
        totalA = 0.0
        totalB = 0.0

        for t in range(horizon):
            aA = agentA.act(s, epsA)
            aB = agentB.act(s, epsB)

            cnt_C_A += int(aA == C)
            cnt_C_B += int(aB == C)

            if prev_aA is not None:
                cnt_switch_A += int(aA != prev_aA)
                cnt_switch_B += int(aB != prev_aB)
            prev_aA, prev_aB = aA, aB

            s_next, rA, rB, done = env.step(aA, aB)
            occ_counts[(aA, aB)] += 1

            agentA.update(s, aA, rA, s_next, done)
            agentB.update(s, aB, rB, s_next, done)

            totalA += rA
            totalB += rB
            s = s_next
            if done:
                break

        ep_return_A[e] = totalA
        ep_return_B[e] = totalB
        pC_A[e] = cnt_C_A / horizon
        pC_B[e] = cnt_C_B / horizon
        switch_A[e] = cnt_switch_A / max(1, horizon - 1)
        switch_B[e] = cnt_switch_B / max(1, horizon - 1)

        for i, k in enumerate(OCC_KEYS):
            occ[e, i] = occ_counts[k] / horizon

    # Rolling metrics
    roll_RA = rolling_mean(ep_return_A, window)
    roll_RB = rolling_mean(ep_return_B, window)
    roll_pC_A = rolling_mean(pC_A, window)
    roll_pC_B = rolling_mean(pC_B, window)
    roll_occ = np.stack([rolling_mean(occ[:, i], window) for i in range(4)], axis=1)

    roll_var_RA = rolling_var(ep_return_A, window)
    roll_var_RB = rolling_var(ep_return_B, window)

    conv_time, l1 = detect_convergence(
        rolling_occ=roll_occ,
        rolling_var_A=roll_var_RA,
        rolling_var_B=roll_var_RB,
        delta=conv_delta,
        tau=conv_tau,
        K=conv_K,
    )

    roll_diff = rolling_mean(ep_return_A - ep_return_B, window)

    return {
        "ep_return_A": ep_return_A,
        "ep_return_B": ep_return_B,
        "rolling_return_A": roll_RA,
        "rolling_return_B": roll_RB,
        "pC_A": pC_A,
        "pC_B": pC_B,
        "rolling_pC_A": roll_pC_A,
        "rolling_pC_B": roll_pC_B,
        "occ": occ,                 # CC,DD,CD,DC
        "rolling_occ": roll_occ,
        "switch_A": switch_A,
        "switch_B": switch_B,
        "rolling_diff": roll_diff,
        "l1_occ_change": l1,
        "rolling_var_RA": roll_var_RA,
        "rolling_var_RB": roll_var_RB,
        "convergence_time_episode": conv_time,
        "params": {
            "episodes": episodes,
            "horizon": horizon,
            "window": window,
            "seed": seed,
            "conv_delta": conv_delta,
            "conv_tau": conv_tau,
            "conv_K": conv_K,
        },
    }