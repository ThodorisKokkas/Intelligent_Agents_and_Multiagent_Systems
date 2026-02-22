import numpy as np
from env_pd import RepeatedPD, C
from agent_iql import IndependentQLearner, IQLConfig
from agent_qminimax import QMinimaxLearner, QMinimaxConfig
from metrics import OCC_KEYS, rolling_mean, rolling_var, detect_convergence

def run_experiment(episodes=50000, horizon=200, window=200, seed=0):
    rng = np.random.default_rng(seed)
    env = RepeatedPD(horizon=horizon, rng=rng)

    agentA = IndependentQLearner(IQLConfig(), np.random.default_rng(seed+1))
    agentB = QMinimaxLearner(QMinimaxConfig(), np.random.default_rng(seed+2))

    ep_return_A = np.zeros(episodes)
    ep_return_B = np.zeros(episodes)
    occ = np.zeros((episodes,4))

    for e in range(episodes):
        s = env.reset()
        epsA = agentA.epsilon(e)
        epsB = agentB.epsilon(e)
        totalA = totalB = 0

        occ_counts = {k:0 for k in OCC_KEYS}

        for t in range(horizon):
            aA = agentA.act(s, epsA)
            aB = agentB.act(s, epsB)

            s_next, rA, rB, done = env.step(aA,aB)

            # zero-sum shaping
            rA_zs = rA - rB
            rB_zs = -rA_zs

            agentA.update(s,aA,rA_zs,s_next,done)
            agentB.update(s,a_self=aB,a_opp=aA,r=rB_zs,s_next=s_next,done=done)

            totalA += rA_zs
            totalB += rB_zs
            occ_counts[(aA,aB)] += 1

            s = s_next
            if done:
                break

        ep_return_A[e] = totalA
        ep_return_B[e] = totalB

        for i,k in enumerate(OCC_KEYS):
            occ[e,i] = occ_counts[k]/horizon

    roll_occ = np.stack([rolling_mean(occ[:,i],window) for i in range(4)],axis=1)
    roll_var_A = rolling_var(ep_return_A,window)
    roll_var_B = rolling_var(ep_return_B,window)
    conv_time, l1 = detect_convergence(roll_occ, roll_var_A, roll_var_B)

    return {
        "rolling_occ": roll_occ,
        "ep_return_A": ep_return_A,
        "ep_return_B": ep_return_B,
        "rolling_var_RA": roll_var_A,
        "rolling_var_RB": roll_var_B,
        "l1_occ_change": l1,
        "convergence_time_episode": conv_time,
        "params":{"window":window}
    }