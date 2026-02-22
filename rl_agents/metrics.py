import numpy as np

# occupancy order: CC, DD, CD, DC (όπως ζήτησες)
C, D = 0, 1
OCC_KEYS = [(C, C), (D, D), (C, D), (D, C)]
OCC_LABELS = ["(C,C)", "(D,D)", "(C,D)", "(D,C)"]

def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    x = x.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    c = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out[i] = (c[i + 1] - c[start]) / (i - start + 1)
    return out

def rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64, copy=False)
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        seg = x[start:i + 1]
        out[i] = float(np.var(seg))
    return out

def detect_convergence(rolling_occ: np.ndarray,
                       rolling_var_A: np.ndarray,
                       rolling_var_B: np.ndarray,
                       delta: float = 0.01,
                       tau: float = 5.0,
                       K: int = 200):
    """
    Convergence heuristic:
      stable if:
        - L1 change of rolling occupancy < delta
        - rolling reward variance for both agents < tau
      convergence_time = first episode where stable holds for K consecutive
    """
    episodes = rolling_occ.shape[0]
    l1 = np.zeros(episodes, dtype=np.float64)
    if episodes > 1:
        l1[1:] = np.sum(np.abs(rolling_occ[1:] - rolling_occ[:-1]), axis=1)

    stable = (l1 < delta) & (rolling_var_A < tau) & (rolling_var_B < tau)

    conv_time = None
    if episodes >= K and np.any(stable):
        for i in range(episodes - K):
            if np.all(stable[i:i + K]):
                conv_time = int(i)
                break

    return conv_time, l1