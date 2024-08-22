import numpy as np

def f_L1_reg(w: np.ndarray, mu: np.float64) -> np.float64:
    return np.linalg.norm(w, 1) * mu

def f_L2_reg(w: np.ndarray, mu: np.float64) -> np.float64:
    return np.linalg.norm(w, 2) ** 2 * mu / 2

def f_elastic_reg(w: np.ndarray, mu: np.float64, alpha: float) -> np.float64:
    return f_L1_reg(w, mu) * alpha + f_L2_reg(w, mu) * (1 - alpha)

def gradf_L1_reg(w: np.ndarray, mu: np.float64) -> np.ndarray:
    return np.sign(w) * mu

def gradf_L2_reg(w: np.ndarray, mu: np.float64) -> np.ndarray:
    return mu * w

def gradf_elastic_reg(w: np.ndarray, mu: np.float64, alpha: float) -> np.ndarray:
    return gradf_L1_reg(w, mu) * alpha + gradf_L2_reg(w, mu) * (1 - alpha)