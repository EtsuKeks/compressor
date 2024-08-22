import numpy as np

def f_log(i: int, fp:dict, w: np.ndarray) -> np.float64:
    if i < 0:
        X, Y = fp['X'], fp['Y']
    else:
        X, Y = fp['Xs'][i], fp['Ys'][i]

    ywx = -np.multiply(np.dot(X, w), Y)
    to_be_summed = np.log(1 + np.exp(ywx))
    res = 1 / X.shape[0] * np.sum(to_be_summed)
    return res

def f_linear(i: int, fp: dict, w: np.ndarray) -> np.float64:
    if i < 0:
        X, Y = fp['X'], fp['Y']
    else:
        X, Y = fp['Xs'][i], fp['Ys'][i]

    return 1 / X.shape[0] * np.linalg.norm((np.dot(X, w) - Y), 2) ** 2

def gradf_log(i: int, fp: dict, w: np.ndarray) -> np.ndarray:
    if i < 0:
        X, Y, X_diag_y = fp['X'], fp['Y'], fp['X_diag_y']
    else:
        X, Y, X_diag_y = fp['Xs'][i], fp['Ys'][i], fp['X_diag_ys'][i]

    ywx = -np.multiply(np.dot(X, w), Y)
    to_be_multiplied = np.exp(ywx) / (1 + np.exp(ywx))
    res = 1 / X.shape[0] * np.dot(-X_diag_y, to_be_multiplied)
    return res

def gradf_linear(i: int, fp: dict, w: np.ndarray) -> np.ndarray:
    if i < 0:
        X, Y = fp['X'], fp['Y']
    else:
        X, Y = fp['Xs'][i], fp['Ys'][i]
        
    res = 2 / X.shape[0] * (np.dot(X.T, np.dot(X, w)) - np.dot(X.T, Y))
    return res