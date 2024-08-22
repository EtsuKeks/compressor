import numpy as np

def by_func(fp: dict, ctx: dict) -> np.float64:
    return abs(fp['f'](-1, ctx['w']) - fp['f'](-1, fp['act_val']) ) / \
        abs( fp['f'](-1, fp['w0']) - fp['f'](-1, fp['act_val']))

def by_val(fp: dict, ctx: dict) -> np.float64:
    return np.linalg.norm( ctx['w'] - fp['act_val'], 2 ) / \
        np.linalg.norm( fp['w0'] - fp['act_val'], 2 )

def by_grad(fp: dict, ctx: dict) -> np.float64:
    return np.linalg.norm( fp['gradf'](-1, ctx['w']), 2 ) / \
        np.linalg.norm( fp['gradf'](-1, fp['w0']), 2 )