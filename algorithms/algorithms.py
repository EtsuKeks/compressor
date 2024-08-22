from algorithms.compressors import Compressor
import numpy as np

def QGD(params: dict, compressor: Compressor):
    algop = params['algorithm_params']
    fp = params['function_params']
    
    grads = np.zeros((fp['w0'].shape[0], fp['n']))
    masks = grads.copy()
    
    w = fp['w0'].copy()
    ws = [w]
    ctx = { 'w': w, 'k': 0, 'ws': ws, 'i': -1}
    y = [fp['criterion'](fp, ctx)]
    avg_density = 0
    x = [avg_density]

    for k in range(algop['max_iter']):
        for i in range(fp['n']):
            ctx = { 'w': w, 'k': k, 'ws': ws, 'i': i }
            grads[:, i] = fp['gradf'](i, w)
            masks[:, i] = compressor.zip(params, ctx, grads[:, i])
            avg_density += np.sum(masks[:, i] != 0) / masks.shape[0] / fp['n']
            grads[:, i] = np.multiply(grads[:, i], masks[:, i])

        grad = 1 / fp['n'] * np.sum(grads, axis=1)

        w = w - 1 / fp['L'] * grad
        ws.append(w)
        ctx = { 'w': w, 'k': k+1, 'ws': ws, 'i': -1 }
        y.append(fp['criterion'](fp, ctx))
        x.append(avg_density)

    return x, y

def QAGD(params: dict, compressor: Compressor):
    algop = params['algorithm_params']
    fp = params['function_params']
    
    gamma = np.sqrt(fp['L'] / fp['mu'])

    grads = np.zeros((fp['w0'].shape[0], fp['n']))
    masks = grads.copy()
    
    w_new = fp['w0'].copy()
    w_old = fp['w0'].copy()
    ws_new = [w_new]
    ws_old = [w_old]
    ctx = { 'w': w_new, 'w_old': w_old, 'k': 0, 'ws': ws_new, 'ws_old': ws_old, 'i': -1 }
    y = [fp['criterion'](fp, ctx)]
    avg_density = 0
    x = [avg_density]

    for k in range(algop['max_iter']):
        w_old = w_new + gamma * (w_new - w_old)
        
        for i in range(fp['n']):
            ctx = { 'w': w_old, 'w_old': w_new, 'k': k, 'ws': ws_new, 'ws_old': ws_old, 'i': i }
            grads[:, i] = fp['gradf'](i, w_old)
            masks[:, i] = compressor.zip(params, ctx, grads[:, i])
            avg_density += np.sum(masks[:, i]) / masks.shape[0] / fp['n']
            grads[:, i] = np.multiply(grads[:, i], masks[:, i])

        grad = 1 / fp['n'] * np.sum(grads, axis=1)

        w_new = w_old - 1 / fp['L'] * grad

        ws_new.append(w_new)
        ws_old.append(w_old)
        ctx = { 'w': w_new, 'w_old': w_old, 'k': k+1, 'ws': ws_new, 'ws_old': ws_old, 'i': -1 }
        y.append(fp['criterion'](fp, ctx))
        x.append(avg_density)

    return x, y