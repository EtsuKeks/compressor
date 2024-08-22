from utils import init_plots, get_plot, get_functions, get_L, get_act_val, get_data, get_divided_data
import numpy as np

def play(params: dict):
    init_plots(params)
    fp = params['function_params']
    fp['X'], fp['Y'], fp['X_diag_y'] = get_data(fp)
    fp['Xs'], fp['Ys'], fp['X_diag_ys'] = get_divided_data(fp)
    fp['w0'] = np.zeros(fp['X'].shape[1])
    fp['L'] = get_L(fp)
    fp['f'], fp['gradf'], fp['criterion'] = get_functions(fp)
    fp['act_val'] = get_act_val(fp)
    get_plot(params)