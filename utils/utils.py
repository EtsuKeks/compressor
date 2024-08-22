from F.regulizers import f_L1_reg, f_L2_reg, f_elastic_reg, gradf_L1_reg, gradf_L2_reg, gradf_elastic_reg
from F.funcs import f_log, f_linear, gradf_log, gradf_linear
from F.criterions import by_val, by_func, by_grad
from algorithms.algorithms import QAGD, QGD
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import re
import os

def init_plots(params: dict):
    actp = params['activation_params']
    if actp['iterable_1'] is not None:
        if actp['iterable_2'] is not None:
            tuple = (15 * len(actp['iterable_1']), 
                     15 * len(actp['iterable_2']))
        else:
            tuple = (15 * len(actp['iterable_1']), 15)
    else:
        tuple = (15, 15)

    params_for_plots = {'legend.fontsize': 'medium',
         'figure.figsize': tuple,
         'axes.labelsize': 'medium',
         'axes.titlesize':'medium',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
    
    plt.rcParams.update(params_for_plots)

def get_data(fp: dict):
    file_name = './datasets/' + fp['dataset'] + '.txt'
    if os.path.exists(file_name):
        data = load_svmlight_file('./datasets/' + fp['dataset'] + '.txt')
        X, Y = data[0].toarray(), data[1]
        Y = 2 * Y - 3
    else: 
        raise ValueError(f"No such file: {file_name}.")
    return X, Y, np.dot(X.T, np.diag(Y))

def get_divided_data(fp: dict):
    X, Y = fp['X'], fp['Y']
    points_per_device = int(X.shape[0] / fp['n'])

    Xs = []
    Ys = []
    X_diag_ys = []
    for j in range(fp['n']):
        X_deviced = []
        Y_deviced = []

        if (j == fp['n'] - 1):
            X_deviced = X[j * points_per_device :, :]
            Y_deviced = Y[j * points_per_device :]
        else:
            X_deviced = X[j * points_per_device : (j + 1) * points_per_device, :]
            Y_deviced = Y[j * points_per_device : (j + 1) * points_per_device]
        Xs.append(X_deviced)
        Ys.append(Y_deviced)
        X_diag_ys.append(np.dot(X_deviced.T, np.diag(Y_deviced)))

    return Xs, Ys, X_diag_ys

def get_L(fp: dict) -> np.float64:
    X = fp['X']
    if fp['task_type'] == 'log':
        L = np.abs(np.max(np.linalg.eig(1/X.shape[0] * np.dot(X.T, X))[0]))
        
        if fp['mu'] is None:
            raise ValueError(f"Invalid value for mu: {fp['mu']}.")
        
    elif fp['task_type'] == 'linear':
        L = np.abs(np.max(np.linalg.eig(2/X.shape[0] * np.dot(X.T, X))[0]))
    
        if fp['mu'] is None:
            fp['mu'] = np.min(np.linalg.eig(1/X.shape[0] * np.dot(X.T, X))[0]) / L
    
            if fp['mu'] < 0:
                raise ValueError(f"Invalid value for mu with task_type == linear: {fp['mu']}.")    
    
    if fp['regularization_type'] == 'L2':
        L += fp['mu'] * L
    
    elif fp['regularization_type'] == 'elastic':
        L += (1 - fp['alpha']) * fp['mu'] * L

    return L

def get_answer(fp: dict) -> np.float64:
    w = fp['w0'].copy()
    for i in range(fp['max_iter_answer']):
        if i % 1000 == 0:
            progress = int(i/fp['max_iter_answer']*100)+1
            progress_bar = f"Progress calculating act_val: [{progress * '#':<100}] {progress}%"
            print(progress_bar, end='\r')
        w = w - 1 / 10 / fp['L'] * fp['gradf'](-1, w)
    return w

def get_act_val(fp: dict) -> np.ndarray:
    file_name = './act_vals/' + fp['dataset'] + '_' + fp['task_type'] + '_' + f"mu={fp['mu']}" + '.txt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            content = file.read()

        nums = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|\d+', content)
        nums = [float(num) for num in nums]
        act_val = np.array(nums)
    else:
        print('-'*137 + f"\nFile {file_name} doesn't exist, calculating act_val...\n" + '-'*137)
        act_val = get_answer(fp)
        print("\nact_val calculation completed!")
        with open(file_name, 'w') as file:
            file.write(str(act_val))

    return act_val

def get_functions(fp: dict):
    mu = fp['mu'] * fp['L']
    typical_size = fp['X'].shape[1]
    if fp['regularization_type'] is None:
        f_reg = lambda w: 0
        gradf_reg = lambda w: np.zeros(typical_size)
    elif fp['regularization_type'] == 'L1':
        f_reg = lambda w: f_L1_reg(w, mu)
        gradf_reg = lambda w: gradf_L1_reg(w, mu)
    elif fp['regularization_type'] == 'L2':
        f_reg = lambda w: f_L2_reg(w, mu)
        gradf_reg = lambda w: gradf_L2_reg(w, mu)
    elif fp['regularization_type'] == 'elastic' and fp['alpha'] is not None:
        f_reg = lambda w: f_elastic_reg(w, mu, fp['alpha'])
        gradf_reg = lambda w: gradf_elastic_reg(w, mu, fp['alpha'])
    else: 
        raise ValueError(f"Invalid value for regularization_type param: {fp['regularization_type']}.")

    if fp['task_type'] == 'log':
        f = lambda i, w: f_log(i, fp, w) + f_reg(w)
        gradf = lambda i, w: gradf_log(i, fp, w) + gradf_reg(w)
    elif fp['task_type'] == 'linear':
        f = lambda i, w: f_linear(i, fp, w) + f_reg(w)
        gradf = lambda i, w: gradf_linear(i, fp, w) + gradf_reg(w)
    else: 
        raise ValueError(f"Invalid value for task_type param: {fp['task_type']}.")
    
    if fp['criterion'] == 'by_val':
        criterion = by_val
    elif fp['criterion'] == 'by_func':
        criterion = by_func
    elif fp['criterion'] == 'by_grad':
        criterion = by_grad
    else: 
        raise ValueError(f"Invalid value for criterion param: {fp['criterion']}.")

    return f, gradf, criterion

def get_plot(params: dict):
    actp = params['activation_params']
    actp_iterable_1 = actp['iterable_1']
    actp_iterable_2 = actp['iterable_2']
    actp_iterable_3 = actp['iterable_3']

    algorithm_name = params['algorithm_params']['name']
    if algorithm_name == 'QGD':
        algo = QGD
    elif algorithm_name == 'QAGD':
        algo = QAGD
    else: 
        raise ValueError(f"Invalid value: {algorithm_name}.")
    
    if actp_iterable_3 is None:
        if actp_iterable_2 is None:
            if actp_iterable_1 is None:
                get_plot_0_axis(algo, params)
            else:
                get_plot_1_axis(actp_iterable_1, algo, params)
        else:
            if actp_iterable_1 is None: raise ValueError(f"iterable_1 is None while iterable_2 is not None.")
            get_plot_2_axis(actp_iterable_1, actp_iterable_2, algo, params)
    else:
        if actp_iterable_1 is None or actp_iterable_2 is None: raise ValueError(f"iterable_1 or iterable_2 is None while iterable_3 is not None.")
        get_plot_3_axis(actp_iterable_1, actp_iterable_2, actp_iterable_3, algo, params)

    plt.savefig('./results/' + datetime.now().strftime("%d.%m.%Y__%H:%M:%S") + '.png', bbox_inches='tight')
        
def get_plot_0_axis(algo, params: dict):
    x_axis_name = params['algorithm_params']['x_axis_name']
    y_axis_name = params['algorithm_params']['y_axis_name']
    Compressors = params['activation_params']['compressors']
    compressor_names = params['activation_params']['compressor_names']

    fig, axs = plt.subplots()
    for k in range(len(Compressors)):
        compressor = Compressors[k]()
        x, y = algo(params, compressor)
        axs.plot(x, np.log(y), label=compressor_names[k] + '.')
        axs.set_xlabel(x_axis_name)
        axs.set_ylabel(y_axis_name)
        axs.legend(loc="upper right")

        progress = int((k+1)/len(Compressors) * 100)
        progress_bar = f"Progress calculating plots: [{progress * '#':<100}] {progress}%"
        print(progress_bar, end='\r')

def get_plot_1_axis(actp_iterable_1: list, algo, params: dict):
    actp_iterable_1_name = params['activation_params']['iterable_1_name']
    x_axis_name = params['algorithm_params']['x_axis_name']
    y_axis_name = params['algorithm_params']['y_axis_name']
    Compressor = params['activation_params']['compressors'][0]
    compressor_name = params['activation_params']['compressor_names'][0]

    fig, axs = plt.subplots(len(actp_iterable_1))
    for i in range(len(actp_iterable_1)):
            compressor = Compressor(actp_iterable_1[i])
            x, y = algo(params, compressor)
            axs[i].plot(x, np.log(y), label=compressor_name + '. ' + \
                        actp_iterable_1_name + ' = ' + str(round(actp_iterable_1[i], 3)))
            axs[i].set_xlabel(x_axis_name)
            axs[i].set_ylabel(y_axis_name)
            axs[i].legend(loc="upper right")

            progress = int((i+1)/len(actp_iterable_1)*100)
            progress_bar = f"Progress calculating plots: [{progress * '#':<100}] {progress}%"
            print(progress_bar, end='\r')

def get_plot_2_axis(actp_iterable_1: list, actp_iterable_2: list, algo, params: dict):
    actp_iterable_1_name = params['activation_params']['iterable_1_name']
    actp_iterable_2_name = params['activation_params']['iterable_2_name']
    x_axis_name = params['algorithm_params']['x_axis_name']
    y_axis_name = params['algorithm_params']['y_axis_name']
    Compressor = params['activation_params']['compressors'][0]
    compressor_name = params['activation_params']['compressor_names'][0]

    fig, axs = plt.subplots(len(actp_iterable_1), len(actp_iterable_2))
    for i in range(len(actp_iterable_1)):
        for j in range(len(actp_iterable_2)):
                compressor = Compressor(actp_iterable_1[i], actp_iterable_2[j])
                x, y = algo(params, compressor)
                axs[i, j].plot(x, np.log(y), label=compressor_name + '. ' + \
                            actp_iterable_1_name + '=' + str(round(actp_iterable_1[i], 3)) + \
                            ', ' + actp_iterable_2_name + '=' + str(round(actp_iterable_2[j], 3)))
                axs[i, j].set_xlabel(x_axis_name)
                axs[i, j].set_ylabel(y_axis_name)
                axs[i, j].legend(loc="upper right")

                progress = int((i*len(actp_iterable_1)+j+1)/len(actp_iterable_2)/len(actp_iterable_1)*100)
                progress_bar = f"Progress calculating plots: [{progress * '#':<100}] {progress}%"
                print(progress_bar, end='\r')
    
def get_plot_3_axis(actp_iterable_1: list, actp_iterable_2: list, actp_iterable_3: list, algo, params: dict):
    actp_iterable_1_name = params['activation_params']['iterable_1_name']
    actp_iterable_2_name = params['activation_params']['iterable_2_name']
    actp_iterable_3_name = params['activation_params']['iterable_3_name']
    x_axis_name = params['algorithm_params']['x_axis_name']
    y_axis_name = params['algorithm_params']['y_axis_name']
    Compressor = params['activation_params']['compressors'][0]
    compressor_name = params['activation_params']['compressor_names'][0]

    fig, axs = plt.subplots(len(actp_iterable_1), len(actp_iterable_2))
    for i in range(len(actp_iterable_1)):
        for j in range(len(actp_iterable_2)):
            for k in range(len(actp_iterable_3)):
                compressor = Compressor(actp_iterable_1[i], actp_iterable_2[j], actp_iterable_3[k])
                x, y = algo(params, compressor)
                axs[i, j].plot(x, np.log(y), label=compressor_name + '. ' + \
                               actp_iterable_1_name + '=' + str(round(actp_iterable_1[i], 3)) + \
                            ', ' + actp_iterable_2_name + '=' + str(round(actp_iterable_2[j], 3)) + \
                            ', ' + actp_iterable_3_name + '=' + str(round(actp_iterable_3[k], 3)))
                axs[i, j].set_xlabel(x_axis_name)
                axs[i, j].set_ylabel(y_axis_name)
                axs[i, j].legend(loc="upper right")

                progress = int((i*len(actp_iterable_1)*len(actp_iterable_2)+j*len(actp_iterable_2)+k+1)/\
                               len(actp_iterable_2)/len(actp_iterable_1)/len(actp_iterable_3)*100)
                progress_bar = f"Progress calculating plots: [{progress * '#':<100}] {progress}%"
                print(progress_bar, end='\r')