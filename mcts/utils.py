# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def from_unit_cube(point, lb, ub):
    assert np.all(lb <= ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    new_point = point * (ub - lb) + lb
    return new_point

def latin_hypercube(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points

def standardization(y, mean=None, std=None):
    if mean is not None and std is not None:
        return (y - mean) / std
    
    if y.ndim == 1: 
        axis = 0
    elif y.ndim == 2:  
        axis = 0  
    else:
        raise ValueError("Input must be a 1D or 2D array.")
    
    mean = np.mean(y, axis=axis)
    std = np.std(y, axis=axis)
    
    std = np.where(std == 0, np.finfo(float).eps, std)
    
    if y.ndim == 2:
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
    
    return (y - mean) / std

def minmax(X, min=None, max=None):
    if min is not None and max is not None:
        res = (X - min) / (max - min)
        res = np.nan_to_num(res)
        return res
    
    min = np.min(X, axis=0)  
    max = np.max(X, axis=0)  
    return (X - min) / (max - min)

def get_data_in_node(node, name):
    assert len(node.bag_source) == len(node.source_id)
    if name in node.source_id:
        id_to_data = zip(node.source_id, node.bag_source)
        samples = [sample for id, sample in id_to_data if id==name]
        cands = [cand for cand, _ in samples]
        cands_Y = [value for _, value in samples]
        return True, cands, cands_Y
    else:
        return False, None, None

def build_model(X, y, lb, ub):
    noise = 0.1
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2, normalize_y=False)
    X = minmax(X, min=lb, max=ub)
    y = standardization(y)
    gpr.fit(X, y)
    return gpr

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1.flatten() - x2.flatten()))

def get_N_best(X, y, num_best):
    indices = np.argsort(-y, axis=0)[:num_best].flatten() 
    selected_X = X[indices]
    return selected_X

def get_num_best(X, N):
    return max(int(len(X) * N if N < 1 else N), 1)
     
def get_N_best_mean(history_X, history_y, target_X, target_y, N):
    num_best_history = get_num_best(history_X, N)
    num_best_target = get_num_best(target_X, N)
    history_x = np.mean(get_N_best(history_X, history_y, num_best_history), axis=0)
    target_x = np.mean(get_N_best(target_X, target_y, num_best_target), axis=0)
    return history_x, target_x
    
def manhattan_distance_N(history_X, history_y, target_X, target_y, N=1.0):
    x1, x2 = get_N_best_mean(history_X, history_y, target_X, target_y, N) 
    return manhattan_distance(x1, x2)
    
def Kendall_coefficient(history_model, target_X, target_y, lb, ub):
    mu_list = list()
    for idx in range(target_X.shape[0]):
        x = minmax(target_X[idx,:], lb, ub).reshape(1,-1)
        mu_list.append(history_model.predict(x)[0])
        
    target_y = np.asarray(target_y,  dtype=np.float32).reshape(-1)
    target_y = standardization(target_y)
    
    rank_loss = 0
    for i in range(len(target_y)):
        for j in range(len(target_y)):
            if (target_y[i] < target_y[j]) ^ (mu_list[i] < mu_list[j]):
                rank_loss += 1
    return rank_loss

def kl_divergence(kde_p, kde_q, samples):
    p_samples = kde_p(samples)
    q_samples = kde_q(samples)
    return np.mean(np.log(p_samples / q_samples))

def KL_distance(history_X, history_y, target_X, target_y, lb, ub):
    if target_X.shape[0] == 1:
        return 0
    
    history_X, target_X = minmax(history_X, lb, ub), minmax(target_X, lb, ub)
    history_y, target_y = standardization(history_y), standardization(target_y)
    
    history_data = np.hstack((history_X, history_y.reshape(-1, 1)))
    target_data = np.hstack((target_X, target_y.reshape(-1, 1)))
    
    history_data = np.unique(history_data, axis=0)
    target_data = np.unique(target_data, axis=0)
    
    
    pca = PCA(n_components=0.95)
    target_data_pca = pca.fit_transform(target_data)
    history_data_pca = pca.transform(history_data)
    
    if target_data_pca.shape[1] > 3:
        pca = PCA(n_components=3)
        target_data_pca = pca.fit_transform(target_data)
        history_data_pca = pca.transform(history_data)
    
    assert target_data_pca.shape[1] <= 3
    assert history_data_pca.shape[1] <= 3
    
    kde_history = gaussian_kde(history_data_pca.T)
    kde_target = gaussian_kde(target_data_pca.T)
    
    # samples = history_data.T
    if history_data_pca.shape[0] <= 500:
        samples = history_data_pca.T
    else:
        samples = history_data_pca[np.random.choice(history_data_pca.shape[0], 500, replace=False)].T
    kl_div = kl_divergence(kde_history, kde_target, samples)
    return kl_div

    