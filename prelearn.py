import pickle
import numpy as np
from mcts import MCTS
import os
from mcts.utils import minmax

def mcts_prelearn(args, problem, model_saving_dir):
    '''
    MCTS-transfer prelearning stage

    Args: 
        problem
        model_saving_dir

    Returns:
        prelearned_model_path
    '''
    search_space_id = args.search_space_id
    dataset_id = args.dataset_id
    similar = args.similar
    mode = args.mode

    prelearned_model_path = os.path.join(model_saving_dir, f'mcts{search_space_id}{similar}.pkl')       
    if os.path.exists(prelearned_model_path):
        return prelearned_model_path
    
    dim = problem.dim
    lb = problem.lb
    ub = problem.ub

    if similar == 'similar' or similar == 'mix-similar':
        data_X, data_Y, source_id, history = problem.load_similar_source_data()
    elif similar == 'combine' or similar == 'mix-both':
        data_X, data_Y, source_id, history = problem.load_mixed_source_data()
    elif similar == 'unsimilar':
        assert mode == "Sphere2D"
        data_X, data_Y, source_id, history = problem.load_unsimilar_source_data()
    
    assert dim == data_X.shape[1]

    data_X = minmax(data_X, min=lb, max=ub)

    agent = MCTS(
        lb = lb,              # the lower bound of each problem dimensions
        ub = ub,              # the upper bound of each problem dimensions
        dims = dim,          # the problem dimensions
        ninits = 0,      # the number of random samples used in initializations 
        func = None,               # function object to be optimized
        Cp = 1,              # Cp for MCTS
        leaf_size = min(max(10, np.round(data_X.shape[0]/1500)),30), # tree leaf size (max points in leaf nodes)
        kernel_type = args.kernel_type, # SVM configruation
        gamma_type = 'auto',    # SVM configruation
        state = 'learn-only',
        source_X = data_X,
        source_Y = data_Y,
        source_id = source_id,
        stage = 0,
        search_space_id = search_space_id,
        dataset_id = dataset_id,
        similar = similar,
        )
    
    agent.learn(data_X, data_Y)
    agent.history_sample = history
    
    with open(prelearned_model_path, 'wb') as f:
        print('='*10, f'training model {prelearned_model_path}', '='*10)
        pickle.dump(agent, f)
    return prelearned_model_path
    