import json
import numpy as np
import os
from mcts.utils import standardization

def read_data_from_json(path):
    with open(path, 'r') as f:
        json_data = json.load(f)
    X = np.array(json_data['X'])
    y = np.array(json_data['y']).reshape(-1,1)
    return X, y
        

def update_data(X, y, sid, did, data_X, data_Y, history, source_id, is_maximize):
    assert X.shape[0] == y.shape[0]
    source_id.extend([f'{sid}+{did}']*X.shape[0])
    if sid not in history:
        history[sid] = dict()
    history[sid][did] = dict()
    
    if is_maximize == 1:
        idx = np.argmax(y)
    else:
        idx = np.argmin(y)
        
    history[sid][did]['X'] = X
    history[sid][did]['y'] = y * is_maximize
    # history[sid][did]['model'] = build_model(X, y, lb, ub)
    history[sid][did]['X_optimal'] = X[idx, :]
    
    
    data_X = X if data_X is None else np.vstack((data_X, X))
    data_Y = y if data_Y is None else np.vstack((data_Y, y))
    return data_X, data_Y, history, source_id

def generate_datasets(data, sid, did, data_X, data_Y, history, source_id, is_maximize):
    X = np.array(data[sid][did]['X'])
    y = np.array(data[sid][did]['y']).reshape(-1,1)
    y = standardization(y)
    return update_data(X, y, sid, did, data_X, data_Y, history, source_id, is_maximize)

def generate_mix_datasets(sid, similar, data_X, data_Y, history, source_id, is_maximize):
    assert similar in ['mix-similar', 'mix-both']
    mix_data_dir = os.path.join(os.getcwd(), 'data', 'generated_data',sid)

    dir = os.path.join(mix_data_dir,'similar')
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            X, y = read_data_from_json(file_path)
            y = standardization(y)
            data_X, data_Y, history, source_id = update_data(X, y, 'similar', os.path.splitext(file)[0], data_X, data_Y, history, source_id, is_maximize)
    if similar == 'mix-both':
        dir = os.path.join(mix_data_dir,'unsimilar')
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                X, y = read_data_from_json(file_path)
                y = standardization(y)
                data_X, data_Y, history, source_id = update_data(X, y, 'unsimilar', os.path.splitext(file)[0], data_X, data_Y, history, source_id, is_maximize)
    return data_X, data_Y, history, source_id