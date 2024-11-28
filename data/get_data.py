from tqdm import tqdm
from data.utils import generate_datasets, generate_mix_datasets

def get_similar_data(search_space_id, dataset_id, data, is_maximize):
    data_X = None
    data_Y = None
    
    history = dict()
    source_id = []

    datasets = tqdm(data[search_space_id].keys())
    datasets.set_description('Load Source Datasets...')
    history[search_space_id] = dict()
    for did in datasets:
        if did == dataset_id:
            continue
        data_X, data_Y, history, source_id = generate_datasets(data, search_space_id, did, data_X, data_Y, history, source_id, is_maximize)

    data_Y = data_Y * is_maximize
    assert data_X.shape[0] == len(source_id)
    return data_X, data_Y, source_id, history

def get_combined_data(search_space_id, data, mode, is_maximize, dim):
    data_X = None
    data_Y = None

    history = dict()
    source_id = []
    if mode == 'hpob':
        dim_task={
            2: ['5860', '5970'],
            6: ['5859', '5889'],
            9: ['7607', '7609'],
            16: ['5906', '5971']
        }
        assert search_space_id in dim_task[dim]
        search_spaces = dim_task[dim][:]
    else:
        search_spaces = list(data.keys())

    for sid in search_spaces:
        history[sid] = dict()
        datasets = tqdm(data[sid].keys(), desc='Load Source Datasets...')
        for did in datasets:
            data_X, data_Y, history, source_id = generate_datasets(data, sid, did, data_X, data_Y, history, source_id, is_maximize)
    data_Y = data_Y * is_maximize
    assert data_X.shape[0] == len(source_id)
    return data_X, data_Y, source_id, history
    
def get_unsimilar_data(search_space_id, dataset_id, data, is_maximize):
    assert search_space_id == "Sphere2D"
    data_X = None
    data_Y = None
    history = dict()
    source_id = []
    datasets = tqdm(data[search_space_id].keys())
    datasets.set_description('Load Source Datasets...')

    history[search_space_id] = dict()
    for did in datasets:
        if did == dataset_id or did == '(5.0, 5.0)':
            continue
        data_X, data_Y, history, source_id = generate_datasets(data, search_space_id, did, data_X, data_Y, history, source_id, is_maximize)
    sample_num = data_Y.shape[0]
    data_Y = data_Y * is_maximize
    assert data_Y.shape[0] == 220
    assert data_X.shape[0] == len(source_id)
    return data_X, data_Y, source_id, history

def get_real_world_mixed_data(search_space_id, dataset_id, data, similar, is_maximize):
    data_X = None
    data_Y = None
    
    history = dict()
    source_id = []
    datasets = tqdm(data[search_space_id].keys())
    datasets.set_description('Load Source Datasets...')

    history[search_space_id] = dict()
    for did in datasets:
        if did == dataset_id:
            continue
        data_X, data_Y, history, source_id = generate_datasets(data, search_space_id, did, data_X, data_Y, history, source_id, is_maximize)

    sample_num = data_Y.shape[0]
    data_X, data_Y, history, source_id = generate_mix_datasets(search_space_id, similar, data_X, data_Y, history, source_id, is_maximize)
    data_Y = data_Y * is_maximize
    if similar == 'mix-similar':
        assert data_Y.shape[0] == sample_num + 7*300
    elif similar == 'mix-both':
        assert data_Y.shape[0] == sample_num + 14*300
    assert data_X.shape[0] == len(source_id)
    return data_X, data_Y, source_id, history


def get_design_bench_mixed_data(search_space_id, data, similar, is_maximize):
    data_X = None
    data_Y = None
    
    history = dict()
    source_id = []
    search_spaces = list(data.keys())
    for sid in search_spaces:
        history[sid] = dict()
        datasets = tqdm(data[sid].keys(), desc="Load Source Datasets...")
        for did in datasets:
            data_X, data_Y, history, source_id = generate_datasets(data, sid, did, data_X, data_Y, history, source_id, is_maximize)
    sample_num = data_Y.shape[0]
    data_X, data_Y, history, source_id = generate_mix_datasets(search_space_id, similar, data_X, data_Y, history, source_id, is_maximize)
    data_Y = data_Y * is_maximize

    if similar == 'mix-similar':
        assert data_Y.shape[0] == sample_num + 7*300
    elif similar == 'mix-both':
        assert data_Y.shape[0] == sample_num + 14*300
    assert data_X.shape[0] == len(source_id)
    return data_X, data_Y, source_id, history


# import numpy as np
# from mcts.utils import standardization
# import os
# from .utils import read_data_from_json

# def get_source_data(search_space_id, dataset_id, data, similar, dims, mode):
#     data_X = None
#     data_Y = None
    
#     history = dict()
#     source_id = []
    
#     is_maximize = 1 if mode in ["bbob", "real_world", "design_bench"] else -1  #bbob的数据是最大化的
    
#     def update_data(X, y, sid, did, data_X, data_Y, history, source_id):
#         assert X.shape[0] == y.shape[0]
#         source_id.extend([f"{sid}+{did}"]*X.shape[0])
#         if sid not in history:
#             history[sid] = dict()
#         history[sid][did] = dict()
        
#         if is_maximize == 1:
#             idx = np.argmax(y)
#         else:
#             idx = np.argmin(y)
            
#         history[sid][did]["X"] = X
#         history[sid][did]["y"] = y * is_maximize
#         # history[sid][did]["model"] = build_model(X, y, lb, ub)
#         history[sid][did]["X_optimal"] = X[idx, :]
        
        
#         data_X = X if data_X is None else np.vstack((data_X, X))
#         data_Y = y if data_Y is None else np.vstack((data_Y, y))
#         return data_X, data_Y, history, source_id
    
#     def generate_datasets(data, sid, did, data_X, data_Y, history, source_id):
#         X = np.array(data[sid][did]["X"])
#         y = np.array(data[sid][did]["y"]).reshape(-1,1)
#         y = standardization(y)
#         return update_data(X, y, sid, did, data_X, data_Y, history, source_id)
    
#     def generate_mix_datasets(sid, similar, data_X, data_Y, history, source_id):
#         assert similar in ["mix-similar", 'mix-unsimilar', 'mix-both']
#         mix_data_dir = f"data/generated_data/{sid}"
#         if similar in ["mix-similar", "mix-both"]:
#             dir = os.path.join(mix_data_dir,"similar/")
#             for root, dirs, files in os.walk(dir):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     X, y = read_data_from_json(file_path)
#                     y = standardization(y)
#                     data_X, data_Y, history, source_id = update_data(X, y, 'similar', os.path.splitext(file)[0], data_X, data_Y, history, source_id)
#         if similar in ["mix-unsimilar", "mix-both"]:
#             dir = os.path.join(mix_data_dir,"unsimilar/")
#             for root, dirs, files in os.walk(dir):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     X, y = read_data_from_json(file_path)
#                     y = standardization(y)
#                     data_X, data_Y, history, source_id = update_data(X, y, 'unsimilar', os.path.splitext(file)[0], data_X, data_Y, history, source_id)
#         return data_X, data_Y, history, source_id
    
#     if similar in ["unsimilar", "combine"] and mode != "Sphere2D":
#         if mode == "hpob":
#             dim_task={
#                 2: ["5860", "5970"],
#                 3: ["4796"],
#                 6: ["5859", "5889"],
#                 8: ["5891"],
#                 9: ["7607", "7609"],
#                 16: ["5906", "5971"]
#                 }
#             assert search_space_id in dim_task[dims]
#             search_spaces = dim_task[dims][:]
#         elif mode in ["bbob", "real_world", "design_bench"]:
#             search_spaces = list(data.keys())
#         if similar == "unsimilar":
#             search_spaces.remove(search_space_id)
        
#         for sid in search_spaces:
#             history[sid] = dict()
#             datasets = tqdm(data[sid].keys(), desc="Load Source Datasets...")
#             for did in datasets:
#                 data_X, data_Y, history, source_id = generate_datasets(data, sid, did, data_X, data_Y, history, source_id)
#         data_Y = data_Y * is_maximize
#     else:
#         if mode == 'design_bench':
#             search_spaces = list(data.keys())
#             for sid in search_spaces:
#                 history[sid] = dict()
#                 datasets = tqdm(data[sid].keys(), desc="Load Source Datasets...")
#                 for did in datasets:
#                     data_X, data_Y, history, source_id = generate_datasets(data, sid, did, data_X, data_Y, history, source_id)
#         else:
#             datasets = tqdm(data[search_space_id].keys())
#             datasets.set_description("Load Source Datasets...")

#             history[search_space_id] = dict()
#             for did in datasets:
#                 if did == dataset_id:
#                     continue
#                 if similar == "unsimilar" and mode == "Sphere2D":
#                     if did == "(5.0, 5.0)":
#                         continue
#                 data_X, data_Y, history, source_id = generate_datasets(data, search_space_id, did, data_X, data_Y, history, source_id)
        
#         sample_num = data_Y.shape[0]
#         if "mix" in similar:
#             data_X, data_Y, history, source_id = generate_mix_datasets(search_space_id, similar, data_X, data_Y, history, source_id)
#         data_Y = data_Y * is_maximize
#         if similar in ["mix-similar", "mix-unsimilar"]:
#             assert data_Y.shape[0] == sample_num + 7*300
#         elif similar == "mix-both":
#             assert data_Y.shape[0] == sample_num + 14*300
            
#         if mode == "Sphere2D":
#             if similar == "similar":
#                 assert data_Y.shape[0] == 330
#             else:
#                 assert data_Y.shape[0] == 220
        
#     assert data_X.shape[0] == len(source_id)
#     return data_X, data_Y, source_id, history


