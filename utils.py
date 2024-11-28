import pickle

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print('='*10, f'Load model from {path}', '='*10)
    return model