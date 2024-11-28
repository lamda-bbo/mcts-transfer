import numpy as np
from functions.utils import get_data
from data.get_data import get_similar_data, get_unsimilar_data

class Sphere2DProblem:
    def __init__(self, args):
        self.args = args
        self.dim = 2
        self.lb = np.full(self.dim, -10)
        self.ub = np.full(self.dim, 10)
        self.search_space_id = args.search_space_id
        self.center_x, self.center_y = eval(args.dataset_id)
        self.data = self.load_data()
        self.DATA_MAXIMIZE = -1

    def load_data(self):
        return get_data(self.args.mode)
    
    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_unsimilar_source_data(self):
        return get_unsimilar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
        
    def __call__(self, x: np.ndarray):
        return float(np.sum((x - np.array([self.center_x, self.center_y]))**2))

        
