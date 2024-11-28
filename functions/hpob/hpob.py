import numpy as np
import os
import xgboost as xgb
from functions.hpob.hpob_handler import HPOBHandler
from data.get_data import get_similar_data, get_combined_data

dim_dict = {
    '5860': 2,
    '5970': 2,
    '5859': 6,
    '5889': 6,
    '7607': 9,
    '7609': 9,
    '5906': 16,
    '5971': 16,
}

class HPOBProblem:
    def __init__(self, args = None):
        self.args = args
        self.search_space_id = self.args.search_space_id
        self.dataset_id = self.args.dataset_id
        self.dim = dim_dict[self.search_space_id]
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)
        self.surrogate = self.load_surrogate()
        self.data = self.load_data()
        self.DATA_MAXIMIZE = -1
    
    def load_surrogate(self):
        surrogate_path = os.path.join(os.getcwd(), 'functions', 'hpob', 'saved-surrogates')
        surrogate_name=f'surrogate-{self.search_space_id}-{self.dataset_id}'
        bst_surrogate = xgb.Booster()
        bst_surrogate.load_model(os.path.join(surrogate_path,f'{surrogate_name}.json'))
        return bst_surrogate
    
    def load_data(self):
        hpob_hdlr = HPOBHandler(root_dir=os.path.join(os.getcwd(), 'data', 'hpob-data'), mode='v3', surrogates_dir=os.path.join(os.getcwd(), 'functions', 'hpob', 'saved-surrogates'))
        return hpob_hdlr.meta_train_data
    
    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)

    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)

    def __call__(self, x):
        return self.surrogate.predict(xgb.DMatrix(x.reshape(-1, self.dim)))[0]