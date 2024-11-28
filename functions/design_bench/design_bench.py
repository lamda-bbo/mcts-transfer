import numpy as np
from functions.utils import get_data
from data.get_data import get_design_bench_mixed_data

class Superconductor:
    def __init__(self, args = None):
        import design_bench
        from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
        self.dataset = SuperconductorDataset()
        x = self.dataset.x
        self.lb = np.array(x.min(axis=0))
        self.ub = np.array(x.max(axis=0))
        self.dim = 86
        self.task = design_bench.make('Superconductor-RandomForest-v0')
        self.DATA_MAXIMIZE = 1
        self.args = args
        self.data = get_data(self.args.search_space_id)

    def load_similar_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)

    def __call__(self, x) -> float: 
        x = x.reshape((-1, self.dim))
        return -1 * self.task.predict(x)
    
class AntMorphology:
    def __init__(self, args = None):
        import design_bench
        from design_bench.datasets.continuous.superconductor_dataset import AntMorphologyDataset
        self.dataset = AntMorphologyDataset()
        x = self.dataset.x
        self.lb = np.array(x.min(axis=0))
        self.ub = np.array(x.max(axis=0))
        self.dim = 60
        self.task = design_bench.make('AntMorphology-Exact-v0')
        self.DATA_MAXIMIZE = 1
        self.args = args
        self.data = get_data(self.args.search_space_id)

    def load_similar_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def load_similar_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def __call__(self, x) -> float: 
        x = x.reshape((-1, self.dim))
        return -1 * self.task.predict(x)

class DKittyMorphology:
    def __init__(self, args = None):
        import design_bench
        from design_bench.datasets.continuous.superconductor_dataset import DKittyMorphologyDataset
        self.dataset = DKittyMorphologyDataset()
        x = self.dataset.x
        self.lb = np.array(x.min(axis=0))
        self.ub = np.array(x.max(axis=0))
        self.dim = 56
        self.task = design_bench.make('DKittyMorphology-Exact-v0')
        self.DATA_MAXIMIZE = 1
        self.args = args
        self.data = get_data(self.args.search_space_id)

    def load_similar_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def load_similar_source_data(self):
        return get_design_bench_mixed_data(self.args.search_space_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def __call__(self, x) -> float: 
        x = x.reshape((-1, self.dim))
        return -1 * self.task.predict(x)