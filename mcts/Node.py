# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
from .Classifier import Classifier
import json
import numpy as np
import math
import operator

class Node:
    obj_counter   = 0
    # If a leave holds >= SPLIT_THRESH, we split into two new nodes.
    
    def __init__(self, parent = None, dims = 0, reset_id = False, kernel_type = "rbf", gamma_type = "auto", stage = 0, search_space_id = None):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        # if not is_root:
        #     assert type( parent ) == type( self )
        self.dims          = dims
        self.x_bar_source  = float('inf')
        self.x_bar         = float('inf')
        self.potential     = float('inf')
        self.n             = 0
        self.uct           = 0
        self.classifier    = Classifier( [], self.dims, kernel_type, gamma_type, stage = stage)
        self.classifier.node = self
        self.stage = stage
        self.search_space_id = search_space_id
        
        #insert curt into the kids of parent
        self.parent        = parent        
        self.kids          = [] # 0:good, 1:bad
        
        self.bag_source    = []
        self.source_id     = []
        self.bag_target    = []
        self.target_id     = []
        self.bag           = []
        
        self.is_svm_splittable = False 
        
        if reset_id:
            Node.obj_counter = 0

        self.id            = Node.obj_counter
                
        #data for good and bad kids, respectively
        Node.obj_counter += 1
    
    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append( good_kid )
        self.kids.append( bad_kid )
        assert self.kids[0].classifier.get_mean() >= self.kids[1].classifier.get_mean()
        
    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False
    
    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False 
            
    def visit(self):
        self.n += 1
        
    def print_bag(self):
        sorted_bag = sorted(self.bag.items(), key=operator.itemgetter(1))
        print("BAG"+"#"*10)
        for item in sorted_bag:
            print(item[0],"==>", item[1])            
        print("BAG"+"#"*10)
        print('\n')
        
    def update_bag(self, samples, source_id = None, target_samples = None, target_id = None):
        assert len(samples) > 0
        
        self.bag.clear()
        self.bag.extend( samples )
        
        if source_id:
            self.source_id.clear()
            self.source_id.extend( source_id )
        
            assert len(self.bag)==len(self.source_id)
        
        if target_samples:
            self.bag_target.clear()
            self.bag_target.extend( target_samples )
        
        if target_id:
            self.target_id.clear()
            self.target_id.extend( target_id )
        
        assert len(self.bag_target) == len(self.target_id)
        
        self.classifier.update_samples( self.bag )
        if len(self.bag) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_splittable_svm()

        self.x_bar             = self.classifier.get_mean()
        self.n                 = len( self.bag ) + len( self.bag_source)
    
    def update_bag_source(self):
        boundary = self.classifier.svm
        samples = [sample for sample, value in self.bag_source]
        if len(samples) == 0:
            self.kids[0].bag_source = []
            self.kids[0].source_id = []
            self.kids[1].bag_source = []
            self.kids[1].source_id = []
            return
        samples = np.array(samples).reshape(-1, self.dims)
        labels = boundary.predict(samples)
        good_source_idx = np.where(labels == 0)[0]
        bad_source_idx = np.where(labels == 1)[0]
        self.kids[0].bag_source = [self.bag_source[i] for i in list(good_source_idx)]
        self.kids[0].source_id = [self.source_id[i] for i in list(good_source_idx)]
        self.kids[1].bag_source = [self.bag_source[i] for i in list(bad_source_idx)]
        self.kids[1].source_id = [self.source_id[i] for i in list(bad_source_idx)]
        
    def update_bag_target(self):
        boundary = self.classifier.svm
        samples = [sample for sample, value in self.bag_target]
        if len(samples) == 0:
            self.kids[0].bag_target = []
            self.kids[0].target_id = []
            self.kids[1].bag_target = []
            self.kids[1].target_id = []
            return
        samples = np.array(samples).reshape(-1, self.dims)
        labels = boundary.predict(samples)
        good_target_idx = np.where(labels == 0)[0]
        bad_target_idx = np.where(labels == 1)[0]
        self.kids[0].bag_target = [self.bag_target[i] for i in list(good_target_idx)]
        self.kids[0].target_id = [self.target_id[i] for i in list(good_target_idx)]
        self.kids[1].bag_target = [self.bag_target[i] for i in list(bad_target_idx)]
        self.kids[1].target_id = [self.target_id[i] for i in list(bad_target_idx)]
        
    def add_sample(self, sample):
        '''
        add sample into node

        Args:
            sample: (x, y)
        
        Returns:
            Null
        '''
        self.bag.append(sample)
        self.classifier.update_samples(self.bag)
        if len(self.bag) <= 2:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_splittable_svm()

        self.x_bar             = self.classifier.get_mean()
        self.n                 = len( self.bag ) + len( self.bag_source)
        
    def clear_data(self):
        self.bag.clear()
    
    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)
    
    def pad_str_to_8chars(self, ins, total):
        if len(ins) <= total:
            ins += ' '*(total - len(ins) )
            return ins
        else:
            return ins[0:total]
            
    def get_rand_sample_from_bag(self):
        if len( self.bag ) > 0:
            upeer_boundary = len(list(self.bag))
            rand_idx = np.random.randint(0, upeer_boundary)
            return self.bag[rand_idx][0]
        else:
            return None
            
    def get_parent_str(self):
        return self.parent.get_name()
            
    def propose_samples_transformer(self, algo, num_samples, path, lb, ub, samples, from_node=False, model_path=None):
        proposed_X = self.classifier.propose_samples_transformer(algo, num_samples, path, lb, ub, samples, source_samples = self.bag_source if from_node else None, model_path=model_path)
        return proposed_X
    
    def propose_samples_bo(self, num_samples, path, lb, ub, samples, from_node=False, local=False):
        proposed_X = self.classifier.propose_samples_bo(num_samples, path, lb, ub, samples, source_samples = self.bag_source if from_node else None, local=local)
        return proposed_X
    
    def propose_samples_turbo(self, num_samples, path, func):
        proposed_X, fX = self.classifier.propose_samples_turbo(num_samples, path, func)
        return proposed_X, fX

    def propose_samples_rand(self, num_samples):
        assert num_samples > 0
        samples = self.classifier.propose_samples_rand(num_samples)
        return samples
        
    def __str__(self):
        name   = self.get_name()
        name   = self.pad_str_to_8chars(name, 7)
        name  += ( self.pad_str_to_8chars( 'is good:' + str(self.is_good_kid() ), 15 ) )
        name  += ( self.pad_str_to_8chars( 'is leaf:' + str(self.is_leaf() ), 15 ) )
        
        val    = 0
        name  += ( self.pad_str_to_8chars( ' val:{0:.4f}   '.format(round(self.get_xbar(), 3) ), 20 ) )
        name  += ( self.pad_str_to_8chars( ' potential:{0:.4f}   '.format(round(self.get_potential(), 3) ), 20 ) )
        name  += ( self.pad_str_to_8chars( ' uct:{0:.4f}   '.format(round(self.get_uct(), 3) ), 20 ) )

        name  += self.pad_str_to_8chars( 'sp/n:'+ str(len(self.bag))+"/"+str(self.n), 15 )
        # upper_bound = np.around( np.max(self.classifier.X, axis = 0), decimals=2 )
        # lower_bound = np.around( np.min(self.classifier.X, axis = 0), decimals=2 )
        # boundary    = ''
        # for idx in range(0, self.dims):
            # boundary += str(lower_bound[idx])+'>'+str(upper_bound[idx])+' '
            
        #name  += ( self.pad_str_to_8chars( 'bound:' + boundary, 60 ) )

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)
        
        name += (' parent:' + parent)
        
        kids = ''
        kid  = ''
        for k in self.kids:
            kid   = self.pad_str_to_8chars( k.get_name(), 10 )
            kids += kid
        name  += (' kids:' + kids)
        
        return name
    

    def get_uct(self, Cp = 10):
        # TODO: fix bugs on xbar
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        if self.stage == 0:
            return self.x_bar + 2*Cp*math.sqrt( 2* np.power(self.parent.n, 0.5) / self.n )
        return self.potential + 2*Cp*math.sqrt( 2* np.power(self.parent.n, 0.5) / self.n )
    
    def get_xbar(self):
        return self.x_bar
    
    def get_xbar_source(self):
        return self.x_bar_source
    
    def get_potential(self):
        return self.potential
    
    def get_n(self):
        return self.n
        
    def train_and_split(self):
        assert len(self.bag) >= 2
        self.classifier.update_samples( self.bag )
        good_kid_data, bad_kid_data, good_source_id, bad_source_id = self.classifier.split_data()
        assert len( good_kid_data ) + len( bad_kid_data ) ==  len( self.bag )
        if self.stage == 1:
            assert good_source_id is None
            assert bad_source_id is None
        return good_kid_data, bad_kid_data, good_source_id, bad_source_id

    def plot_samples_and_boundary(self, func):
        name = self.get_name() + ".pdf"
        self.classifier.plot_samples_and_boundary(func, name)

    def sample_arch(self):
        if len(self.bag) == 0:
            return None
        net_str = np.random.choice( list(self.bag.keys() ) )
        del self.bag[net_str]
        return json.loads(net_str )