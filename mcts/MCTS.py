# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
import datetime
from .Node import Node
from .utils import latin_hypercube, from_unit_cube, manhattan_distance_N, Kendall_coefficient, KL_distance
from torch.quasirandom import SobolEngine
import torch
from tqdm import tqdm
import wandb
from .utils import get_data_in_node
from omegaconf import DictConfig, OmegaConf
import hydra

ts = datetime.datetime.utcnow() + datetime.timedelta(hours=+8)
ts_name = f'-ts{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}'

class MCTS:
    def __init__(self, lb, ub, dims, ninits, func, Cp = 1, leaf_size = 20, kernel_type = 'rbf', gamma_type = 'auto', state = None, source_X = None, source_Y = None, target_samples=[], target_id=[], source_id = None, stage = None, search_space_id = None, dataset_id = None, similar = None, log = True, mode=None):
        self.dims                    =  dims
        self.samples                 =  [] # (samples, value)
        self.values                  =  []
        self.nodes                   =  []
        self.Cp                      =  Cp
        self.lb                      =  lb
        self.ub                      =  ub
        self.ninits                  =  ninits
        self.func                    =  func
        self.curt_best_value         =  float('-inf')
        self.curt_best_sample        =  None
        self.best_value_trace        =  []
        self.sample_counter          =  0
        self.visualization           =  False
        self.weights                 =  dict()
        
        self.LEAF_SAMPLE_SIZE        =  leaf_size
        self.kernel_type             =  kernel_type
        self.gamma_type              =  gamma_type
        
        self.solver_type             = 'bo' #solver can be 'bo' or 'turbo'
        
        print('gamma_type:', gamma_type)
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node( parent = None, dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type, stage = stage, search_space_id = search_space_id )
        self.nodes.append( root )
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.state = state
        self.stage = stage
        self.search_space_id = search_space_id
        self.dataset_id = dataset_id
        
        self.source_id = source_id
        self.history_sample = None
        self.target_samples = target_samples
        self.target_id = target_id
        self.similar = similar
        self.weight_update = None
        self.log = log
        self.decay_factor = 1.0
        self.mode = mode
        self.method = None
        self.model_path = None
        
        if self.state is None:
            self.ROOT.bag_target.extend( target_samples )
            self.ROOT.target_id.extend( target_id )
            self.init_train()
        else:
            self.ROOT.bag_target.extend( target_samples )
            self.ROOT.target_id.extend( target_id )
            self.init_learn(source_X, source_Y)
            
    def populate_training_data(self):
        # only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root  = Node(parent = None,   dims = self.dims, reset_id = True, kernel_type = self.kernel_type, gamma_type = self.gamma_type, stage = self.stage, search_space_id = self.search_space_id )
        self.nodes.append( new_root )
        
        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag( self.samples, self.source_id, self.target_samples, self.target_id )
        
    
    def get_subtree_nodes(self, root):
        '''
        get nodes from subtree except root

        Args:
            node (Node): the root of the subtree

        Returns:
            all_nodes (list): nodes from subtree except node
        '''
        queue = [root]
        all_nodes = []
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)

            # add node into spliting queue
            for child in node.kids:
                queue.append(child)
        all_nodes.remove(root)
        return all_nodes
    
    def clear_subtree_data(self, node):
        subtree_nodes = self.get_subtree_nodes(node)
        for node in subtree_nodes:
            node.clear_data()
    
    def reconstruct_tree(self):
        '''
        if value(right)>value(left), reconstruct the tree from parent(left, right)
        '''
        queue = [self.ROOT]
        
        while queue:
            node = queue.pop(0)
            if node.is_leaf() == False and node.kids[0].get_potential() < node.kids[1].get_potential():
                self.clear_subtree_data(node) 
                node.kids.clear()
            elif len(node.kids) > 0:
                queue.append(node.kids[0])
                queue.append(node.kids[1])

                    
    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            if node.is_leaf() == True and len(node.bag) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable == True:
                status.append( True  )
            else:
                status.append( False )
        return np.array( status )
        
    def get_split_idx(self):
        split_by_samples = np.argwhere( self.get_leaf_status() == True ).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
    
    def split_tree(self):
        while self.is_splitable():
            to_split = self.get_split_idx()
            #print('==>to split:', to_split, ' total:', len(self.nodes) )
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable == True
                # print('spliting node:', parent.get_name(), len(parent.bag))
                
                good_kid_data, bad_kid_data, good_source_id, bad_source_id = parent.train_and_split()
                
                if self.stage == 1:
                    assert good_source_id is None
                    assert bad_source_id is None
                    
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent
                assert len(good_kid_data) + len(bad_kid_data) == len(parent.bag)
                assert len(good_kid_data) > 0
                assert len(bad_kid_data)  > 0
                good_kid = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type, stage=self.stage, search_space_id = self.search_space_id )
                bad_kid  = Node(parent = parent, dims = self.dims, reset_id = False, kernel_type = self.kernel_type, gamma_type = self.gamma_type, stage=self.stage, search_space_id = self.search_space_id )
                
                good_kid.update_bag( good_kid_data, good_source_id )
                bad_kid.update_bag(  bad_kid_data, bad_source_id  )
            
                parent.update_kids( good_kid = good_kid, bad_kid = bad_kid )
                parent.update_bag_target()
                
                if self.stage ==1:
                    self.update_potential(good_kid)
                    self.update_potential(bad_kid)
                    parent.update_bag_source() 
                    parent.update_bag_target()
                    
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
            #print('continue split:', self.is_splitable())
                
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        assert len(self.ROOT.bag) == len(self.samples)
        assert len(self.nodes)    == 1
        self.split_tree()
        # self.print_tree()
        # self.print_best_node()
        
    def reset_nodes_list(self):
        def dfs(node):
            self.nodes.append(node)
            if node.is_leaf() == False:
                dfs(node.kids[0])
                dfs(node.kids[1]) 
        self.nodes.clear()
        dfs(self.ROOT)

    # Done: complete tree fine-tune part
    def dynamic_treeify_from_subtree(self):
        self.reconstruct_tree()
        self.reset_nodes_list()
        
        reconstruct_times = len(self.get_split_idx()) if self.is_splitable() else 0
        self.split_tree()
        # self.print_tree()
        # self.print_best_node()
        return reconstruct_times
        
    def collect_samples(self, sample, value = None):
        # Done: to perform some checks here
        if value == None:
            value = self.func(sample)*-1
            
        if value > self.curt_best_value:
            self.curt_best_value  = value
            self.curt_best_sample = sample 
            self.best_value_trace.append( (value, self.sample_counter) )
        self.sample_counter += 1
        self.samples.append( (sample, value) )
        self.values.append(value)
        
        if self.state is None:
            if self.log:
                curt_best_value = self.curt_best_value  if self.mode in ['real_world','design_bench'] else np.absolute(self.curt_best_value)
                wandb.log({
                    'sample counter': self.sample_counter,
                    'sample value': np.absolute(value),
                    'best value': curt_best_value,
                    'node number': len(self.nodes),
                })
        return value
    
    def get_standard_xbar(self, bag):
        assert len(self.values) == len(self.samples)
        mean_y = np.mean(self.values)
        std_y = np.std(self.values) if len(self.values) > 1 else None
        y_values = [y for (x, y) in bag]
        standardized_y_values = [(y - mean_y) / std_y if std_y else 0. for y in y_values]
        return np.mean(standardized_y_values)
        
    def init_train(self):
        # here we use latin hyper space to generate init samples in the search space

        init_points = latin_hypercube(self.ninits, self.dims)
        init_points = from_unit_cube(init_points, self.lb, self.ub)
    
        for point in init_points:
            self.collect_samples(point)
        
        print('='*10 + 'collect '+ str(len(self.samples) ) +' points for initializing MCTS'+'='*10)
        print('lb:', self.lb)
        print('ub:', self.ub)
        print('Cp:', self.Cp)
        print('inits:', self.ninits)
        print('dims:', self.dims)
        print('='*58)
    
    def init_learn(self, source_X, source_Y):
        dataloader = tqdm(range(source_X.shape[0]))
        dataloader.set_description('Load X, Y from source task...')
        for i in dataloader:
            self.collect_samples(source_X[i], source_Y[i])
        print('='*10 + 'load '+ str( source_X.shape[0] ) +' points for pre-learning MCTS'+'='*10)
        print('dims:', self.dims)
        print('='*58)

    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)
        print('node num:', len(self.nodes))

    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print('=====>loads:', len(self.samples),' samples' )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print('dumping the agent.....')
        with open(node_path,'wb') as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, 'wb') as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'best_values_trace'
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, 'a') as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_xbar() )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            if curt_node.is_leaf() == False and self.visualization == True:
                curt_node.plot_samples_and_boundary(self.func)
            print('=>', curt_node.get_name(), end=' ' )
        print('')
        return curt_node, path

    def select_best(self):
        curt_node = self.ROOT
        path = [ ]
        while curt_node.is_leaf() == False:
            path.append( (curt_node, 0) )
            curt_node = curt_node.kids[0]
            print('=>', curt_node.get_name(), end=' ' )
        return curt_node, path
    
    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        path      = [ ]
        
        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append( i.get_uct(self.Cp) )
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append( (curt_node, choice) )
            curt_node = curt_node.kids[choice]
            print('=>', curt_node.get_name(), end=' ' )
        print('')
        self.decompose(curt_node)
        print([item[1] for item in path])
        return curt_node, path
    
    def decompose(self, node):
        uct = node.get_uct(self.Cp)
        potential = node.potential
        exploration = node.get_uct(self.Cp) - potential
        current = self.get_standard_xbar(node.bag) if len(node.bag) > 0 else 0
        history = potential - current
        sum = np.absolute(exploration)+np.absolute(current)+np.absolute(history)
        exploration_percent = np.absolute(exploration) / sum
        current_percent = np.absolute(current) / sum
        history_percent = np.absolute(history) / sum
        print(f'uct={uct}, exploration={exploration_percent}, current={current_percent}, history={history_percent}')
        wandb.log({
            'exploration': exploration_percent,
            'current': current_percent,
            'history': history_percent,
        })
    
    def backpropogate(self, leaf, acc):
        '''
        update x_bar
        '''
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n    += 1
            curt_node       = curt_node.parent

    def backpropogate_for_all(self, leaf, sample, acc):
        '''
        update x_bar and samples
        '''
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.add_sample((sample, acc))
            curt_node = curt_node.parent
        
    def get_leaves(self):
        leaves = []
        for node in self.nodes:
            if node.is_leaf():
                leaves.append(node)
        return leaves 
    
    def get_samples(self):
        X  = []
        y  = []
        for sample in self.samples:
            X.append(sample[0])
            y.append(sample[1])
        X  = np.asarray(X).reshape(len(self.samples), -1)
        y = np.asarray(y).reshape(-1)
        return X, y
    
    def get_similarity(self, similarity, N, sid, did):
        target_X, target_y = self.get_samples()
        if similarity in ['optimal', 'topN', 'Npercent']:
            return manhattan_distance_N(self.history_sample[sid][did]['X'], self.history_sample[sid][did]['y'], target_X, target_y, N)
        elif similarity == 'distribution':
            return Kendall_coefficient(self.history_sample[sid][did]['model'], target_X, target_y, self.lb, self.ub)
        elif similarity == 'KL':
            return KL_distance(self.history_sample[sid][did]['X'], self.history_sample[sid][did]['y'], target_X, target_y, self.lb, self.ub)
        else:
            raise ValueError
    
    def rank_by_value(self, distances):
        sorted_items = sorted(distances.items(), key=lambda item: item[1])
        ranked_dict = {}
        current_rank = 0
        previous_value = None
        for index, (key, value) in enumerate(sorted_items):
            if value != previous_value:
                current_rank = index
            ranked_dict[key] = current_rank
            previous_value = value
        return ranked_dict
    
    def update_weights(self, similarity, N, alpha):
        distances = dict()
        for sid in self.history_sample.keys():
            for did in self.history_sample[sid].keys():
                name = f'{sid}+{did}'
                if self.weight_update=='all-one':
                    self.weights[name] = 1.0
                else:
                    distances[name] = self.get_similarity(similarity, N, sid, did)
        if self.weight_update=='exponential':
            ranked_dict = self.rank_by_value(distances)
            self.weights = {key: 1 / 2**rank for key, rank in ranked_dict.items()}
        elif self.weight_update =='linear-half':
            ranked_dict = self.rank_by_value(distances)
            task_num = len(ranked_dict)
            self.weights = {}
            for key, rank in ranked_dict.items():
                if rank < np.round(task_num * alpha) :
                    weight = 1 - rank / (task_num * alpha)
                else:
                    weight = 0.1
                self.weights[key] = weight
        
    def update_potential(self, node):
        potential = 0
        weight_sum = 1e-6
        for sid in self.history_sample.keys():
            for did in self.history_sample[sid].keys():
                name = f'{sid}+{did}'
                have_data, _, history_y = get_data_in_node(node, name)
                if have_data:
                    potential = potential + self.weights[name] * np.mean(history_y)
                    weight_sum += self.weights[name]
        potential /= weight_sum
        if len(node.bag) > 0:   
            potential += self.get_standard_xbar(node.bag)
        node.potential = potential  
        
    def search(self, iterations, threshold, local):
        totol_time_start = time.time()
        self.LEAF_SAMPLE_SIZE = threshold
        for idx in tqdm(range(self.sample_counter, iterations)):
            print('')
            print('='*10)
            print('iteration:', idx)
            print('='*10)
            self.dynamic_treeify()
            leaf, path = self.select()
            for i in range(0, 1):
                if self.solver_type == 'bo':
                    samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples, from_node=True, local=local )
                elif self.solver_type == 'turbo':
                    samples, values = leaf.propose_samples_turbo( 10000, path, self.func )
                else:
                    raise Exception('solver not implemented')
                for idx in range(0, len(samples)):
                    if self.solver_type == 'bo':
                        value = self.collect_samples( samples[idx])
                    elif self.solver_type == 'turbo':
                        value = self.collect_samples( samples[idx], values[idx] )
                    else:
                        raise Exception('solver not implemented')
                    
                    self.backpropogate( leaf, value )
                    
            print('total samples:', len(self.samples) )
            print('current best f(x):', np.absolute(self.curt_best_value) )
            # print('current best x:', np.around(self.curt_best_sample, decimals=1) )
    
            print('current best x:', self.curt_best_sample )
        
        totol_time_end = time.time()
        totol_time = totol_time_end - totol_time_start
        wandb.log({
            'totol time': totol_time
        })
        if self.log:    
            wandb.finish()

    def learn(self, data_X, data_Y):
        self.dynamic_treeify()
    
    def check_bag_target(self):
        leaves = self.get_leaves()
        count = 0
        for leaf in leaves:
            count+=len(leaf.target_id)
            assert len(leaf.bag_target)==len(leaf.target_id)
        assert count == len(self.target_id)

    def print_best_node(self):
        node = self.ROOT
        while node.kids:
            node = node.kids[0]
        print('best node', node)

    def search_from_tree(self, iterations, threshold, local, similarity, N, gamma, alpha):
        # only maintain tree structure
        self.samples = []
        self.values = []
        self.curt_best_value = float('-inf')
        self.curt_best_sample  =  None
        self.best_value_trace  =  []
        self.sample_counter    =  0
        self.stage = 1
        for node in self.nodes:
            node.bag_source = node.bag[:]
            assert len(node.bag_source)==len(node.source_id)
            node.bag.clear()
            node.x_bar_source = node.x_bar
            node.potential = node.x_bar
            node.x_bar = 0
            node.is_svm_splittable = False
            node.stage = 1
            node.classifier.samples = []
            node.classifier.X = np.array([])
            node.classifier.fX = np.array([])
            node.classifier.stage = 1
        self.LEAF_SAMPLE_SIZE = threshold
        
        # debug
        self.check_bag_target()
        
        # wandb init
        if self.log:
            wandb.init(
                    project='mcts-transfer',
                    name=f'mcts-transfer-{self.search_space_id}-{self.dataset_id}-{ts_name}',
                    job_type=f'{self.method}-new',
                    tags=[f'dim={self.dims}', f'similar={self.similar}', f'search_space_id={self.search_space_id}', f'dataset_id={self.dataset_id}']
                )
            
        for iter in tqdm(range(self.sample_counter, iterations)):
            print('')
            print('='*10)
            print('iteration:', iter)
            print('='*10)
            
            leaf, path = self.select()
            for i in range(0, 1):
                samples = leaf.propose_samples_bo( 1, path, self.lb, self.ub, self.samples, from_node=True, local=local )
                samples = np.nan_to_num(samples, nan=0.0)
                
                for idx in range(0, len(samples)):
                    eval_start_time = time.time()
                    value = self.collect_samples( samples[idx])
                    eval_end_time = time.time()
                    evaluation_time = eval_end_time - eval_start_time                                        
                        
                    curt_best_value = self.curt_best_value  if self.mode in ['real_world','superconductor', 'ant', 'dkitty'] else np.absolute(self.curt_best_value)
                    if self.log:
                        wandb.log({
                            'sample counter': self.sample_counter,
                            'sample value': np.absolute(value),
                            'best value': curt_best_value,
                            'node number': len(self.nodes),
                        })
                    
                    self.backpropogate_for_all( leaf, samples[idx], value )
                    self.update_weights(similarity, N, alpha)
                    
                    
                    for key in self.weights.keys():
                        self.weights[key] = self.weights[key] * self.decay_factor
                        
                    for node in self.nodes:
                        self.update_potential(node)
                        
                    if self.log:
                        wandb.log(self.weights)

                reconstruct_times = self.dynamic_treeify_from_subtree() # calculate subtree reconstruct frequency
            
            # debug
            self.check_bag_target()
            self.decay_factor *= gamma
            print('total samples:', len(self.samples) )
            print('current best f(x):', np.absolute(self.curt_best_value) )
            # print('current best x:', self.curt_best_sample )

        if self.log:
            wandb.finish()
            
    
        