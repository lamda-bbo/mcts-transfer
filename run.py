from prelearn import mcts_prelearn
import argparse
import warnings
from functions import *
import os
from utils import load_model
import numpy as np
np.random.seed(1203)

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='hpob', choices=['Sphere2D', 'hpob', 'bbob', 'real_world', 'design_bench'])
parser.add_argument('--search-space-id', type=str, default='4796')
parser.add_argument('--dataset-id', type=str, default='3549')
parser.add_argument('--method', type=str, default='mcts-transfer')
parser.add_argument('--iteration', type=int, default=100)
parser.add_argument('--rep', type=int, default=3)

# parameters
parser.add_argument('--similar', type=str, choices=['similar', 'combine', 'mix-similar', 'mix-both', 'unsimilar']) # source data composition
parser.add_argument('--weight-update', type=str, default='linear-half', choices=['all-one', 'linear-half', 'exponential']) # weight update methods
parser.add_argument('--alpha', type=float, default=0.5) # consider {alpha} percent tasks as important tasks, only valid for linear-half weight update method
parser.add_argument('--Cp', type=float, default=1.0) # MCTS Cp
parser.add_argument('--gamma', type=float, default=0.99) # weight decay factor
parser.add_argument('--threshold', type=int, default=10) # leaf threshold
parser.add_argument('--kernel-type', type=str, default='rbf') # binary classifier kernel-type
parser.add_argument('--local', action='store_true') # local modeling / global modeling 
parser.add_argument('--similarity', type=str, default='optimal', choices=['optimal', 'topN', 'Npercent','distribution', 'KL']) # similarity measuring methods
parser.add_argument('--N', type=float, default=1.0) # use top {N} points to update similarity, only valid for similarity=optimal(N=1.0), topN(N>1), Npercent(0<N<1)
args = parser.parse_args()

if args.similarity == 'optimal':
    args.N=1.0

problems  = {
    # Sphere2D
    'Sphere2D': Sphere2DProblem,

    # BBOB
    'GriewankRosenbrock': GriewankRosenbrock,
    'Lunacek': Lunacek,
    'Rastrigin': Rastrigin,
    'RosenbrockRotated': RosenbrockRotated,
    'SharpRidge': SharpRidge,

    # Design-bench
    'superconductor': Superconductor,
    'ant': AntMorphology,
    'dkitty': DKittyMorphology,

    # Real-world
    'LunarLander': LunarLanderProblem,
    'RobotPush': PushReward,
    'Rover': Rover,

    # HPOB
    '5860': HPOBProblem,
    '5970': HPOBProblem,
    '5859': HPOBProblem,
    '5889': HPOBProblem,
    '7607': HPOBProblem,
    '7609': HPOBProblem,
    '5906': HPOBProblem,
    '5971': HPOBProblem,
}

def run_mcts_transfer(args):
    '''
    Workflow of MCTS-transfer
    '''
    search_space_id = args.search_space_id
    dataset_id = args.dataset_id

    problem = problems[search_space_id](args)

    model_saving_dir = os.path.join(os.getcwd(), 'mcts_models', f'model_{args.mode}', args.kernel_type)
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir, exist_ok=True)
    
    # ========================== MCTS-transfer Pre-learning Start ==========================
    prelearned_model_path = mcts_prelearn(args, problem, model_saving_dir) 
    print('='*10, f'Model has been pre-learned for problem {search_space_id}', '='*10)
    # ========================== MCTS-transfer Pre-learning End ==========================

    # ========================== MCTS-transfer Optimization Start ==========================
    for i in range(args.rep):
        print('='*10, f'Start optimizing problem {search_space_id} from the pre-learned tree', '='*10)
        model = load_model(prelearned_model_path)
        model.Cp = args.Cp
        model.weight_update = args.weight_update
        model.func = problem
        model.dataset_id = dataset_id
        model.decay_factor = 1.0
        model.mode = args.mode
        model.kernel_type = args.kernel_type
        model.search_from_tree(iterations = args.iteration, threshold = args.threshold, local = args.local, similarity = args.similarity, N = args.N, gamma = args.gamma, alpha = args.alpha)
    # ========================== MCTS-transfer Optimization End ==========================

run_mcts_transfer(args)
    
    