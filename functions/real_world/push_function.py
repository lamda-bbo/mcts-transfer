from functions.real_world.push_utils import b2WorldInterface, make_base, create_body, end_effector, run_simulation
import numpy as np
from functions.utils import get_data
from data.get_data import get_real_world_mixed_data


class PushReward:
    def __init__(self, args):
        self.dim = 14
        self.lb = np.zeros(self.dim) # for algorithm
        self.ub = np.ones(self.dim) # for algorithm
        self.args = args
        self.data = self.load_data()
        self.DATA_MAXIMIZE = 1

        # domain of this function
        self.xmin = [-5., -5., -10., -10., 2., 0., -5., -5., -10., -10., 2., 0., -5., -5.]
        self.xmax = [5., 5., 10., 10., 30., 2.*np.pi, 5., 5., 10., 10., 30., 2.*np.pi, 5., 5.]

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

    def load_similar_source_data(self):
        return get_real_world_mixed_data(self.args.search_space_id, self.args.dataset_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_real_world_mixed_data(self.args.search_space_id, self.args.dataset_id, self.data, self.args.similar, self.DATA_MAXIMIZE)
    
    def load_data(self):
        return get_data(self.args.mode)
    
    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) \
            + np.linalg.norm(np.array(self.gxy2) - np.array(self.sxy2))
    @property
    def dx(self):
        # dimension of the input
        return self._dx
    
    def __call__(self, x):
        # returns the reward of pushing two objects with two robots
        x = np.array(x)  #[0, 1]
        lb = np.array(self.xmin)
        ub = np.array(self.xmax)
        x = x * (ub - lb) + lb  #[xmin, xmax]
        # print(x)
        
        rx = float(x[0])
        ry = float(x[1])
        xvel = float(x[2])
        yvel = float(x[3])
        simu_steps = int(float(x[4]) * 10)
        init_angle = float(x[5])
        rx2 = float(x[6])
        ry2 = float(x[7])
        xvel2 = float(x[8])
        yvel2 = float(x[9])
        simu_steps2 = int(float(x[10]) * 10)
        init_angle2 = float(x[11])
        rtor = float(x[12])
        rtor2 = float(x[13])
        
        initial_dist = self.f_max

        world = b2WorldInterface(True)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
            'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, 'rectangle', (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, 'circle', 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2,ry2), base, init_angle2, hand_shape, hand_size)
        (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel, \
                                      xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        result = initial_dist - ret1 - ret2 
        return -1 * result

# def push(x):
#     f = PushReward()
#     lb = np.array(f.xmin)
#     ub = np.array(f.xmax)
#     x = x * (ub - lb) + lb
#     print(x)
#     return f(x)

# def get_dim():
#     return len(PushReward().xmin)

def main():
    f = PushReward()
    # x = np.random.uniform(f.xmin, f.xmax)
    x = np.random.rand(14)
    print('Input = {}'.format(x))
    print('Output = {}'.format(f(x)))
    # print(push(x))

if __name__ == '__main__':
    main()