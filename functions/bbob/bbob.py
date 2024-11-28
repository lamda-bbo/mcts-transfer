from __future__ import annotations

'''Experimenter function implementations for BBOB functions.'''

import hashlib
import math
from typing import Any, Callable, Sequence
import numpy as np
from scipy import stats
from functions.utils import get_data
from data.get_data import get_combined_data, get_similar_data

## Utility Functions for BBOB.
def LambdaAlpha(alpha: float, dim: int) -> np.ndarray:
    '''The BBOB LambdaAlpha matrix creation function.

    Args:
      alpha: Function parameter.
      dim: Dimension of matrix created.

    Returns:
      Diagonal matrix of dimension dim with values determined by alpha.
    '''
    lambda_alpha = np.zeros([dim, dim])
    for i in range(dim):
      exp = (0.5 * (float(i) / (dim - 1))) if dim > 1 else 0.5
      lambda_alpha[i, i] = alpha**exp
    return lambda_alpha


def ArrayMap(vector: np.ndarray, fn: Callable[[float], float]) -> np.ndarray:
    '''Create a new array by mapping fn() to each element of the original array.

    Args:
      vector: ndarray to be mapped.
      fn: scalar function for mapping.

    Returns:
      New ndarray be values mapped by fn.
    '''
    results = np.zeros(vector.shape)
    for i, v in enumerate(vector.flat):
      results.flat[i] = fn(v)
    return results


def Tosz(element: float) -> float:
    '''The BBOB T_osz function.

    Args:
      element: float input.

    Returns:
      Tosz(input).
    '''
    x_carat = 0.0 if element == 0 else math.log(abs(element))
    c1 = 10.0 if element > 0 else 5.5
    c2 = 7.9 if element > 0 else 3.1
    return np.sign(element) * math.exp(
        x_carat + 0.049 * (math.sin(c1 * x_carat) + math.sin(c2 * x_carat)))


def Tasy(vector: np.ndarray, beta: float) -> np.ndarray:
    '''The BBOB Tasy function.

    Args:
      vector: ndarray
      beta: Function parameter

    Returns:
      ndarray with values determined by beta.
    '''
    dim = len(vector)
    result = np.zeros([dim, 1])
    for i, val in enumerate(vector.flat):
      if val > 0:
        t = i / (dim - 1.0) if dim > 1 else 1
        exp = 1 + beta * t * (val**0.5)
      else:
        exp = 1
      result[i] = val**exp
    return result


def SIndex(dim: int, to_sz) -> float:
    '''Calculate the BBOB s_i.

    Assumes i is 0-index based.

    Args:
      dim: dimension
      to_sz: values

    Returns:
      float representing SIndex(i, d, to_sz).
    '''
    s = np.zeros([
        dim,
    ])
    for i in range(dim):
      if dim > 1:
        s[i] = 10**(0.5 * (i / (dim - 1.0)))
      else:
        s[i] = 10**0.5
      if i % 2 == 0 and to_sz[i] > 0:
        s[i] *= 10
    return s


def Fpen(vector: np.ndarray) -> float:
    '''The BBOB Fpen function.

    Args:
      vector: ndarray.

    Returns:
      float representing Fpen(vector).
    '''
    return sum([max(0.0, (abs(x) - 5.0))**2 for x in vector.flat])


def _IntSeeds(any_seeds: Sequence[Any], *, byte_length: int = 4) -> list[int]:
    '''Array of integers that can be used as random state seed.'''
    int_seeds = []
    for s in any_seeds:
      # Encode into 4 byte_length worth of a hexadecimal string.
      hashed = hashlib.shake_128(str(s).encode('utf-8')).hexdigest(byte_length)
      int_seeds.append(int(hashed, 16))
    return int_seeds


def _ToFloat(a: int, b: np.ndarray) -> np.ndarray:
    '''Convert a%b where b is an int into a float on [-0.5, 0.5].'''
    return (np.int64(a) % b) / np.float64(b) - 0.5


def _R(dim: int, seed: int, *moreseeds: Any) -> np.ndarray:
    '''
    Returns an orthonormal rotation matrix.

    Args:
      dim: size of the resulting matrix.
      seed: int seed. If set to 0, this function returns an identity matrix
        regardless of *moreseeds.
      *moreseeds: Additional parameters to include in the hash. Arguments are
        converted to strings first.

    Returns:
      Array of shape (dim, dim), representing a rotation matrix.
    '''
    if seed == 0:
      return np.identity(dim)
    rng = np.random.default_rng(_IntSeeds(((seed, dim) + moreseeds)))
    return stats.special_ortho_group.rvs(dim, random_state=rng)

## BBOB Functions.
class GriewankRosenbrock:
    def __init__(self, args = None):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)
        self.args = args
        self.data = get_data(self.args.mode)   
        self.DATA_MAXIMIZE = 1

    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)
    
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        '''Implementation for BBOB GriewankRosenbrock function.'''
        dim = len(arr)
        r_x = np.matmul(_R(dim, seed, b'R'), arr)
        # Slightly off BBOB documentation in order to center optima at origin.
        # Should be: max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,)).
        z_arr = max(1.0, (dim**0.5) / 8.0) * r_x + np.ones((dim,))
        s_arr = np.zeros(dim)
        for i in range(dim - 1):
          s_arr[i] = 100.0 * (z_arr[i]**2 - z_arr[i + 1])**2 + (z_arr[i] - 1)**2

        total = 0.0
        for i in range(dim - 1):
          total += (s_arr[i] / 4000.0 - math.cos(s_arr[i]))

        return (10.0 * total) / (dim - 1) + 10

    
class Lunacek:
    def __init__(self, args = None):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)
        self.args = args
        self.data = get_data(self.args.mode)
        self.DATA_MAXIMIZE = 1

    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)
    
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        '''Implementation for BBOB Lunacek function.'''
        dim = len(arr)
        arr.shape = (dim, 1)
        mu0 = 2.5
        s = 1.0 - 1.0 / (2.0 * (dim + 20.0)**0.5 - 8.2)
        mu1 = -((mu0**2 - 1) / s)**0.5

        x_opt = np.array([mu0 / 2] * dim)
        x_hat = np.array([2 * arr[i, 0] * np.sign(x_opt[i]) for i in range(dim)])
        x_vec = x_hat - mu0
        x_vec.shape = (dim, 1)
        x_vec = np.matmul(_R(dim, seed, b'R'), x_vec)
        z_vec = np.matmul(LambdaAlpha(100, dim), x_vec)
        z_vec = np.matmul(_R(dim, seed, b'Q'), z_vec)

        s1 = sum([(val - mu0)**2 for val in x_hat])
        s2 = sum([(val - mu1)**2 for val in x_hat])
        s3 = sum([math.cos(2 * math.pi * z) for z in z_vec.flat])
        return min(s1, dim + s * s2) + 10.0 * (dim - s3) + 10**4 * Fpen(arr)

class Rastrigin:
    def __init__(self, args = None):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)
        self.args = args
        self.data = get_data(self.args.mode)
        self.DATA_MAXIMIZE = 1

    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)
    
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:
        '''Implementation for BBOB Rastrigin function.'''
        dim = len(arr)
        arr.shape = (dim, 1)
        z = np.matmul(_R(dim, seed, b'R'), arr)
        z = Tasy(ArrayMap(z, Tosz), 0.2)
        z = np.matmul(_R(dim, seed, b'Q'), z)
        z = np.matmul(LambdaAlpha(10.0, dim), z)
        z = np.matmul(_R(dim, seed, b'R'), z)
        return float(10 * (dim - np.sum(np.cos(2 * math.pi * z))) +
                    np.sum(z * z, axis=0))

class RosenbrockRotated:
    def __init__(self, args = None):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)
        self.args = args
        self.data = get_data(self.args.mode)
        self.DATA_MAXIMIZE = 1

    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)  
    
    def __call__(self, arr: np.ndarray, seed: int = 0) -> float:  
        '''Implementation for BBOB RosenbrockRotated function.'''
        dim = len(arr)
        r_x = np.matmul(_R(dim, seed, b'R'), arr)
        z = max(1.0, (dim**0.5) / 8.0) * r_x + 0.5 * np.ones((dim,))
        return float(
            sum([
                100.0 * (z[i]**2 - z[i + 1])**2 + (z[i] - 1)**2
                for i in range(dim - 1)
            ]))

class SharpRidge:
    def __init__(self, args = None):
        self.dim = 10
        self.lb = np.full(self.dim, -5)
        self.ub = np.full(self.dim, 5)
        self.args = args
        self.data = get_data(self.args.mode)
        self.DATA_MAXIMIZE = 1

    def load_similar_source_data(self):
        return get_similar_data(self.args.search_space_id, self.args.dataset_id, self.data, self.DATA_MAXIMIZE)
    
    def load_mixed_source_data(self):
        return get_combined_data(self.args.search_space_id, self.data, self.args.mode, self.DATA_MAXIMIZE, self.dim)

    def __call__(self, arr: np.ndarray, seed: int = 0) -> float: 
        '''Implementation for BBOB SharpRidge function.'''
        dim = len(arr)
        arr.shape = (dim, 1)
        z_vec = np.matmul(_R(dim, seed, b'R'), arr)
        z_vec = np.matmul(LambdaAlpha(10, dim), z_vec)
        z_vec = np.matmul(_R(dim, seed, b'Q'), z_vec)
        return z_vec[0, 0]**2 + 100 * np.sum(z_vec[1:]**2)**0.5
    
if __name__ == '__main__':
    problem = SharpRidge()
    print(problem(np.ones(10)))
