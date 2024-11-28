from typing import Optional, Sequence, Mapping

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz


def to_parameters(count, features: Mapping[str, np.ndarray]):
    # similar to to_parameters of DefaultTrialConvert in vizier/pyvizier/converts/core.py
    parameters = [
        vz.ParameterDict() for _ in range(count)
    ]

    for key, values in features.items():
        for param_dict, value in zip(parameters, values):
            param_dict[key] = value.item()
    return parameters


def random_sample_one(pc, rng, count=1):
    if pc.type == vz.ParameterType.DOUBLE:
        ret = pc.bounds[0] + (pc.bounds[1] - pc.bounds[0]) * rng.rand(count, 1)
    elif pc.type == vz.ParameterType.INTEGER:
        ret = rng.randint(pc.bounds[0], pc.bounds[1]+1, size=(count, 1))
    elif pc.type in [vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE]:
        ret = rng.choice(pc.feasible_values, size=(count, 1))
    else:
        raise NotImplementedError
    return ret


def random_sample(count, search_space, rng):
    sample = dict()
    for pc in search_space.parameters:
        sample[pc.name] = random_sample_one(pc, rng, count)
    return sample


def mutate_operator(trial, search_space, rng):
    sample = dict() # str: np.ndarray with size (count, 1)
    dim = len(search_space.parameters)
    mutate_idx = rng.randint(0, dim)
    for i, pc in enumerate(search_space.parameters):
        if mutate_idx == i:
            sample[pc.name] = random_sample_one(pc, rng)
        else:
            sample[pc.name] = np.array([[trial.parameters[pc.name]]])
    return sample