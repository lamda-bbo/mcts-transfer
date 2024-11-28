from vizier import pyvizier as vz

from problems.metabo_synthetic import MetaBOSyntheticTorch


def problem_statement(search_space_id, dataset_id):
    if search_space_id == 'Branin2':
        dim = 2
    elif search_space_id == 'Hartmann3':
        dim = 3
    else:
        raise ValueError('Unsupported search space: {}'.format(search_space_id))

    f = MetaBOSyntheticTorch(search_space_id, dataset_id)
    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(dim):
        root.add_float_param('x{}'.format(i), 0, 1)
    metric = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    )
    problem.metric_information.append(metric)
    return problem, f

