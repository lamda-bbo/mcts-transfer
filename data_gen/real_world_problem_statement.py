from vizier import pyvizier as vz

from problems.real_world_problem import RealWorldProblem


def problem_statement(search_space_id, dataset_id):
    f = RealWorldProblem(search_space_id, dataset_id)
    dim, lb, ub = f.dim, f.lb, f.ub
    assert (lb == 0).all()
    assert (ub == 1).all()
    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(f.dim):
        root.add_float_param('x{}'.format(i), 0, 1)
    metric = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    )
    problem.metric_information.append(metric)
    return problem, f
