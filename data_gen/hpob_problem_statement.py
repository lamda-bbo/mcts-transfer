from vizier import pyvizier as vz

from problems.hpob_problem import HPOBProblem


def problem_statement(search_space_id, dataset_id, root_dir):
    f = HPOBProblem(search_space_id, dataset_id, root_dir)
    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(f.dim):
        root.add_float_param('x{}'.format(i), 0.0, 1.0)
    metric = vz.MetricInformation(
        name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    )
    problem.metric_information.append(metric)
    return problem, f
