from typing import Optional, Sequence

import numpy as np
import pandas as pd
from vizier import algorithms as vza
from vizier import pyvizier as vz
from hebo.design_space.design_space import DesignSpace 
from hebo.optimizers.hebo import HEBO

from data_gen_utils import to_parameters, random_sample


class HeBODesigner(vza.PartiallySerializableDesigner):
    def __init__(
        self,
        problem_statement: vz.ProblemStatement,
        **hebo_kwargs,
    ):
        self._problem_statement = problem_statement
        self._metric_name = self._problem_statement.metric_information.item().name
        self._maximize = (
            self._problem_statement.metric_information.item().goal == vz.ObjectiveMetricGoal.MAXIMIZE
        )
        self._search_space = self._problem_statement.search_space

        param_list = []
        for pc in self._search_space.parameters:
            if pc.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
                param = {
                    'name': pc.name,
                    'type': 'num' if pc.type == vz.ParameterType.DOUBLE else 'int',
                    'lb': pc.bounds[0],
                    'ub': pc.bounds[1],
                }
            elif pc.type == vz.ParameterType.CATEGORICAL:
                param = {
                    'name': pc.name,
                    'type': 'cat',
                    'categories': pc.feasible_values,
                }
            else:
                raise NotImplementedError
            param_list.append(param)

        self._hebo_space = DesignSpace().parse(param_list)
        self._opt = HEBO(self._hebo_space, **hebo_kwargs)

    def suggest(self, count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
        rec = self._opt.suggest(n_suggestions=count)
        sample = dict()
        for pc in self._search_space.parameters:
            sample[pc.name] = rec[pc.name].values.reshape(-1, 1)

        return [
            vz.TrialSuggestion(p) for p in to_parameters(count, sample)
        ]

    def update(
        self,
        completed: vza.CompletedTrials,
        all_active: vza.ActiveTrials,
    ) -> None:
        del all_active
        all_trials = [trial for trial in completed.trials]
        all_metrics = [trial.final_measurement.metrics for trial in all_trials]
        all_values = [metrics[self._metric_name].value for metrics in all_metrics]

        names = self._hebo_space.para_names
        rec = {name: [] for name in names}
        for trial in all_trials:
            for name in names:
                rec[name].append(trial.parameters[name].value)
        rec = pd.DataFrame(rec)
        self._opt.observe(rec, np.array(all_values).reshape(-1, 1))

    def dump(self) -> vz.Metadata:
        raise NotImplementedError

    def load(self, metadata: vz.Metadata) -> None:
        raise NotImplementedError
        