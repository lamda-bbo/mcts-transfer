from typing import Optional, Sequence

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz

from data_gen_utils import to_parameters, random_sample, mutate_operator


class HillClimbingDesigner(vza.PartiallySerializableDesigner):
    def __init__(
        self,
        problem_statement: vz.ProblemStatement,
        *,
        seed: Optional[int] = None,
    ):
        self._problem_statement = problem_statement
        self._metric_name = self._problem_statement.metric_information.item().name
        self._maximize = (
            self._problem_statement.metric_information.item().goal == vz.ObjectiveMetricGoal.MAXIMIZE
        )
        self._search_space = self._problem_statement.search_space
        self._rng = np.random.RandomState(seed)
        self._metadata_ns = 'hill_climbing'
        self._best_trial = None

    def suggest(self, count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
        assert count == 1
        count = count or 1
        if self._best_trial is None:
            sample = random_sample(count, self._search_space, self._rng)
        else:
            sample = mutate_operator(self._best_trial, self._search_space, self._rng)
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

        if self._maximize:
            idx = np.argmax(all_values)
            if self._best_trial is None or \
                self._best_trial.final_measurement.metrics[self._metric_name].value < all_values[idx]:
                self._best_trial = all_trials[idx]
        else:
            idx = np.argmin(all_values)
            if self._best_trial is None or \
                self._best_trial.final_measurement.metrics[self._metric_name].value > all_active[idx]:
                self._best_trial = all_trials[idx]

    def dump(self) -> vz.Metadata:
        metadata = vz.Metadata()
        metadata.ns(self._metadata_ns)['best_trial'] = self._best_trial
        return metadata

    def load(self, metadata: vz.Metadata) -> None:
        metadata = metadata.ns(self._metadata_ns)
        self._best_trial = metadata['best_trial']
