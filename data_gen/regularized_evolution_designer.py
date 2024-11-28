from typing import Optional, Sequence
from collections import deque

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz

from data_gen_utils import to_parameters, random_sample, mutate_operator


class RegularizedEvolutionDesigner(vza.PartiallySerializableDesigner):
    def __init__(
        self,
        problem_statement: vz.ProblemStatement,
        *,
        population_size: int = 25,
        tournament_size: int = 5,
        seed: Optional[int] = None,
    ):
        self._problem_statement = problem_statement
        self._population_size = population_size
        self._population = deque()
        self._tournament_size = tournament_size
        self._metric_name = self._problem_statement.metric_information.item().name
        self._maximize = (
            self._problem_statement.metric_information.item().goal == vz.ObjectiveMetricGoal.MAXIMIZE
        )
        self._search_space = self._problem_statement.search_space
        self._rng = np.random.RandomState(seed)
        self._metadata_ns = 'regularized_evolution'

    def suggest(self, count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
        assert count == 1
        count = count or 1
        if len(self._population) < self._population_size:
            sample = random_sample(count, self._search_space, self._rng)
        else:
            tournament_idx = np.random.choice(self._population_size, size=self._tournament_size, replace=False)
            tournament = [self._population[i] for i in tournament_idx]
            tournament_values = [i.final_measurement.metrics[self._metric_name].value for i in tournament]
            if self._maximize:
                idx = np.argmax(tournament_values)
            else:
                idx = np.argmin(tournament_values)
            parent = tournament[idx]
            sample = mutate_operator(parent, self._search_space, self._rng)

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
        self._population.extend(all_trials)
        while len(self._population) > self._population_size:
            self._population.popleft()

    def dump(self) -> vz.Metadata:
        raise NotImplementedError

    def load(self, metadata: vz.Metadata) -> None:
        raise NotImplementedError
        