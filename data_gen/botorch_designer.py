from typing import Optional, Sequence

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.algorithms import designers
from vizier._src.algorithms.core.abstractions import ActiveTrials, CompletedTrials
import torch
import botorch
import gpytorch
from gpytorch.kernels import ScaleKernel, MaternKernel
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch import fit_gpytorch_mll


class BotorchDesigner(vza.PartiallySerializableDesigner):
    def __init__(
        self,
        problem_statement: vz.ProblemStatement,
    ):
        self._problem_statement = problem_statement
        self._metric_name = self._problem_statement.metric_information.item().name
        assert self._problem_statement.metric_information.item().goal == \
            vz.ObjectiveMetricGoal.MAXIMIZE
        self._dim = len(self._problem_statement.search_space.parameters)
        self._lb, self._ub = [], []
        for param in self._problem_statement.search_space.parameters:
            self._lb.append(param.bounds[0])
            self._ub.append(param.bounds[1])
        self._lb = torch.tensor(self._lb)
        self._ub = torch.tensor(self._ub)

        # self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = torch.device('cpu')

        self._init_designer = designers.RandomDesigner(self._problem_statement.search_space)
        self._trials = []
        self._X = torch.zeros((0, self._dim))
        self._Y = torch.zeros((0, 1))

    def _preprocess(self):
        train_X = (self._X - self._lb) / (self._ub - self._lb)
        train_X = torch.nan_to_num(train_X)
        train_Y = (self._Y - self._Y.mean()) / (self._Y.std() + 1e-6)
        train_X, train_Y = train_X.to(self._device), train_Y.to(self._device)
        
        return train_X.double(), train_Y.double()

    def _create_model(self, train_X, train_Y):
        covar_module = ScaleKernel(MaternKernel())
        model = SingleTaskGP(train_X, train_Y, covar_module=covar_module)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def _create_acqf(self, model, train_X, train_Y):
        AF = ExpectedImprovement(model, train_Y.max())
        return AF

    def _optimize_acqf(self, AF):
        bounds = torch.vstack((torch.zeros(self._dim), torch.ones(self._dim))).double().to(self._device)
        next_X, _ = botorch.optim.optimize_acqf(AF, bounds=bounds, q=1, num_restarts=3, raw_samples=128)
        return next_X

    def _postprocess(self, next_X):
        next_X = next_X.to('cpu')
        next_X = self._lb + next_X * (self._ub - self._lb)
        return next_X

    def suggest(self, count: Optional[int]=None) -> Sequence[vz.TrialSuggestion]:
        count = count or 1
        assert count == 1
        if len(self._trials) < 10:
            next_trial = self._init_designer.suggest()[0]
        else:
            train_X, train_Y = self._preprocess()
            mll, model = self._create_model(train_X, train_Y)
            fit_gpytorch_mll(mll)
            AF = self._create_acqf(model, train_X, train_Y)
            next_X = self._optimize_acqf(AF)
            next_X = self._postprocess(next_X)

            param_dict = vz.ParameterDict()
            for i, name in enumerate(self._problem_statement.search_space.parameter_names):
                param_dict[name] = next_X[0][i].item()
            next_trial = vz.TrialSuggestion(param_dict)
            
        return [next_trial]

    def update(
        self,
        completed: CompletedTrials,
        all_active: ActiveTrials
    ) -> None:
        del all_active
        all_trails = [trial for trial in completed.trials]
        self._trials.extend(all_trails)
        for trial in all_trails:
            X = torch.zeros(self._dim)
            Y = torch.zeros(1)
            for i, name in enumerate(self._problem_statement.search_space.parameter_names):
                X[i] = trial.parameters.get(name).value
            Y[0] = trial.final_measurement.metrics[self._metric_name].value
            
            self._X = torch.vstack((self._X, X))
            self._Y = torch.vstack((self._Y, Y))

    def dump(self):
        raise NotImplementedError
    
    def load(self, metadata) -> None:
        raise NotImplementedError