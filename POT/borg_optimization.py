from POT.tree import PTree
import numpy as np
import pandas as pd

from RICE_model.IAM_RICE import RICE

from platypus import NSGAII, Problem, Real
from POT.pyborg import BorgMOEA


class PolicyTreeOptimizer:
    def __init__(self, model, action_names,  action_bounds, discrete_actions, feature_names, feature_bounds, discrete_features, epsilon=0.1, max_nfe=1000, max_depth=4, population_size=100):
        self.model = model
        # Tree variables
        self.action_names = action_names
        self.action_bounds = action_bounds
        self.discrete_actions = discrete_actions
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds
        self.discrete_features = discrete_features
        self.max_depth = max_depth
        # Optimization variables
        self.epsilon = epsilon
        self.max_nfe = max_nfe
        self.population_size = population_size

    def random_tree(self, terminal_ratio=0.5,
                    # discrete_actions=True,
                    # discrete_features=None,
                    ):

        num_features = len(self.feature_names)
        num_actions = len(self.action_names) # SD changed

        depth = np.random.randint(1, self.max_depth + 1)
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and \
                                          np.random.rand() < terminal_ratio):
                if self.discrete_actions:
                    L.append([str(np.random.choice(self.action_names))])
                else:
                    # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
                    a = np.random.choice(num_actions)  # SD changed
                    action_name = self.action_names[a]
                    action_value = np.random.uniform(*self.action_bounds[a])
                    action_input = f'{action_name}_{action_value}'
                    # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
                    L.append([action_input])  # SD changed

            else:
                x = np.random.choice(num_features)
                v = np.random.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def borg_problem(self, x):
        T = self.random_tree()
        m1, m2, m3 = self.model.POT_control_Herman(T)
        # print(m1, m2, m3, T)
        return m1, m2, m3

    # def borg_problem(self, x):
    #     T = self.random_tree()
    #     m1, m2 = self.model.POT_control(T)
    #     # print(m1, m2, T)
    #     return m1, m2

    def run(self):
        problem = Problem(1, 3)
        # problem = Problem(1, 2)
        problem.types[:] = Real(-10, 10)
        problem.function = self.borg_problem

        # define and run the Borg algorithm for 10000 evaluations
        algorithm = BorgMOEA(problem, epsilons=self.epsilon, population_size=self.population_size)
        algorithm.run(self.max_nfe)

        results_dict = {}
        # print the results
        for idx, solution in enumerate(algorithm.result):
            # print(solution.variables, solution.objectives)
            results_dict[idx] = str(solution.objectives)

        df = pd.DataFrame.from_dict(results_dict, orient='index')
        return df




# -------------------------------------------------------------------------------------------------------------------

# import numpy as np
# import datetime
# import functools
# import logging
# import time
#
# from .tree import PTree
# from POT.executors import SequentialExecutor
#
#
# logger = logging.getLogger(__name__)
#
#
# def function_runner(func, solution):
#     # model.f has side effects: it changes values on P
#     # so for parallel running  we want to return
#     # also the modified P
#     logger.debug("trying to run {} for {}".format(func, solution))
#     results = func(solution)
#     logger.debug("succesfully ran {} with {}: {}".format(func, solution,
#                                                          results))
#
#     return solution, results
#
#
# class PolicyTreeOptimizer:
#     def __init__(self, f, feature_bounds, discrete_actions=False,
#                  action_bounds=None, action_names=None,
#                  population_size=100, mu=15, max_depth=4, mut_prob=0.9,
#                  cx_prob=0.9, feature_names=None,
#                  discrete_features=None, multiobj=False, epsilons=None):
#
#         self.f = functools.partial(function_runner, f)
#         self.num_features = len(feature_bounds)
#         self.feature_bounds = feature_bounds
#         self.discrete_actions = discrete_actions
#         self.action_bounds = action_bounds
#         self.action_names = action_names
#         self.popsize = population_size
#         self.mu = mu
#         self.max_depth = max_depth
#         self.mut_prob = mut_prob
#         self.cx_prob = cx_prob
#         self.feature_names = feature_names
#         self.discrete_features = discrete_features
#         self.multiobj = multiobj
#         self.epsilons = epsilons
#
#         if feature_names is not None and \
#                 len(feature_names) != len(feature_bounds):
#             raise ValueError(('feature_names and feature_bounds '
#                               'must be the same length.'))
#
#         if discrete_features is not None and \
#                 len(discrete_features) != len(feature_bounds):
#             raise ValueError(('discrete_features and feature_bounds '
#                               'must be the same length.'))
#
#         if discrete_actions:
#             if action_names is None or action_bounds is not None:
#                 raise ValueError(('discrete_actions must be run with '
#                                   'action_names, (which are strings), '
#                                   'and not action_bounds.'))
#         else:
#             if action_bounds is None:
#                 raise ValueError(('Real-valued actions (which is the case by '
#                                   'default, discrete_actions=False) must include '
#                                   'action_bounds. Currently only one action is '
#                                   'supported, so bounds = [lower, upper].'))
#
#         if mu > population_size:
#             raise ValueError(('Number of parents (mu) cannot be greater '
#                               'than the population_size.'))
#
#     def run(self, max_nfe=1000, log_frequency=100, snapshot_frequency=100,
#             executor=SequentialExecutor()):
#         '''Run the optimization algorithm
#
#         Parameters
#         ----------
#         max_nfe : int, optional
#         log_frequency :  int, optional
#         snapshot_frequency : int or None, optional
#                              int specifies frequency of storing convergence
#                              information. If None, no convergence information
#                              is retained.
#         executor : subclass of BaseExecutor, optional
#
#         Returns
#         -------
#         best_p
#             best solution or archive in case of many objective
#         best_f
#             best score(s)
#         snapshots
#             if snapshot_frequency is not None, convergence information
#
#         '''
#
#         start_time = time.time()
#         nfe, last_log, last_snapshot = 0, 0, 0
#
#         self.best_f = None
#         self.best_p = None
#         self.population = np.array([self.random_tree() for _ in
#                                     range(self.popsize)])
#
#         # if snapshot_frequency is not None:
#         #     snapshots = {'nfe': [], 'time': [], 'best_f': [],
#         #                  'best_P': []}
#         # else:
#         #     snapshots = None
#
#         while nfe < max_nfe:
#             for member in self.population:
#                 member.clear_count()  # reset action counts to zero
#
#             # evaluate objectives
#             population, objectives = executor.map(self.f, self.population)
#
#             self.objectives = objectives
#             self.population = np.asarray(population)
#
#             for member in population:
#                 member.normalize_count()  # convert action count to percent
#
#             nfe += self.popsize
#         print(self.population)
#         print(self.objectives)
#
#         #     self.iterate()
#         #
#         #     if nfe >= last_log + log_frequency:
#         #         last_log = nfe
#         #         elapsed = datetime.timedelta(
#         #             seconds=time.time() - start_time).seconds
#         #
#         #         if not self.multiobj:
#         #             logger.info(self.process_log_message.format(nfe,
#         #                                                         elapsed, self.best_f, self.best_p))
#         #         else:
#         #             # TODO:: to be tested
#         #             logger.info('# nfe = %d\n%s\n%s' % (nfe, self.best_f,
#         #                                                 self.best_f.shape))
#         #
#         #     if nfe >= last_snapshot + snapshot_frequency:
#         #         last_snapshot = nfe
#         #         snapshots['nfe'].append(nfe)
#         #         snapshots['time'].append(elapsed)
#         #         snapshots['best_f'].append(self.best_f)
#         #         snapshots['best_P'].append(self.best_p)
#         #
#         # if snapshot_frequency:
#         #     return self.best_p, self.best_f, snapshots
#         # else:
#         #     return self.best_p, self.best_f
#
#     def random_tree(self, terminal_ratio=0.5):
#         '''
#
#         Parameters
#         ----------
#         terminal_ration : float, optional
#
#         '''
#
#         depth = np.random.randint(1, self.max_depth + 1)
#         L = []
#         S = [0]
#
#         while S:
#             current_depth = S.pop()
#
#             # action node
#             if current_depth == depth or (current_depth > 0 and \
#                                           np.random.rand() < terminal_ratio):
#                 if self.discrete_actions:
#                     L.append([str(np.random.choice(self.action_names))])
#                 else:
#                     L.append([np.random.uniform(*self.action_bounds)])
#
#             else:
#                 x = np.random.choice(self.num_features)
#                 v = np.random.uniform(*self.feature_bounds[x])
#                 L.append([x, v])
#                 S += [current_depth + 1] * 2
#
#         T = PTree(L, self.feature_names, self.discrete_features)
#         T.prune()
#         return T
