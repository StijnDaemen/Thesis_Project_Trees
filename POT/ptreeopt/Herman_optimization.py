from __future__ import division

import copy
import datetime
import functools
import logging
import time

import numpy as np

from .tree import PTree
from POT.ptreeopt.executors import SequentialExecutor

logger = logging.getLogger(__name__)


def function_runner(func, solution):
    # model.f has side effects: it changes values on P
    # so for parallel running  we want to return
    # also the modified P
    logger.debug("trying to run {} for {}".format(func, solution))
    results = func(solution)
    logger.debug("succesfully ran {} with {}: {}".format(func, solution,
                                                         results))
    
    return solution, results

class PTreeOpt(object):
    '''Algorithm for optimizing policy trees
    
    Parameters
    ----------
    f : callable
    feature_bounds : 
    discrete_actions : boolean, optional
    action_bounds : 
    action_names : 
    population size : int, optional
    mu : float, optional
    max_depth : int, optional
    mut_prob : float, optional
    cx_prob : float, optional
    feature_names : 
    discrete_features : 
    multiobj : bool, optional
    epsilons : 
    
    
    Raises
    ------
    ValueError
    
    '''
    
    process_log_message = ('{} nfe; {} sec; '
                '{}; {}')

    def __init__(self, f, feature_bounds, discrete_actions=False,
                 action_bounds=None, action_names=None,
                 population_size=100, mu=15, max_depth=4, mut_prob=0.9,
                 cx_prob=0.9, feature_names=None,
                 discrete_features=None, multiobj=False, epsilons=None):

        self.f = functools.partial(function_runner, f)
        self.num_features = len(feature_bounds)
        self.feature_bounds = feature_bounds
        self.discrete_actions = discrete_actions
        self.action_bounds = action_bounds
        self.action_names = action_names
        self.popsize = population_size
        self.mu = mu
        self.max_depth = max_depth
        self.mut_prob = mut_prob
        self.cx_prob = cx_prob
        self.feature_names = feature_names
        self.discrete_features = discrete_features
        self.multiobj = multiobj
        self.epsilons = epsilons

        if feature_names is not None and\
           len(feature_names) != len(feature_bounds):
            raise ValueError(('feature_names and feature_bounds '
                              'must be the same length.'))

        if discrete_features is not None and\
           len(discrete_features) != len(feature_bounds):
            raise ValueError(('discrete_features and feature_bounds '
                              'must be the same length.'))

        if discrete_actions:
            if action_names is None or action_bounds is not None:
                raise ValueError(('discrete_actions must be run with '
                                  'action_names, (which are strings), '
                                  'and not action_bounds.'))
        else:
            if action_bounds is None:
                raise ValueError(('Real-valued actions (which is the case by '
                                'default, discrete_actions=False) must include '
                                'action_bounds. Currently only one action is '
                                'supported, so bounds = [lower, upper].'))

        if mu > population_size:
            raise ValueError(('Number of parents (mu) cannot be greater '
                              'than the population_size.'))

    def iterate(self):
        # TODO:: have to separate selection
        # one selection function for multi objective
        # one selection function for single objective

        # selection: find index numbers for parents
        if not self.multiobj:

            parents = self.select_truncation(self.objectives)

            if self.best_f is None or self.objectives[parents[0]] < self.best_f:
                self.best_f = self.objectives[parents[0]]
                self.best_p = copy.deepcopy(self.population[parents[0]])

        else:
            # print(f"self.population: {self.population}")
            # print(f"self.objectives: {self.objectives}")
            parents = [self.binary_tournament(self.population, self.objectives)
                       for _ in range(self.mu)]
            # print(f"parents: {parents}")
            # print(f"best_f: {self.best_f}")
            # print(f"best_p: {self.best_p}")

            if self.best_f is None:
                self.best_f = self.objectives[parents]
                self.best_p = self.population[parents]
            else:
                self.best_p, self.best_f = self.archive_sort(self.best_p,
                                                 self.best_f, self.population, 
                                                 self.objectives)
        # print(f"parents: {parents}")
        # print(f"best_f: {self.best_f}")
        # print(f"best_p: {self.best_p}")

        # generate child solutions. strategy:
        # best parent remains unchanged (elitist)
        # children generated by crossover+mutation with probability cx_prob
        # random new children with probability (1-cx_prob)
        children = set(range(self.popsize)) - set(parents)

        for i in parents[1:]:
            child = self.mutate(self.population[i])
            child.prune()
            self.population[i] = child

        # then crossover to develop the rest of the population
        for i in children:

            if np.random.rand() < self.cx_prob:
                P1, P2 = np.random.choice(
                    self.population[parents], 2, replace=False)
                child = self.crossover(P1, P2)[0]

                # bloat control
                while child.get_depth() > self.max_depth:
                    child = self.crossover(P1, P2)[0]

            else:  # replace with randomly generated child
                child = self.random_tree()
                # child = np.random.choice(self.population[parents], 1)[0]

            child = self.mutate(child)
            child.prune()
            self.population[i] = child

    def run(self, max_nfe=1000, log_frequency=100, snapshot_frequency=100,
            executor=SequentialExecutor()):
        '''Run the optimization algorithm
        
        Parameters
        ----------
        max_nfe : int, optional
        log_frequency :  int, optional
        snapshot_frequency : int or None, optional
                             int specifies frequency of storing convergence 
                             information. If None, no convergence information 
                             is retained.
        executor : subclass of BaseExecutor, optional
        
        Returns
        -------
        best_p
            best solution or archive in case of many objective
        best_f
            best score(s)
        snapshots
            if snapshot_frequency is not None, convergence information
        
        '''

        start_time = time.time()
        nfe, last_log, last_snapshot = 0, 0, 0

        self.best_f = None
        self.best_p = None
        self.population = np.array([self.random_tree() for _ in
                                    range(self.popsize)])

        if snapshot_frequency is not None:
            snapshots = {'nfe': [], 'time': [], 'best_f': [],
                         'best_P': []}
        else:
            snapshots = None

        while nfe < max_nfe:
            for member in self.population:
                member.clear_count() # reset action counts to zero

            # evaluate objectives            
            population, objectives = executor.map(self.f, self.population)
            
            self.objectives = objectives
            self.population = np.asarray(population)

            for member in population:
                member.normalize_count() # convert action count to percent

            nfe += self.popsize

            self.iterate()

            if nfe >= last_log + log_frequency:
                last_log = nfe
                elapsed = datetime.timedelta(
                    seconds=time.time() - start_time).seconds

                if not self.multiobj:
                    logger.info(self.process_log_message.format(nfe, 
                                    elapsed, self.best_f, self.best_p))
                else:
                    # TODO:: to be tested
                    logger.info('# nfe = %d\n%s\n%s' % (nfe, self.best_f,
                                                    self.best_f.shape))
                    
            if nfe >= last_snapshot + snapshot_frequency:
                last_snapshot = nfe
                snapshots['nfe'].append(nfe)
                snapshots['time'].append(elapsed)
                snapshots['best_f'].append(self.best_f)
                snapshots['best_P'].append(self.best_p)

        if snapshot_frequency:
            return self.best_p, self.best_f, snapshots
        else:
            return self.best_p, self.best_f

    def random_tree(self, terminal_ratio=0.5):
        '''
        
        Parameters
        ----------
        terminal_ration : float, optional
        
        '''
        
        depth = np.random.randint(1, self.max_depth + 1)
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and\
                                      np.random.rand() < terminal_ratio):
                if self.discrete_actions:
                    L.append([str(np.random.choice(self.action_names))])
                else:
                    # L.append([np.random.uniform(*self.action_bounds)])
                    # The commented line above does not seem to work so I (SD) changed it
                    a = np.random.choice(len(self.action_names))  # SD changed
                    action_name = self.action_names[a]
                    action_value = np.random.uniform(*self.action_bounds[a])
                    action_input = f'{action_name}_{action_value}'
                    L.append([action_input])  # SD changed

            else:
                x = np.random.choice(self.num_features)
                v = np.random.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def select_truncation(self, obj):
        return np.argsort(obj)[:self.mu]

    def crossover(self, P1, P2):
        P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = np.random.choice(feature_ix1)
        index2 = np.random.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    def mutate(self, P, mutate_actions=True):
        P = copy.deepcopy(P)

        for item in P.L:
            if np.random.rand() < self.mut_prob:
                if item.is_feature:
                    low, high = self.feature_bounds[item.index]
                    if item.is_discrete:
                        item.threshold = np.random.randint(low, high+1)
                    else:
                        item.threshold = self.bounded_gaussian(
                            item.threshold, [low, high])
                elif mutate_actions:
                    if self.discrete_actions:
                        item.value = str(np.random.choice(self.action_names))
                    else:
                        # item.value = self.bounded_gaussian(
                        #     item.value, self.action_bounds)
                        # The commented line above does not seem to work so I (SD) changed it
                        a = np.random.choice(len(self.action_names))  # SD changed
                        action_name = self.action_names[a]  # SD changed
                        action_value = np.random.uniform(*self.action_bounds[a])  # SD changed
                        action_input = f'{action_name}_{action_value}'  # SD changed
                        item.value = action_input  # SD changed

        return P

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + np.random.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def dominates(self, a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        return (np.all(a <= b) and np.any(a < b))

    def same_box(self, a, b):
        if self.epsilons:
            a = a // self.epsilons
            b = b // self.epsilons
        return np.all(a == b)

    def binary_tournament(self, P, f):
        # select 1 parent from population P
        # (Luke Algorithm 99 p.138)
        i = np.random.randint(0, P.shape[0], 2)
        # print(f'P.shape[0]: {P.shape[0]}')
        # print(f'f: {f}')
        # print(f'i: {i}')
        a, b = f[i[0]], f[i[1]]
        # print(a, b, self.dominates(a, b))
        # print(np.all(a <= b))
        # print(np.any(a < b))
        if self.dominates(a, b):
            return i[0]
        elif self.dominates(b, a):
            return i[1]
        else:
            return i[0] if np.random.rand() < 0.5 else i[1]

    def archive_sort(self, A, fA, P, fP):
        # print(f'A before: {A}')
        # print(f'P : {P}')
        A = np.hstack((A, P))
        # print(f'A after: {A}')

        # print(f'fA before: {fA}')
        # print(f'fP : {fP}')
        fA = np.vstack((fA, fP))
        # print(f'fA after: {fA}')
        N = len(A)
        keep = np.ones(N, dtype=bool)
        # print(f'keep before: {keep}')
        for i in range(N):
            for j in range(i + 1, N):
                if keep[j] and self.dominates(fA[i, :], fA[j, :]):
                    keep[j] = False

                elif keep[i] and self.dominates(fA[j, :], fA[i, :]):
                    keep[i] = False

                elif self.same_box(fA[i, :], fA[j, :]):
                    keep[np.random.choice([i, j])] = False
        # print(f'keep after: {keep}')
        # print(f'A[keep]: {A[keep]}')
        return (A[keep], fA[keep, :])
