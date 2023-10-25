import numpy as np
from POT.tree import PTree
from folsom import Folsom
# from RICE_model.IAM_RICE import RICE
# from folsom import Folsom
import time
import copy
import pandas as pd
import math

import matplotlib.pyplot as plt

from statistics import mean

import re

'''
The design behind this MOEA is driven by the effectiveness of randomness for our specific multi-objective model.
Very crafty approaches like borg did not work as well as relatively straightforward approaches like Herman's POT
Also, the 'pareto front' from the random search was broader than the other algorithms.

This MOEA is built to better understand randomness.
Random solutions are generated and the extreme and non-dominated ones are picked as parents. 
These parents then recombine to produce offspring that is half the size of the other parent population
The other half is filled with randomly generated trees.
This new generation is checked again for extremeness and dominance 
The ratio of crossover/random generated solutions is now used as the ratio to which new offspring is produced (instead of 50/50)

Expectations: AT first the random solutions will work well but as the pareto front is approached, the recombined solutions will work better 
'''


class Shotgun:
    def __init__(self,
                 model,
                 master_rng,
                 metrics,
                 action_names,
                 action_bounds,
                 feature_names,
                 feature_bounds,
                 max_depth,
                 discrete_actions,
                 discrete_features,
                 mutation_prob,
                 pop_size,
                 max_nfe,
                 epsilons,
                 gamma=4,
                 tau=0.02,
                 save_location=None,
                 title_of_run=None,
                 ):

        self.rng_tree = np.random.default_rng(master_rng.integers(0, 1e9))

        self.model = model
        self.metrics = metrics
        self.action_names = action_names
        self.action_bounds = action_bounds
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds
        self.max_depth = max_depth
        self.discrete_actions = discrete_actions
        self.discrete_features = discrete_features
        self.mutation_prob = mutation_prob
        self.pop_size = pop_size
        self.max_nfe = max_nfe
        self.epsilons = epsilons

        self.epsilon_progress_counter = 0
        self.epsilon_progress_tracker = np.array([])
        self.snapshot_dict = {'nfe': [],
                              'time': [],
                              'Archive_solutions': [],
                              'Archive_trees': []}

        self.action_bounds = [[1,10], [2,10], [3,10], [4,10], [5,10], [6,10], [7,10]]



        self.nfe = 0

    def run(self):

        action_input = ''
        for idx, action in enumerate(self.action_names):
            action_value = round(self.rng_tree.uniform(*self.action_bounds[idx]), 3)
            action_input = action_input + f'{action}_{action_value}|'
        # Remove last '|'
        action_input = action_input[:-1]
        print(action_input)
        print('-------------------------------------')

        nr_actions = 2
        for _ in range(nr_actions):
            # randomly pick an action and a new value from its action bounds
            num_actions = len(self.action_names)
            x = self.rng_tree.choice(num_actions)
            action_name = self.action_names[x]
            action_value_new = round(self.rng_tree.uniform(*self.action_bounds[x]), 3)
            # wrap it in a string, consistent with later processing
            action_substring = f'{action_name}_{action_value_new}|'
            # Replace the substring in the large string
            pattern = re.escape(action_name) + r".*?\|"
            action_input = re.sub(pattern, action_substring, action_input)

        print(action_input)


        print('---------------------------------------')

        # actions = self.rng_mutation_subtree.choice(self.action_names, nr_actions, replace=False)
        # for action_name in actions:
        #     if action_name == 'miu':
        #         action_value = self.rng_mutation_subtree.integers(*self.action_bounds[0])
        #         item.value = GAOperators.replace_miu_substr(self, item.value, f'miu_{action_value}|')
        #     elif action_name == 'sr':
        #         action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[1]), 3)
        #         item.value = GAOperators.replace_sr_substr(self, item.value, f'sr_{action_value}|')
        #     elif action_name == 'irstp':
        #         action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[2]), 3)
        #         item.value = GAOperators.replace_irstp_substr(self, item.value, f'irstp_{action_value}')

        action_input = ''
        for idx, action in enumerate(self.action_names):
            action_value = round(self.rng_tree.uniform(*self.action_bounds[idx]), 3)
            action_input = action_input+f'{action}_{action_value}|'
        # Remove last '|'
        action_input = action_input[:-1]
        print(action_input)
        print('-------------------------------------')

        # Generate initial random population
        self.population = np.array([self.spawn(heritage='random') for _ in range(self.pop_size)])
        print(f'size pop: {np.size(self.population)}')

        # Add epsilon non-dominated and extreme solutions to VIPs
        self.VIPs = np.array([self.population[0]])
        for sol in self.population:
            self.enter_VIPs(sol)
        print(f'size VIPs: {np.size(self.VIPs)}')
        return

    def random_tree(self, terminal_ratio=0.5):
        num_features = len(self.feature_names)

        depth = self.rng_tree.integers(1, self.max_depth + 1)
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and
                                          self.rng_tree.random() < terminal_ratio):
                if self.discrete_actions:
                    L.append([str(self.rng_tree.choice(self.action_names))])
                else:
                    action_input = f'miu_{self.rng_tree.integers(*self.action_bounds[0])}|sr_{round(self.rng_tree.uniform(*self.action_bounds[1]), 3)}|irstp_{round(self.rng_tree.uniform(*self.action_bounds[2]), 3)}'
                    L.append([action_input])
            else:
                x = self.rng_tree.choice(num_features)
                v = self.rng_tree.uniform(*self.feature_bounds[x])
                L.append([x, v])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        return T

    def spawn(self, heritage):
        organism = Organism()
        organism.dna = self.random_tree()
        organism.fitness = self.policy_tree_model_fitness(organism.dna)
        organism.heritage = heritage
        return organism

    def policy_tree_model_fitness(self, T):
        metrics = np.array(self.model.f(T))
        return metrics

    def dominates(self, a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
        # So to circumvent this but still allow multiobjective optimisation in different directions under the
        # constraint that every number is positive, just add a large number to every index.

        large_number = 1000000000
        a = a + large_number
        b = b + large_number

        return np.all(a <= b) and np.any(a < b)

    def enter_VIPs(self, candidate_solution):
        epsilon_progress = False
        for member in self.VIPs:
            if self.dominates(candidate_solution.fitness, member.fitness - self.epsilons):
                # self.Archive.remove(member)
                self.VIPs = self.VIPs[~np.isin(self.VIPs, member)]
                epsilon_progress = True
            elif self.dominates(member.fitness - self.epsilons, candidate_solution.fitness):
                # Check if they fall in the same box, if so, keep purely dominant solution
                if self.dominates(candidate_solution.fitness, member.fitness):
                    # self.Archive.remove(member)
                    self.VIPS = self.VIPs[~np.isin(self.VIPs, member)]
                elif self.dominates(member.fitness, candidate_solution.fitness):
                    return
                # return
        self.VIPs = np.append(self.VIPs, candidate_solution)
        if epsilon_progress:
            self.epsilon_progress_counter += 1
        return


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None
        self.heritage = None