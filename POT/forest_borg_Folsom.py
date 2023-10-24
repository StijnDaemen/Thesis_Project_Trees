import numpy as np
from POT.tree import PTree
from RICE_model.IAM_RICE import RICE
from folsom import Folsom
import copy
import pandas as pd
import math
import time
import re

import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.patches as patches

from collections import Counter
import sqlite3

import pickle


class ForestBorgFolsom:
    def __init__(self, pop_size, master_rng,
                 years_10,
                 regions,
                 metrics,
                 action_names,
                 action_bounds,
                 feature_names,
                 feature_bounds,
                 max_depth,
                 discrete_actions,
                 discrete_features,
                 mutation_prob,
                 max_nfe,
                 epsilons,
                 gamma=4,
                 tau=0.02,
                 save_location=None,
                 title_of_run=None,):

        self.pop_size = pop_size

        self.rng_init = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_populate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_natural_selection = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_tree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_crossover = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_gauss = np.random.default_rng(master_rng.integers(0, 1e9))

        self.rng_iterate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_tournament = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_population = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_revive = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_crossover_subtree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutation_point = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutation_subtree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_choose_operator = np.random.default_rng(master_rng.integers(0, 1e9))

        self.model = Folsom('folsom/data/folsom-daily-w2016.csv',
                       sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)

        self.metrics = metrics
        self.action_names = action_names
        self.action_bounds = action_bounds
        self.feature_names = feature_names
        self.feature_bounds = feature_bounds
        self.max_depth = max_depth
        self.discrete_actions = discrete_actions
        self.discrete_features = discrete_features
        self.mutation_prob = mutation_prob
        self.max_nfe = max_nfe
        self.epsilons = epsilons
        self.gamma = gamma
        self.tau = tau
        self.tournament_size = 2

        self.epsilon_progress_counter = 0
        self.epsilon_progress_tracker = np.array([])
        self.snapshot_dict = {'nfe': [],
                              'time': [],
                              'Archive_solutions': [],
                              'Archive_trees': []}

        self.number_of_restarts = 0

        self.GAOperators = {'mutation_point_1': [0],
                            'mutation_point_2': [0],
                            'mutation_point_3': [0],
                            'mutation_subtree_1': [0],
                            'mutation_subtree_2': [0],
                            'mutation_subtree_3': [0],
                            'mutation_random': [0],
                            }

        self.nfe = 0

        self.start_time = time.time()

        self.save_location = save_location
        self.file_name = title_of_run
        # Initialize sqlite database to save outcomes of a run - note use one universal database called 'Experiments.db'
        if save_location:
            self.database = f'{save_location}/Experiments.db'
        else:
            self.database = None

    def run(self, snapshot_frequency=100):
        self.population = np.array([self.spawn() for _ in range(self.pop_size)])
        print(f'size pop: {np.size(self.population)}')

        # Add the epsilon non-dominated solutions from the population to the Archive (initialize the Archive with the initial population, running add_to_Archive() function ensures no duplicates will be present.
        self.Archive = np.array([self.population[0]])
        for sol in self.population:
            self.add_to_Archive(sol)
        print(f'size Archive: {np.size(self.Archive)}')

        # -- main loop -----------------------------------------
        last_snapshot = 0
        main_loop_counter = 0
        # log_counter = 0
        while self.nfe < self.max_nfe:
            self.iterate(main_loop_counter)
            main_loop_counter += 1

            if self.nfe >= last_snapshot + snapshot_frequency:
                last_snapshot = self.nfe
                self.snapshot_dict['nfe'].append(self.nfe)
                self.snapshot_dict['time'].append((time.time() - self.start_time) / 60)
                self.snapshot_dict['Archive_solutions'].append([item.fitness for item in self.Archive])
                self.snapshot_dict['Archive_trees'].append([str(item.dna) for item in self.Archive])

                intermediate_time = time.time()
                print(
                    f'\rnfe: {self.nfe}/{self.max_nfe} -- epsilon convergence: {self.epsilon_progress_counter} -- elapsed time: {(intermediate_time - self.start_time) / 60} min -- number of restarts: {self.number_of_restarts}',
                    end='', flush=True)

            # log_counter += 1
            # if log_counter % 50 == 0:
            #     # # Archive snapshots
            #     # data_dict = {}
            #     # for idx, item in enumerate(self.Archive):
            #     #     data_dict[f'{self.nfe}_{idx}_snapshot'] = [item.fitness[0], item.fitness[1], item.fitness[2], item.fitness[3],
            #     #                                              str(item.dna)]
            #     # df = pd.DataFrame.from_dict(data_dict, orient='index')
            #     #
            #     # pickle.dump(self.Archive, open(f'{self.save_location}/{self.file_name}_Archive_snapshots.pkl', 'wb'))
            #     #
            #     # conn = sqlite3.connect(self.database)
            #     # df.to_sql(name=f'archive_snapshots_{self.file_name}', con=conn, if_exists='append')
            #     # conn.commit()
            #     # conn.close()
            #     self.snapshot_dict['nfe'].append(self.nfe)
            #     self.snapshot_dict['time'].append((time.time() - self.start_time) / 60)
            #     self.snapshot_dict['Archive_solutions'].append([item.fitness for item in self.Archive])
            #     self.snapshot_dict['Archive_trees'].append([str(item.dna) for item in self.Archive])
            #
            # if log_counter % 50 == 0:
            #     intermediate_time = time.time()
            #     print(
            #         f'\rnfe: {self.nfe}/{self.max_nfe} -- epsilon convergence: {self.epsilon_progress_counter} -- elapsed time: {(intermediate_time - self.start_time) / 60} min -- number of restarts: {self.number_of_restarts}',
            #         end='', flush=True)

        # -- Create visualizations of the run -------------------
        self.end_time = time.time()

        data_dict = {}
        for item in self.Archive:
            data_dict[item.dna] = [item.fitness[0], item.fitness[1], item.fitness[2]]
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        # df.to_excel('500000nfe_0_05epsilon_Archive_solutions.xlsx')

        print(
            f'Total elapsed time: {(self.end_time - self.start_time) / 60} min -- {len(self.Archive)} non-dominated solutions were found.')

        MOEAVisualizations(self.save_location).visualize_generational_series(self.epsilon_progress_tracker,
                                                         title=f'epsilon_progress_{self.file_name}',
                                                         x_label='generation', y_label='epsilon-progress', save=True)
        #
        # Archive_in_objective_space = []
        # for member in self.Archive:
        #     Archive_in_objective_space.append(member.fitness)
        # # print(Archive_in_objective_space)
        # MOEAVisualizations(self.save_location).visualize_organisms_objective_space(Archive_in_objective_space,
        #                                                        title=f'3d_{self.file_name}',
        #                                                        x_label='welfare', y_label='damages',
        #                                                        z_label='temp. overshoots', save=True)

        # Visualize operator distribution
        MOEAVisualizations(self.save_location).visualize_operator_distribution(self.GAOperators,
                                                           title=f'operator_distribution_{self.file_name}',
                                                           x_label='Generation', y_label='Count', save=True)
        return df, self.snapshot_dict

    def iterate(self, i):
        if i%500 == 0:
            # Check gamma (the population to Archive ratio)
            gamma = len(self.population) / len(self.Archive)

            # Trigger restart if the latest epsilon tracker value is not different from the previous 3 -> 4ht iteration without progress.
            # Officially in the borg paper I believe it is triggered if the latest epsilon tracker value is the same as the one of that before

            if self.check_unchanged(self.epsilon_progress_tracker):
                self.restart(self.Archive, self.gamma, self.tau)
            # Check if gamma value warrants a restart (see Figure 2 in paper borg)
            elif (gamma > 1.25 * self.gamma) or (gamma < 0.75 * self.gamma):
                # self.revive_search(gamma)
                self.restart(self.Archive, self.gamma, self.tau)

        # Selection of recombination operator
        parents_required = 2
        parents = []
        # One parent is uniformly randomly selected from the archive
        parents.append(self.rng_iterate.choice(self.Archive))
        # The other parent(s) are selected from the population using tournament selection
        for parent in range(parents_required-1):
            parents.append(self.tournament(self.tournament_size))

        # Create the offspring
        offspring = Organism()
        offspring.dna = GAOperators.crossover_subtree(self, parents[0].dna, parents[1].dna)[0]
        # Let it mutate (in-built chance of mutation)
        offspring = self.mutate_with_feedbackloop(offspring)
        offspring.fitness = self.policy_tree_RICE_fitness(offspring.dna)
        self.nfe += 1

        # Add to population
        self.add_to_population(offspring)

        # Add to Archive if eligible
        self.add_to_Archive(offspring)

        # Update the epsilon progress tracker
        self.epsilon_progress_tracker = np.append(self.epsilon_progress_tracker, self.epsilon_progress_counter)

        # Record GA operator distribution
        self.record_GAOperator_distribution()
        return

    def record_GAOperator_distribution(self):
        # Count the occurrences of each attribute value
        distribution = Counter(member.operator for member in self.Archive)
        for key in self.GAOperators.keys():
            if key in distribution:
                self.GAOperators[key].append(distribution[key])
            else:
                self.GAOperators[key].append(0)
        return

    def check_unchanged(self, lst):
        if len(lst) > 3:
            for i in range(3, len(lst)):
                if lst[i - 3] == lst[i - 2] == lst[i - 1]:
                    return True
        return False

    def mutate_with_feedbackloop(self, offspring):
        # TODO:: This is super hacky and bad programming, must change action handling and operator selection after proof-of-concept
        # Mutation based on performance feedback loop
        operator = GAOperators.choose_mutation_operator(self)
        if operator == 'mutation_point_1':
            offspring.dna = GAOperators.mutation_point(self, offspring.dna, 1)
            offspring.operator = operator
        elif operator == 'mutation_point_2':
            offspring.dna = GAOperators.mutation_point(self, offspring.dna, 2)
            offspring.operator = operator
        elif operator == 'mutation_point_3':
            offspring.dna = GAOperators.mutation_point(self, offspring.dna, 3)
            offspring.operator = operator
        elif operator == 'mutation_subtree_1':
            offspring.dna = GAOperators.mutation_subtree(self, offspring.dna, 1)
            offspring.operator = operator
        elif operator == 'mutation_subtree_2':
            offspring.dna = GAOperators.mutation_subtree(self, offspring.dna, 2)
            offspring.operator = operator
        elif operator == 'mutation_subtree_3':
            offspring.dna = GAOperators.mutation_subtree(self, offspring.dna, 3)
            offspring.operator = operator
        elif operator == 'mutation_random':
            offspring.dna = GAOperators.mutation_random(self, offspring.dna)
            offspring.operator = operator
        return offspring

    def restart(self, current_Archive, gamma, tau):
        self.population = np.array([])
        self.population = current_Archive
        new_size = gamma * len(current_Archive)
        # Inject mutated Archive members into the new population
        while len(self.population) < new_size:
            # Select a random solution from the Archive
            volunteer = self.rng_revive.choice(current_Archive)
            volunteer = self.mutate_with_feedbackloop(volunteer)
            volunteer.fitness = self.policy_tree_RICE_fitness(volunteer.dna)
            # # Now try with completely new solutions as that seemed kind of promising in trials
            # volunteer = self.spawn()
            self.nfe += 1
            # Add new solution to population
            if self.population.size > 0:
                self.add_to_population(volunteer)
            else:
                self.population = np.append(self.population, volunteer)
            # Update Archive with new solution
            self.add_to_Archive(volunteer)
        # Adjust tournament size to account for the new population size
        self.tournament_size = max(2, math.floor(tau * new_size))
        self.number_of_restarts += 1
        return

    def tournament(self, k):
        # Choose k random members in the population
        members = self.rng_tournament.choice(self.population, k)
        # Winner is defined by pareto dominance.
        # If there are no winners, take a random member, if there are, take a random winner.
        winners = []
        for idx in range(len(members)-1):
            if self.dominates(members[idx].fitness, members[idx+1].fitness):
                winners.append(members[idx])
            elif self.dominates(members[idx+1].fitness, members[idx].fitness):
                winners.append(members[idx+1])

        if not winners:
            return self.rng_tournament.choice(members, 1)[0]
        else:
            return self.rng_tournament.choice(winners, 1)[0]

    def spawn(self):
        organism = Organism()
        organism.dna = self.random_tree()
        organism.fitness = self.policy_tree_RICE_fitness(organism.dna)
        return organism

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

    def policy_tree_RICE_fitness(self, T):
        metrics = np.array(self.model.f(T))
        return metrics

    def crossover(self, P1, P2):
        P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = self.rng_crossover.choice(feature_ix1)
        index2 = self.rng_crossover.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    # def mutate(self, P, mutate_actions=True):
        # P = copy.deepcopy(P)
        #
        # for item in P.L:
        #     if self.rng_mutate.random() < self.mutation_prob:
        #         if item.is_feature:
        #             low, high = self.feature_bounds[item.index]
        #             if item.is_discrete:
        #                 item.threshold = self.rng_mutate.integers(low, high + 1)
        #             else:
        #                 item.threshold = self.bounded_gaussian(
        #                     item.threshold, [low, high])
        #         elif mutate_actions:
        #             if self.discrete_actions:
        #                 item.value = str(self.rng_mutate.choice(self.action_names))
        #             else:
        #                 # print(item)
        #                 # print(self.action_bounds)
        #                 # print(item.value)
        #                 # item.value = self.bounded_gaussian(
        #                 #     item.value, self.action_bounds)
        #
        #                 # --------
        #                 # a = np.random.choice(len(self.action_names))  # SD changed
        #                 # action_name = self.action_names[a]
        #                 # action_value = np.random.uniform(*self.action_bounds[a])
        #                 # action_input = f'{action_name}_{action_value}'
        #                 # # print(action_input)
        #                 # item.value = action_input
        #
        #                 # action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
        #                 action_input = f'miu_{self.rng_mutate.integers(*self.action_bounds[0])}|sr_{round(self.rng_mutate.uniform(*self.action_bounds[1]),3 )}|irstp_{round(self.rng_mutate.uniform(*self.action_bounds[2]), 3)}'
        #                 item.value = action_input
        #
        # return P

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + self.rng_gauss.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def natural_selection(self, A):
        N = len(A)
        keep = np.ones(N, dtype=bool)

        for i in range(N):
            for j in range(i + 1, N):
                if keep[j] and self.dominates(A[i].fitness-self.epsilons, A[j].fitness):
                    keep[j] = False

                elif keep[i] and self.dominates(A[j].fitness-self.epsilons, A[i].fitness):
                    keep[i] = False

                elif self.same_box(A[i].fitness, A[j].fitness):
                    keep[self.rng_natural_selection.choice([i, j])] = False

        return A[keep]

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

    def same_box(self, a, b):
        if np.any(self.epsilons):
            a = a // self.epsilons
            b = b // self.epsilons
        return np.all(a == b)

    def add_to_Archive(self, candidate_solution):
        epsilon_progress = False
        for member in self.Archive:
            if self.dominates(candidate_solution.fitness, member.fitness - self.epsilons):
                # self.Archive.remove(member)
                self.Archive = self.Archive[~np.isin(self.Archive, member)]
                epsilon_progress = True
            elif self.dominates(member.fitness - self.epsilons, candidate_solution.fitness):
                # Check if they fall in the same box, if so, keep purely dominant solution
                if self.dominates(candidate_solution.fitness, member.fitness):
                    # self.Archive.remove(member)
                    self.Archive = self.Archive[~np.isin(self.Archive, member)]
                elif self.dominates(member.fitness, candidate_solution.fitness):
                    return
                # return
        self.Archive = np.append(self.Archive, candidate_solution)
        if epsilon_progress:
            self.epsilon_progress_counter += 1
        return

    def add_to_population(self, offspring):
        # If the offspring dominates one or more population members, the offspring replaces
        # one of these dominated members randomly.

        #  If the offspring is dominated by at least one population member, the offspring
        #  is not added to the population.

        # Otherwise, the offspring is nondominated and replaces a randomly selected member
        # of the population.

        members_to_be_randomly_rejected = []
        for idx, member in enumerate(self.population):
            if self.dominates(offspring.fitness, member.fitness):
                members_to_be_randomly_rejected.append(idx)
            elif self.dominates(member.fitness, offspring.fitness):
                return

        if members_to_be_randomly_rejected:
            # self.population.pop(self.rng_population.choice(members_to_be_randomly_rejected))
            # self.population.append(offspring)
            self.population = self.population[~np.isin(self.population, self.rng_population.choice(members_to_be_randomly_rejected))]
            self.population = np.append(self.population, offspring)
        else:
            kick_out = self.rng_population.choice(len(self.population))
            # self.population.pop(kick_out)
            # self.population.append(offspring)
            self.population = self.population[~np.isin(self.population, kick_out)]
            self.population = np.append(self.population, offspring)
        return


class GAOperators(ForestBorgFolsom):
    # TODO:: actions are hardcoded in, plus putting them in a string like this is very hacky. Generalize and redesign action handling (rn I just need it to work and I had better ask for help from someone who is better at programming than me)
    def replace_miu_substr(self, input_str, replacement):
        # The pattern will find a string that starts with "miu_" and ends with "|".
        pattern = re.compile(r'miu_.*?\|')
        # re.sub replaces the matched pattern with the specified replacement string.
        return re.sub(pattern, replacement, input_str)

    def replace_sr_substr(self, input_str, replacement):
        # The pattern will find a string that starts with "miu_" and ends with "|".
        pattern = re.compile(r'sr_.*?\|')
        # re.sub replaces the matched pattern with the specified replacement string.
        return re.sub(pattern, replacement, input_str)

    def replace_irstp_substr(self, input_str, replacement):
        # The pattern will find a string that starts with "miu_" and ends with "|".
        pattern = re.compile(r'irstp_.*')
        # re.sub replaces the matched pattern with the specified replacement string.
        return re.sub(pattern, replacement, input_str)

    def choose_mutation_operator(self, zeta=1):
        operators = ['mutation_point_1',
                     'mutation_point_2',
                     'mutation_point_3',
                     'mutation_subtree_1',
                     'mutation_subtree_2',
                     'mutation_subtree_3',
                     'mutation_random']
        # Initially give every operator an equal chance, then feedback loop based on occurance in self.Archive
        operator_dict = {}
        for operator in operators:
            num_solutions_operator = 0
            for member in self.Archive:
                if member.operator == operator:
                    num_solutions_operator += 1
            operator_dict[operator] = num_solutions_operator+zeta

        probability_dict = {}
        for operator in operator_dict.keys():
            #     resultset = np.array([value for key, value in operator_dict.items() if key not in operator]).sum()
            resultset = np.array([value for key, value in operator_dict.items()]).sum()
            probability = operator_dict[operator] / (resultset)
            probability_dict[operator] = probability

        return self.rng_choose_operator.choice(list(probability_dict.keys()), p=list(probability_dict.values()))

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + self.rng_gauss.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def crossover_subtree(self, P1, P2):
        P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
        # should use indices of ONLY feature nodes
        feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
        feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
        index1 = self.rng_crossover_subtree.choice(feature_ix1)
        index2 = self.rng_crossover_subtree.choice(feature_ix2)
        slice1 = P1.get_subtree(index1)
        slice2 = P2.get_subtree(index2)
        P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
        P1.build()
        P2.build()
        return (P1, P2)

    def mutation_subtree(self, T, nr_actions):
        T = copy.deepcopy(T)
        # should use indices of ONLY feature nodes
        feature_ix = [i for i in range(T.N) if T.L[i].is_feature]
        index = self.rng_mutation_subtree.choice(feature_ix)
        slice = T.get_subtree(index)
        for node in T.L[slice]:
            if node.is_feature:  # if isinstance(node, Feature):
                low, high = self.feature_bounds[node.index]
                if node.is_discrete:
                    node.threshold = self.rng_mutation_subtree.integers(low, high + 1)
                else:
                    node.threshold = self.bounded_gaussian(
                        node.threshold, [low, high])
            else:
                if self.discrete_actions:
                    node.value = str(self.rng_mutation_subtree.choice(self.action_names))
                else:
                    actions = self.rng_mutation_subtree.choice(self.action_names, nr_actions, replace=False)
                    for action_name in actions:
                        if action_name == 'miu':
                            action_value = self.rng_mutation_subtree.integers(*self.action_bounds[0])
                            node.value = GAOperators.replace_miu_substr(self, node.value, f'miu_{action_value}|')
                        elif action_name == 'sr':
                            action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[1]), 3)
                            node.value = GAOperators.replace_sr_substr(self, node.value, f'sr_{action_value}|')
                        elif action_name == 'irstp':
                            action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[2]), 3)
                            node.value = GAOperators.replace_irstp_substr(self, node.value, f'irstp_{action_value}')
        return T

    def mutation_point(self, T, nr_actions):
        # Point mutation at either feature or action node
        T = copy.deepcopy(T)
        item = self.rng_mutation_point.choice(T.L)
        if item.is_feature:
            low, high = self.feature_bounds[item.index]
            if item.is_discrete:
                item.threshold = self.rng_mutate.integers(low, high + 1)
            else:
                item.threshold = self.bounded_gaussian(
                    item.threshold, [low, high])
        else:
            if self.discrete_actions:
                item.value = str(self.rng_mutate.choice(self.action_names))
            else:
                actions = self.rng_mutation_subtree.choice(self.action_names, nr_actions, replace=False)
                for action_name in actions:
                    if action_name == 'miu':
                        action_value = self.rng_mutation_subtree.integers(*self.action_bounds[0])
                        item.value = GAOperators.replace_miu_substr(self, item.value, f'miu_{action_value}|')
                    elif action_name == 'sr':
                        action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[1]), 3)
                        item.value = GAOperators.replace_sr_substr(self, item.value, f'sr_{action_value}|')
                    elif action_name == 'irstp':
                        action_value = round(self.rng_mutation_subtree.uniform(*self.action_bounds[2]), 3)
                        item.value = GAOperators.replace_irstp_substr(self, item.value, f'irstp_{action_value}')
        return T

    def mutation_random(self, P, mutate_actions=True):
        P = copy.deepcopy(P)

        for item in P.L:
            if self.rng_mutate.random() < self.mutation_prob:
                if item.is_feature:
                    low, high = self.feature_bounds[item.index]
                    if item.is_discrete:
                        item.threshold = self.rng_mutate.integers(low, high + 1)
                    else:
                        item.threshold = self.bounded_gaussian(
                            item.threshold, [low, high])
                elif mutate_actions:
                    if self.discrete_actions:
                        item.value = str(self.rng_mutate.choice(self.action_names))
                    else:
                        # action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
                        action_input = f'miu_{self.rng_mutate.integers(*self.action_bounds[0])}|sr_{round(self.rng_mutate.uniform(*self.action_bounds[1]),3 )}|irstp_{round(self.rng_mutate.uniform(*self.action_bounds[2]), 3)}'
                        item.value = action_input

        return P


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None
        self.operator = None


class MOEAVisualizations:
    def __init__(self, save_location):
        self.save_location = save_location
        pass

    def visualize_generational_series(self, series, title=None, x_label=None, y_label=None, save=False):
        x = [x for x in range(len(series))]
        y = series
        plt.scatter(x, y)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if save:
            plt.savefig(f'{self.save_location}/{title}.png', bbox_inches='tight')
        else:
            plt.show()
        # Make sure to close the plt object once done
        plt.close()

    def visualize_organisms_objective_space(self, organisms, title=None, x_label=None, y_label=None, z_label=None, save=False):
        '''

        :param gorganisms: a list of objective space values, for multiple objectives, a list of lists. maximum 3d
        :return: plot of supplied organisms in objective space
        '''

        # The data is supplied as a list of lists in which an inner list is a solution/organism which contains the values for each objective
        # This data structure must be turned into separate list that each contain the objective values of one objective
        data_dict = {}
        for objective in range(1, len(organisms[0])+1):
            data_dict[f'ofv{objective}'] = []

        for idx, item in enumerate(organisms):
            for key_idx, key in enumerate(data_dict.keys()):
                data_dict[key].append(item[key_idx])

        if len(data_dict.keys()) == 2:
            plt.scatter(data_dict['ofv1'], data_dict['ofv2'])
            plt.title(title)
            plt.ylabel(y_label)
            plt.xlabel(x_label)

            if save:
                plt.savefig(f'{title}.png', bbox_inches='tight')
            else:
                plt.show()
            # Make sure to close the plt object once done
            plt.close()
        elif len(data_dict.keys()) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data_dict['ofv1'],
                       data_dict['ofv2'],
                       data_dict['ofv3'])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.view_init(elev=30.0, azim=15)

            if save:
                plt.savefig(f'{self.save_location}/{title}.png', bbox_inches='tight')
            else:
                plt.show()
            # Make sure to close the plt object once done
            plt.close()
        else:
            print("You must supply a list of lists that is either 2d or 3d, solutions in a higher dimensional space cannot be plotted with this function.")
        pass

    def visualize_tree(self, root):
        def add_edges(graph, node, pos, x=0, y=0, layer=1, width=15):
            pos[str(node)] = (x, y)
            if node.l:
                graph.add_edge(str(node), str(node.l))
                le = x - width #/ 2 ** layer
                add_edges(graph, node.l, pos, x=le, y=y - 1, layer=layer + 1)
            if node.r:
                graph.add_edge(str(node), str(node.r))
                ri = x + width #/ 2 ** layer
                add_edges(graph, node.r, pos, x=ri, y=y - 1, layer=layer + 1)
            return (graph, pos)

        def draw_tree(root):
            graph, pos = nx.DiGraph(), {}
            graph, pos = add_edges(graph, root, pos)

            print(pos)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Draw edges
            nx.draw(graph, pos, with_labels=False, arrows=False, ax=ax, node_size=0)

            for node, (x, y) in pos.items():
                # Create a text instance
                text = plt.text(x, y, str(node), ha='center', va='center', fontsize=12)

                # Get the text's bounding box
                bbox = text.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.1, 1.2)
                width = bbox.width / fig.dpi  # Convert bounding box width to inches
                height = bbox.height / fig.dpi  # Convert bounding box height to inches

                # Check if the node is a leaf node
                if graph.out_degree(node) == 0:  # Leaf node
                    rectangle_color = 'lightgreen'
                else:  # Non-leaf node
                    rectangle_color = 'skyblue'

                # Draw a rectangle around the text
                rectangle = patches.Rectangle((x - width / 2, y - height / 2), width*2, height, edgecolor='blue',
                                              facecolor=rectangle_color)
                ax.add_patch(rectangle)

            ax.autoscale()
            plt.axis('off')
            plt.show()

        def preorder_traversal(root):
            if root:
                # if isinstance(root, Action):
                #     print(root.value, end=" ")  # Visit root
                # elif isinstance(root, Feature):
                #     print(root.name, end=" ")  # Visit root
                preorder_traversal(root.l)  # Visit left subtree
                preorder_traversal(root.r)  # Visit right subtree
            return root

        root = preorder_traversal(root)
        draw_tree(root)

    def visualize_operator_distribution(self, distribution_dict, title=None, x_label=None, y_label=None, save=False):
        index_rng = [x for x in range(len(list(distribution_dict.values())[0]))]
        data = distribution_dict

        df = pd.DataFrame(data, index=index_rng)

        fig, ax = plt.subplots()

        # Stacked area chart
        ax.stackplot(df.index, df.T, labels=df.columns, alpha=0.5)

        # Additional chart formatting
        ax.set_ylabel(y_label)
        ax.set_ylabel(x_label)
        ax.set_title(title)
        ax.legend(loc='upper left')

        plt.tight_layout()
        if save:
            plt.savefig(f'{self.save_location}/{title}.png', bbox_inches='tight')
        else:
            plt.show()
        # Make sure to close the plt object once done
        plt.close()
        pass


# if __name__ == '__main__':
#     years_10 = []
#     for i in range(2005, 2315, 10):
#         years_10.append(i)
#
#     regions = [
#         "US",
#         "OECD-Europe",
#         "Japan",
#         "Russia",
#         "Non-Russia Eurasia",
#         "China",
#         "India",
#         "Middle East",
#         "Africa",
#         "Latin America",
#         "OHI",
#         "Other non-OECD Asia",
#     ]
#     master_rng = np.random.default_rng(42)  # Master RNG
#     ForestBorg(pop_size=100, master_rng=master_rng,
#                   years_10=years_10,
#                   regions=regions,
#                   metrics=['period_utility', 'damages', 'temp_overshoots'],
#                   # Tree variables
#                   action_names=['miu', 'sr', 'irstp'],
#                   action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                   feature_names=['mat', 'net_output', 'year'],
#                   feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                   max_depth=4,
#                   discrete_actions=False,
#                   discrete_features=False,
#                   # Optimization variables
#                   mutation_prob=0.5,
#                   max_nfe=2000,
#                   epsilons=np.array([0.05, 0.05, 0.05]),
#                   gamma=4,
#                   tau=0.02,
#                   ).run()
