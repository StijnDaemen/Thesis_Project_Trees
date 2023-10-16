import numpy as np
from POT.tree import PTree
from RICE_model.IAM_RICE import RICE
import copy
import pandas as pd
import math
import time
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from statistics import mean

import networkx as nx
import matplotlib.patches as patches

from POT.tree import Action
from POT.tree import Feature

from collections import Counter


class ForestBorg:
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
                 tau=0.02):

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

        self.model = RICE(years_10, regions)

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

        # self.non_dominated = []
        # self.pareto_front = []
        # THE epsilon dominance archive which is to ensure convergence and diversity.
        self.Archive = np.zeros(self.pop_size)
        self.epsilon_progress_counter = 0
        self.epsilon_progress_tracker = np.zeros(self.max_nfe)

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


        # organism = self.spawn()
        # print(organism.dna)
        # print(organism.dna.root)
        #
        # MOEAVisualizations.visualize_tree(self, organism.dna.root)
        # organism.dna = GAOperators.mutation_subtree(self, organism.dna, 2)
        # MOEAVisualizations.visualize_tree(self, organism.dna.root)

        # MOEAVisualizations.visualize_tree(self, organism.dna.root)
        # organism.dna = self.mutation_point(organism.dna)
        # MOEAVisualizations.visualize_tree(self, organism.dna.root)
        #
        # organism.dna = self.mutate(organism.dna)
        # # self.mutate(organism.dna)
        #
        # MOEAVisualizations.visualize_tree(self, organism.dna.root)

        # MOEAVisualizations.draw_tree()

        # -- Start algorithm ----------------------------------
        # Generate the initial population
        self.population = np.array([self.spawn() for _ in range(self.pop_size)])
        for elem in self.population:
            print(elem.fitness)
        print(f'size pop: {np.size(self.population)}')

        # Add the epsilon non-dominated solutions from the population to the Archive (initialize the Archive with the initial population, running add_to_Archive() function ensures no duplicates will be present.
        self.Archive = np.array([self.population[0]])
        for sol in self.population:
            self.add_to_Archive(sol)
        print(f'size Archive: {np.size(self.Archive)}')
        # for elem in self.Archive:
        #     print(elem.fitness)
        #
        # # Check which solutions are non-dominated
        # self.non_dominated = self.natural_selection(self.population)
        # # Add to the Archive
        # self.Archive = self.non_dominated
        # print(len(self.Archive))
        # for elem in self.Archive:
        #     print(elem.fitness)
        #
        # # Check which solutions are epsilon non-dominated and add them to the Archive
        # self.Archive = self.natural_selection_epsilon(self.population)
        # print(len(self.Archive))

        # -- main loop -----------------------------------------

        nfe = 0
        log_counter = 0
        while self.nfe < self.max_nfe:
            self.iterate(nfe)
            nfe += 1
            log_counter += 1
            if log_counter%10 == 0:
                intermediate_time = time.time()
                print(f'\rnfe: {self.nfe}/{self.max_nfe} -- epsilon convergence: {self.epsilon_progress_counter} -- elapsed time: {(intermediate_time - self.start_time)/60} min -- number of restarts: {self.number_of_restarts}', end='', flush=True)

        # -- Create visualizations of the run -------------------
        self.end_time = time.time()
        print(f'Total elapsed time: {(self.end_time-self.start_time)/60} min -- {len(self.Archive)} non-dominated solutions were found.')

        MOEAVisualizations.visualize_generational_series(self, self.epsilon_progress_tracker, x_label='generation', y_label='epsilon-progress', save=False)

        Archive_in_objective_space = []
        for member in self.Archive:
            Archive_in_objective_space.append(member.fitness)
        print(Archive_in_objective_space)
        MOEAVisualizations.visualize_organisms_objective_space(self, Archive_in_objective_space, title=None,
                                                               x_label='welfare', y_label='damages', z_label='temp. overshoots', save=False)

        # Visualize operator distribution
        MOEAVisualizations.visualize_operator_distribution(self, self.GAOperators, title='operator distribution', x_label='Generation', y_label='Count', save=False)

    def iterate(self, i):
        if i%100 == 0:
            # Check gamma (the population to Archive ratio)
            gamma = len(self.population) / len(self.Archive)
            print(f'gamma: {gamma}')

            # Trigger restart if the latest epsilon tracker value is not different from the previous 3 -> 4ht iteration without progress.
            # Officially in the borg paper I believe it is triggered if the latest epsilon tracker value is the same as the one of that before
            # but that is a bit extreme for this problem as, especially in the first few iterations the Archive is not updated.

            if self.check_unchanged(self.epsilon_progress_tracker):
            # if self.epsilon_progress_tracker[-1] == (
            #         self.epsilon_progress_tracker[-2] + self.epsilon_progress_tracker[-3] +
            #         self.epsilon_progress_tracker[-4]) / 3:
                self.revive_search(gamma)
            # Doing this causes many (too many?) restarts
            # Check if gamma value warrants a restart (see Figure 2 in paper borg)
            elif (gamma > 1.25 * self.gamma) or (gamma < 0.75 * self.gamma):
                self.revive_search(gamma)
        # if len(self.epsilon_progress_tracker) > 50:
        #     if gamma < 0.75 * self.gamma:
        #         self.revive_search(gamma)

        # Selection of recombination operator
        parents_required = 2
        parents = []
        # One parent is uniformly randomly selected from the archive
        parents.append(self.rng_iterate.choice(self.Archive))
        # The other parent(s) are selected from the population using tournament selection
        for parent in range(parents_required-1):
            # k = self.determine_tournament_size(len(self.population) / len(self.Archive))
            parents.append(self.tournament(self.tournament_size))

        # # Create the offspring
        # offspring = Organism()
        # offspring.dna = self.crossover(parents[0].dna, parents[1].dna)[0]
        # # Let it mutate (in-built chance of mutation)
        # offspring.dna = self.mutate(offspring.dna)
        # offspring.fitness = self.policy_tree_RICE_fitness(offspring.dna)
        # offspring.operator = ''

        # Create the offspring
        offspring = Organism()
        offspring.dna = GAOperators.crossover_subtree(self, parents[0].dna, parents[1].dna)[0]
        offspring = self.mutate_with_feedbackloop(offspring)
        offspring.fitness = self.policy_tree_RICE_fitness(offspring.dna)
        self.nfe += 1

        # Add to population
        self.add_to_population(offspring)

        # Add to Archive if eligible
        self.add_to_Archive(offspring)

        # Update the epsilon progress tracker
        # self.epsilon_progress_tracker.append(self.epsilon_progress_counter)
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

    # def determine_tournament_size(self, gamma):
    #     return max(2, int(self.tau * gamma * len(self.Archive)))

    def revive_search(self, gamma):
        print(f'population_size before: {len(self.population)}')
        print(f'archive_size before: {len(self.Archive)}')
        # Empty the population
        self.population = []
        # Fill it with all solutions from the Archive
        self.population = self.Archive
        # Compute size of new population
        new_size = self.gamma*len(self.Archive)
        # Inject mutated Archive members into the population
        while len(self.population) < new_size:
            # Select a random solution from the Archive
            volunteer = self.rng_revive.choice(self.Archive)
            # Mutate new solution
            # volunteer.dna = self.mutate(volunteer.dna)
            volunteer = self.mutate_with_feedbackloop(volunteer)
            volunteer.fitness = self.policy_tree_RICE_fitness(volunteer.dna)
            self.nfe += 1
            # Add new solution to population
            # self.population.append(volunteer)
            self.add_to_population(volunteer)
            # Update Archive with new solution
            self.add_to_Archive(volunteer)
        print(f'population_size after: {len(self.population)}')
        print(f'archive_size after: {len(self.Archive)}')
        self.number_of_restarts += 1
        return

    def restart(self, current_Archive, gamma, tau):
        self.population = []
        population = current_Archive
        new_size = gamma*len(current_Archive)
        # Inject mutated Archive members into the new population
        while len(population) < new_size:
            # Select a random solution from the Archive
            volunteer = self.rng_revive.choice(current_Archive)
            volunteer = self.mutate_with_feedbackloop(volunteer)
            volunteer.fitness = self.policy_tree_RICE_fitness(volunteer.dna)
            self.nfe += 1
            # Add new solution to population
            self.population.append(volunteer)
            # self.add_to_population(volunteer)
            # Update Archive with new solution
            self.add_to_Archive(volunteer)
        # Adjust tournament size to account for the new population size
        self.tournament_size = max(2, math.floor(tau*new_size))
        return population

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
        # print(f'a new tree is made')
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
                    # a = np.random.choice(num_actions)  # SD changed
                    # action_name = self.action_names[a]
                    # action_value = np.random.uniform(*self.action_bounds[a])
                    # action_input = f'{action_name}_{action_value}'
                    # # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
                    # L.append([action_input])  # SD changed

                    # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
                    # number_of_actions_on_leaf_node = np.random.randint(1, num_actions+1)
                    # actions = np.random.choice(num_actions, number_of_actions_on_leaf_node, replace=False)
                    # collect_actions = []
                    # for a in actions:
                    #     action_name = self.action_names[a]
                    #     action_value = np.random.uniform(*self.action_bounds[a])
                    #     action_input = f'{action_name}_{action_value}'
                    #     collect_actions.append(action_input)
                    #     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
                    #     # L.append([action_input])  # SD changed
                    # L.append(['|'.join(collect_actions)])  # SD changed

                    # # Now all actions are part of a leaf
                    # collect_actions = []
                    # for a in range(num_actions):
                    #     action_name = self.action_names[a]
                    #     action_value = np.random.uniform(*self.action_bounds[a])
                    #     action_input = f'{action_name}_{action_value}'
                    #     collect_actions.append(action_input)
                    #     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
                    #     # L.append([action_input])  # SD changed
                    # L.append(['|'.join(collect_actions)])  # SD changed
                    # # print(['|'.join(collect_actions)])

                    action_input = f'miu_{self.rng_tree.integers(*self.action_bounds[0])}|sr_{round(self.rng_tree.uniform(*self.action_bounds[1]), 3)}|irstp_{round(self.rng_tree.uniform(*self.action_bounds[2]), 3)}'
                    L.append([action_input])
            else:
                x = self.rng_tree.choice(num_features)
                v = self.rng_tree.uniform(*self.feature_bounds[x])
                L.append([x, v])
                # print(self.feature_names[x])
                S += [current_depth + 1] * 2

        T = PTree(L, self.feature_names, self.discrete_features)
        T.prune()
        # print(self.rng.integers(10, size=10))
        return T

    def policy_tree_RICE_fitness(self, T):
        metrics = np.array(self.model.POT_control(T))
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

    def mutate(self, P, mutate_actions=True):
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
                        # print(item)
                        # print(self.action_bounds)
                        # print(item.value)
                        # item.value = self.bounded_gaussian(
                        #     item.value, self.action_bounds)

                        # --------
                        # a = np.random.choice(len(self.action_names))  # SD changed
                        # action_name = self.action_names[a]
                        # action_value = np.random.uniform(*self.action_bounds[a])
                        # action_input = f'{action_name}_{action_value}'
                        # # print(action_input)
                        # item.value = action_input

                        # action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
                        action_input = f'miu_{self.rng_mutate.integers(*self.action_bounds[0])}|sr_{round(self.rng_mutate.uniform(*self.action_bounds[1]),3 )}|irstp_{round(self.rng_mutate.uniform(*self.action_bounds[2]), 3)}'
                        item.value = action_input

        return P

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

    # def natural_selection_epsilon(self, list_obj):
    #     A = np.array(list_obj)
    #     N = len(list_obj)
    #     keep = np.ones(N, dtype=bool)
    #
    #     for i in range(N):
    #         for j in range(i + 1, N):
    #             if keep[j] and self.epsilon_dominated(A[i], A[j]):
    #                 keep[j] = False
    #
    #             elif keep[i] and self.epsilon_dominated(A[j], A[i]):
    #                 keep[i] = False
    #
    #             # elif self.same_box(np.array(A[i].fitness), np.array(A[j].fitness)):
    #             #     keep[self.rng_natural_selection.choice([i, j])] = False
    #
    #     return list(A[keep])

    def add_to_pareto_front(self):
        for i, member_candidate in enumerate(self.non_dominated):
            for j, member_established in enumerate(self.pareto_front):
                if self.dominates(member_candidate.fitness, member_established.fitness):
                    self.pareto_front.pop(j)
                    self.pareto_front.append(member_candidate)

        # Remove duplicates from pareto_front if present:
        self.pareto_front = list(set(self.pareto_front))
        return

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

    # def epsilon_dominated(self, a, b):
    #     # Outputs True if a is epsilon non-dominated by b, False otherwise
    #     # In this new implementation maximization is assumed -> NEVERMIND
    #
    #     large_number = 1000000000
    #
    #     a = np.array(a)
    #     a = a + large_number
    #     b = np.array(b)
    #     b = b + large_number
    #     # print(f'a: {a} -/- b-epsilons: {b-self.epsilons}')
    #     # answer = np.all(a <= (b - self.epsilons)) and np.any(a < (b - self.epsilons))
    #     # print(answer)
    #     return np.all(a <= (b + self.epsilons)) and np.any(a < (b + self.epsilons))
    #     # return np.all(a <= b) and np.any(a < b)

    def epsilon_dominated(self, organism1, organism2):
        # Outputs true if organism1 epsilon dominates organism2
        dominates = False
        i = 0
        for a, b in zip(organism1.fitness, organism2.fitness):
            if a > b + self.epsilons[i]:
                return False
            elif a < b - self.epsilons[i]:
                dominates = True
            i += 1
        return dominates

    def add_to_Archive(self, candidate_solution):
        # epsilon_progress = False
        # for member in self.Archive:
        #     if self.epsilon_dominated(member, candidate_solution):
        #         # if the candidate solution is dominated, dont allow it into the Archive
        #         if self.dominates(member.fitness, candidate_solution.fitness):
        #             # Check if the member and candidate_solution fall in the same epsilon box, if so, if the member is dominated continue but the epsilon progress stays False
        #             return
        #         elif self.dominates(candidate_solution.fitness, member.fitness):
        #             self.Archive.remove(member)
        #             self.Archive.append(candidate_solution)
        #         # else:
        #             # print(f'candidate_solution: {candidate_solution.fitness}')
        #             # print(f'member: {member.fitness}')
        #     elif self.epsilon_dominated(candidate_solution, member):
        #         # If a member is dominated by the candidate solution, remove the member from the Archive
        #         self.Archive.remove(member)
        #         # print(f'candidate_solution: {candidate_solution.fitness}')
        #         # print(f'member: {member.fitness}')
        #         epsilon_progress = True
        # self.Archive.append(candidate_solution)
        # if epsilon_progress:
        #     self.epsilon_progress_counter += 1
        # return

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
                return
        self.Archive = np.append(self.Archive, candidate_solution)
        if epsilon_progress:
            self.epsilon_progress_counter += 1
        return

    # def add_to_Archive(self, candidate_solution):
    #     epsilon_progress = True
    #     for idx, member in enumerate(self.Archive):
    #         if self.epsilon_dominated(candidate_solution.fitness, member.fitness):
    #             self.Archive.pop(idx)
    #         elif self.epsilon_dominated(member.fitness, candidate_solution.fitness):
    #             if self.dominates(member.fitness, candidate_solution.fitness):
    #                 return False
    #             else:
    #                 epsilon_progress = False
    #     self.Archive.append(candidate_solution)
    #     if epsilon_progress:
    #         self.epsilon_progress_counter += 1
    #     return True

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
        pass


class GAOperators(ForestBorg):
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

    # def crossover_homologous(self, P1, P2):
    #
    #     return P1, P2

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
                    # print(item)
                    # print(self.action_bounds)
                    # print(item.value)
                    # item.value = self.bounded_gaussian(
                    #     item.value, self.action_bounds)

                    # --------
                    # a = np.random.choice(len(self.action_names))  # SD changed
                    # action_name = self.action_names[a]
                    # action_value = np.random.uniform(*self.action_bounds[a])
                    # action_input = f'{action_name}_{action_value}'
                    # # print(action_input)
                    # item.value = action_input

                    # # action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
                    # action_input = f'miu_{self.rng_mutation_subtree.integers(*self.action_bounds[0])}|sr_{round(self.rng_mutation_subtree.uniform(*self.action_bounds[1]), 3)}|irstp_{round(self.rng_mutation_subtree.uniform(*self.action_bounds[2]), 3)}'
                    # node.value = action_input

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
                # print(item)
                # print(self.action_bounds)
                # print(item.value)
                # item.value = self.bounded_gaussian(
                #     item.value, self.action_bounds)

                # --------
                # a = np.random.choice(len(self.action_names))  # SD changed
                # action_name = self.action_names[a]
                # action_value = np.random.uniform(*self.action_bounds[a])
                # action_input = f'{action_name}_{action_value}'
                # # print(action_input)
                # item.value = action_input

                # # action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
                # action_input = f'miu_{self.rng_mutate.integers(*self.action_bounds[0])}|sr_{round(self.rng_mutate.uniform(*self.action_bounds[1]), 3)}|irstp_{round(self.rng_mutate.uniform(*self.action_bounds[2]), 3)}'
                # item.value = action_input

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
                        # print(item)
                        # print(self.action_bounds)
                        # print(item.value)
                        # item.value = self.bounded_gaussian(
                        #     item.value, self.action_bounds)

                        # --------
                        # a = np.random.choice(len(self.action_names))  # SD changed
                        # action_name = self.action_names[a]
                        # action_value = np.random.uniform(*self.action_bounds[a])
                        # action_input = f'{action_name}_{action_value}'
                        # # print(action_input)
                        # item.value = action_input

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
    def __init__(self):
        pass

    def visualize_generational_series(self, series, title=None, x_label=None, y_label=None, save=False):
        x = [x for x in range(len(series))]
        y = series
        plt.scatter(x, y)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if save:
            plt.savefig(f'{title}.png', bbox_inches='tight')
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
            # ax.set_xlim([-43000, -42000])
            # ax.set_ylim([-10000000, -1000000])
            # ax.set_zlim([2, 6])
            ax.view_init(elev=30.0, azim=15)

            if save:
                plt.savefig(f'{title}.png', bbox_inches='tight')
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

        # def draw_tree(root):
        #     graph, pos = nx.DiGraph(), {}
        #     graph, pos = add_edges(graph, root, pos)
        #
        #     fig, ax = plt.subplots(figsize=(8, 6))
        #
        #     # Draw edges
        #     nx.draw(graph, pos, with_labels=False, arrows=False, ax=ax, node_size=0)
        #
        #     # Draw rectangle nodes
        #     for node, (x, y) in pos.items():
        #         rectangle = patches.Rectangle((x - 0.1, y - 0.1), 0.2, 0.2, edgecolor='blue', facecolor='skyblue')
        #         ax.add_patch(rectangle)
        #
        #         # Add node labels
        #         plt.text(x, y, str(node), ha='center', va='center', fontsize=12)
        #
        #     ax.autoscale()
        #     plt.axis('off')
        #     plt.show()

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
            plt.savefig(f'{title}.png', bbox_inches='tight')
        else:
            plt.show()
        # Make sure to close the plt object once done
        plt.close()

        # -- The commented out code below works only for small dictionaries, not for large time/generational series-----
        #
        # # Data where keys are intended to be colors (segments) and each list represents a column in the bar chart.
        # data = distribution_dict
        #
        # fig, ax = plt.subplots()
        #
        # # Labels for each column
        # labels = [f'{i}' for i in range(len(next(iter(data.values()))))]
        #
        # # Ensure consistent data lengths
        # assert all(len(v) == len(next(iter(data.values()))) for v in data.values()), "Mismatched data lengths"
        #
        # # Initialize cumulative size variable
        # cumulative_size = [0] * len(labels)
        #
        # # Loop through data and create stacked bars
        # for color, segment_data in data.items():
        #     ax.bar(labels, segment_data, label=color, bottom=cumulative_size)
        #
        #     # Update the cumulative size
        #     cumulative_size = [cum_size + seg_data for cum_size, seg_data in zip(cumulative_size, segment_data)]
        #
        # # Format and display the plot
        # ax.set_ylabel(y_label)
        # ax.set_xlabel(x_label)
        # ax.set_title(title)
        # ax.legend()
        #
        # plt.tight_layout()
        # if save:
        #     plt.savefig(f'{title}.png', bbox_inches='tight')
        # else:
        #     plt.show()
        # # Make sure to close the plt object once done
        # plt.close()
        pass


if __name__ == '__main__':
    years_10 = []
    for i in range(2005, 2315, 10):
        years_10.append(i)

    regions = [
        "US",
        "OECD-Europe",
        "Japan",
        "Russia",
        "Non-Russia Eurasia",
        "China",
        "India",
        "Middle East",
        "Africa",
        "Latin America",
        "OHI",
        "Other non-OECD Asia",
    ]
    master_rng = np.random.default_rng(42)  # Master RNG
    ForestBorg(pop_size=100, master_rng=master_rng,
                  years_10=years_10,
                  regions=regions,
                  metrics=['period_utility', 'damages', 'temp_overshoots'],
                  # Tree variables
                  action_names=['miu', 'sr', 'irstp'],
                  action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
                  feature_names=['mat', 'net_output', 'year'],
                  feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                  max_depth=4,
                  discrete_actions=False,
                  discrete_features=False,
                  # Optimization variables
                  mutation_prob=0.5,
                  max_nfe=20000,
                  epsilons=np.array([0.05, 0.05, 0.05]),
                  gamma=4,
                  tau=0.02,
                  )
