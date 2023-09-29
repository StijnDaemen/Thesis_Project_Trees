import numpy as np
from POT.tree import PTree
from RICE_model.IAM_RICE import RICE
import time
import copy
import pandas as pd
import math

import matplotlib.pyplot as plt

from statistics import mean

master_rng = np.random.default_rng(42)  # Master RNG

# # Child RNGs for unconditional and conditional logic
# unconditional_rng = np.random.default_rng(master_rng.integers(0, 1e9))
# conditional_rng = np.random.default_rng(master_rng.integers(0, 1e9))


class Cluster:
    def __init__(self, num_parents, num_children, master_rng=None):
        start = time.time()
        self.graveyard = {}
        self.VIPs = {}
        self.non_dominated = []
        self.pareto_front = []
        self.holding_hands = []
        self.parents = []
        self.children = []
        self.family = []
        self.num_parents = num_parents
        self.num_children = num_children

        self.nfe = 0
        self.generation = 0

        # if unconditional_rng is None:
        #     unconditional_rng = np.random.default_rng()
        # self.unconditional_rng = unconditional_rng  # np.random.default_rng(seed=seed)
        # if conditional_rng is None:
        #     conditional_rng = np.random.default_rng()
        # self.conditional_rng = conditional_rng  # np.random.default_rng(seed=seed)
        # self.rng = np.random.default_rng(seed=44)

        if master_rng is None:
            master_rng = np.random.default_rng(42)

        self.rng_init = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_populate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_natural_selection = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_tree = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_crossover = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_mutate = np.random.default_rng(master_rng.integers(0, 1e9))
        self.rng_gauss = np.random.default_rng(master_rng.integers(0, 1e9))

        self.convergence = []

        # The center position takes the average dist
        self.center_position = []

        # Model variables
        self.years_10 = []
        for i in range(2005, 2315, 10):
            self.years_10.append(i)

        self.regions = [
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
        self.metrics = ['period_utility', 'utility', 'temp_overshoots']
        self.model = RICE(self.years_10, self.regions)
        # Tree variables
        # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
        self.action_names = ['miu', 'sr', 'irstp']
        self.action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
        self.feature_names = ['mat', 'net_output', 'year']
        self.feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
        self.max_depth = 4
        self.discrete_actions = False
        self.discrete_features = False
        # Optimization variables
        self.mutation_prob = 0.5
        self.max_nfe = 30000
        self.epsilons = np.array([0.05, 0.05])

        # Start optimization
        self.iterate()
        # Determine final pareto front
        self.pareto_front = self.natural_selection(self.pareto_front)

        # Record the overall pareto front
        pareto_front_dict = {}
        for member in self.pareto_front:
            pareto_front_dict[str(member.dna)] = member.fitness
        df_pareto_front = pd.DataFrame.from_dict(pareto_front_dict, orient='index', columns=self.metrics)

        df_graveyard = self.turn_to_dataframe(self.graveyard)
        df_VIPs = self.turn_to_dataframe(self.VIPs)

        # df_graveyard.to_excel('graveyard_tests_10.xlsx')
        # df_VIPs.to_excel('VIPs_tests_10.xlsx')
        # df_pareto_front.to_excel('pareto_front_tests_10.xlsx')

        # self.rng = np.random.default_rng(42)
        # print(self.rng.integers(10, size=10))
        # self.rng_1 = np.random.default_rng(42)
        # print(self.rng_1.integers(10, size=10))

        end = time.time()
        self.plot(df_graveyard[self.metrics[0]], df_graveyard[self.metrics[2]], df_VIPs[self.metrics[0]], df_VIPs[self.metrics[2]], df_pareto_front[self.metrics[0]], df_pareto_front[self.metrics[1]])
        # self.indicators_actions_analysis(df_graveyard)
        # print(self.rng_init.integers(10,size=4))

        print(f'elapsed time for Cluster simulation: {end-start}')

    def turn_to_dataframe(self, dict_obj):
        dfs = []
        for i in range(len(dict_obj.keys())):
            # df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
            df = pd.DataFrame.from_dict(dict_obj[i], orient='index', columns=self.metrics)
            df['policy'] = df.index
            df['generation'] = i
            dfs.append(df)
        df = pd.concat(dfs)
        df.reset_index(drop=True, inplace=True)
        # print(df.head)
        return df

    def plot(self, x1, y1, x2, y2, x3, y3):
        # print(self.rng.integers(10, size=10))
        # ys = dist_dict_diff
        # xs = [x for x in range(len(ys))]

        plt.subplot(2, 2, 1)
        plt.scatter(x1, y1)
        plt.title('graveyard')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 2, 2)
        plt.scatter(x2, y2)
        plt.title('VIPs')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 2, 3)
        plt.scatter(x3, y3)
        plt.title('pareto front')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 2, 4)
        y4 = self.convergence
        x4 = [x for x in range(len(y4))]
        plt.plot(x4, y4)
        plt.title('convergence to reference point')
        plt.ylabel('mean generational distance to reference point')
        plt.xlabel('generation')

        plt.show()
        # Make sure to close the plt object once done
        plt.close()
        pass

    def indicators_actions_analysis(self, df):
        action_dict = {}
        indicator_dict = {}
        for policy_in_df in df['policy']:
            policy = policy_in_df.split(',')

            indicators = []
            actions = []
            for pol in policy:
                if '<' in pol:
                    indicators.append(pol)
                elif '|' in pol:
                    actions.append(pol)

            # Separate actions
            actions = [action.split(' ')[1] for action in actions]
            actions = [actions.split('|') for actions in actions]

            for act_ in actions:
                for act in act_:
                    name, value = act.split('_')
                    if name in action_dict.keys():
                        action_dict[name].append(float(value))
                    else:
                        action_dict[name] = [float(value)]

            # Separate indicators
            indicators = [indicator.strip(' ') for indicator in indicators]
            indicators = [indicator.split(' < ') for indicator in indicators]

            for ind in indicators:
                if ind[0] in indicator_dict.keys():
                    indicator_dict[ind[0]].append(float(ind[1]))
                else:
                    indicator_dict[ind[0]] = [float(ind[1])]
        # This assumes all indicator names from the input are also present in the indicator_dict (i.e. they were used in at least one policy tree). Dito fro the actions
        for idx, indicator_name in enumerate(self.feature_names):
            plt.subplot(2, 3, idx+1)
            y = np.array(indicator_dict[indicator_name])
            plt.hist(y)
            plt.title(indicator_name)

        for idx, action_name in enumerate(self.action_names):
            plt.subplot(2, 3, idx+4)
            y = np.array(action_dict[action_name])
            plt.hist(y)
            plt.title(action_name)

        plt.show()
        plt.close()

    def iterate(self):
        nfe = 0
        generation = 0
        while nfe < self.max_nfe:

            # Populate a generation, depending on the number of available parents
            self.populate()

            # Find the non dominated solutions in a generation
            # self.non_dominated = self.natural_selection(self.family)
            rank_1 = self.natural_selection(self.family)
            rank_2 = self.natural_selection(list(set(self.family).difference(rank_1)))
            rank_3 = self.natural_selection(list(set(self.family).difference(rank_2)))
            self.non_dominated = rank_1 + rank_2 + rank_3

            # Add these solutions also to the pareto front if they are non_dominated throughout the generations
            if generation == 0:
                self.pareto_front = self.non_dominated
            self.add_to_pareto_front()

            # self.add_to_archive()

            # Experimental section ##
            # Keep the solutions that can 'hold hands' meaning that if a new solution falls within the radius a
            # previous pareto solution has with another previous pareto solution, it can stay on as a 'holding hands'
            # solution to serve as an alternative to the strictly minimal pareto front
            # pareto_distances = []
            # for i in range(len(self.pareto_front)):
            #     for j in range(i+1, len(self.pareto_front)):
            #         pareto_distances.append(self.distance(self.pareto_front[i].fitness, self.pareto_front[j].fitness))
            # print(pareto_distances)

            # pareto_distances = []
            # for i in range(len(self.pareto_front)-1):
            #     pareto_distances.append([self.distance([self.pareto_front[i].fitness[0]], [self.pareto_front[i+1].fitness[0]]), self.distance([self.pareto_front[i].fitness[1]], [self.pareto_front[i+1].fitness[1]])])

            # pareto_distances = {}
            # # Iterate over metrics
            # for i, m in enumerate(self.pareto_front[0].fitness):
            #     pareto_distances[i] = []
            #     # Iterate over solutions in pareto_front
            #     for sol in range(len(self.pareto_front)-1):
            #         dist = self.distance(self.pareto_front[sol].fitness, self.pareto_front[sol+1].fitness)
            #         if dist != 0:
            #             pareto_distances[i].append(dist)
            #
            # min_vals = []
            # for m in pareto_distances.keys():
            #     min_val = min(pareto_distances[m])
            #     min_vals.append(min_val)
            # print(min_vals)
            # # min_vals = np.array(min_vals)
            #
            # # result = []
            # for idx, elem in enumerate(self.non_dominated):
            #     elem = elem.fitness
            #     result = [a - b for a, b in zip(elem, min_vals)]
            #     self.non_dominated[idx].fitness = result
            #
            # # self.non_dominated = [item - min_val for item, min_val in zip([item.fitness for item in self.non_dominated], min_vals)]
            # self.add_to_pareto_front()


            ###

            # print(f'pareto front: {[value.fitness for value in self.pareto_front]}')
            # # determine position of parents in solution space
            # self.center_position.append(self.determine_center_position(self.parents[0].fitness, self.parents[1].fitness))

            nfe += len(self.family)

            # Record all organisms per generation
            graveyard_dict = {}
            for member in self.family:
                graveyard_dict[str(member.dna)] = member.fitness
            self.graveyard[generation] = graveyard_dict

            # Record the non dominated organisms per generation
            VIPs_dict = {}
            for member in self.non_dominated:
                VIPs_dict[str(member.dna)] = member.fitness
            self.VIPs[generation] = VIPs_dict

            print(f'end of generation: {generation}')
            generation += 1

            # Calculate distance between reference point and non_dominated solutions to track convergence
            x1 = -10
            x2 = -100
            x3 = 0
            P_ref = [x1, x2, x3]
            # combine distances from non_dominated_solutions
            dist_list = []
            for solution in self.non_dominated:
                dist = self.distance(P_ref, solution.fitness)
                dist_list.append(dist)
            self.convergence.append(mean(dist_list))
        return

    def random_tree(self, terminal_ratio=0.5):
        num_features = len(self.feature_names)

        depth = self.rng_tree.integers(1, self.max_depth + 1)
        # print(f'a new tree is made')
        L = []
        S = [0]

        while S:
            current_depth = S.pop()

            # action node
            if current_depth == depth or (current_depth > 0 and \
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

                    action_input = f'miu_{self.rng_tree.integers(*self.action_bounds[0])}|sr_{self.rng_tree.uniform(*self.action_bounds[1])}|irstp_{self.rng_tree.uniform(*self.action_bounds[2])}'
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
        metrics = []
        fitness = self.model.POT_control(T)
        for idx, m in enumerate(self.metrics):
            metrics.append(fitness[idx])
        return metrics

    # def policy_tree_RICE_fitness(self, T):
    #     m1, m2 = self.model.POT_control(T)
    #     # print(m1, m2, m3, T)
    #     return [m1, m2]

    def populate(self):
        # Swipe generational variables
        self.parents = []
        self.children = []
        self.family = []
        self.non_dominated = []

        # If there are no parents, create them
        if not self.pareto_front:
            for _ in range(self.num_parents):
                parent = Organism()
                parent.dna = self.random_tree()
                parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
                self.parents.append(parent)

        # If there are possible parents, choose two and let the rest be children
        elif self.pareto_front:
            if len(self.pareto_front) >= self.num_parents:
                self.parents = self.rng_populate.choice(self.pareto_front, self.num_parents, replace=False)

                # Take the other non_dominated solutions as children
                self.children = [i for i in self.pareto_front if i not in self.parents]
                # Chuck two(?)/ half out if every solution was non-dominated i.e. len(self.non_dominated) >= 8
                if len(self.pareto_front) >= self.num_children+self.num_parents:  # 8
                    idx = self.num_children/2
                    self.children = self.children[idx:]  # 6:
            # Else if there are less pareto solutions than number of parents required, take the pareto solutions as parents and create a random other parent
            else:
                # self.parents = [self.pareto_front[0]]
                self.parents = self.pareto_front
                while len(self.parents) < self.num_parents:
                    # Create a random other parent
                    parent = Organism()
                    parent.dna = self.random_tree()
                    parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
                    self.parents.append(parent)

            # TODO:: In case every family member is non dominated, there is no more progression, what to do?
        # print(f'parent_dna: {[str(value.dna) for value in self.parents]}')
        self.family.extend(self.parents)
        self.family.extend(self.children)

        # print(f'parents: {len(self.parents)}')
        # print(f'children: {len(self.children)}')
        # print(f'family: {len(self.family)}')

        # for _ in range(self.num_children):
        while len(self.children) < self.num_children:
            # print(len(self.children))
            child = Organism()
            P1, P2 = self.rng_populate.choice(
                self.parents, 2, replace=False)
            child.dna = self.crossover(P1.dna, P2.dna)[0]

            # bloat control
            while child.dna.get_depth() > self.max_depth:
                child.dna = self.crossover(P1.dna, P2.dna)[0]

            # Mutate (with probability of mutation accounted for in function)
            child.dna = self.mutate(child.dna)
            child.dna.prune()

            # Welcome child to family
            child.fitness = self.policy_tree_RICE_fitness(child.dna)
            self.children.append(child)
            self.family.append(child)
        return

    def natural_selection(self, list_obj):
        A = np.array(list_obj)
        N = len(list_obj)
        keep = np.ones(N, dtype=bool)

        for i in range(N):
            for j in range(i + 1, N):
                if keep[j] and self.dominates(A[i].fitness, A[j].fitness):
                    keep[j] = False

                elif keep[i] and self.dominates(A[j].fitness, A[i].fitness):
                    keep[i] = False

                elif self.same_box(np.array(A[i].fitness), np.array(A[j].fitness)):
                    keep[self.rng_natural_selection.choice([i, j])] = False

        return list(A[keep])

    def add_to_pareto_front(self):
        for i, member_candidate in enumerate(self.non_dominated):
            for j, member_established in enumerate(self.pareto_front):
                if self.dominates(member_candidate.fitness, member_established.fitness):
                    self.pareto_front.pop(j)
                    self.pareto_front.append(member_candidate)

        # Remove duplicates from pareto_front if present:
        self.pareto_front = list(set(self.pareto_front))
        return



        # # self.non_dominated = []
        #
        # # for member in self.family:
        # #     print(member.fitness)
        # # Get all possible combinations in family
        # for organism_combo in itertools.permutations(list_obj, 2):
        # # for organism_combo in itertools.permutations(self.family, 2):
        #     # print(organism_combo)
        #     if self.dominates(organism_combo[0].fitness, organism_combo[1].fitness):
        #         # print(f'{organism_combo[0]} dominates {organism_combo[1]}')
        #         self.non_dominated.append(organism_combo[0])
        # return

    # def add_to_archive(self):
    #     num_m  = len(self.metrics)
    #     ar = np.zeros(len(self.pareto_front))
    #     for m in range(num_m):
    #         m_list = []
    #         for member in self.pareto_front:
    #             m_list.append(member.fitness[m])
    #         ar_m = np.array(m_list)
    #         ar_m = np.sort(ar_m)
    #         ar = np.vstack((ar, ar_m))
    #     print(ar)
    #     return

    def dominates(self, a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
        # So to circumvent this but still allow multiobjective optimisation in different directions under the
        # constraint that every number is positive, just add a large number to every index.
        large_number = 1000000000

        a = np.array(a)
        a = a + large_number

        b = np.array(b)
        b = b + large_number
        # print(f'a: {a}')
        # print(f'b: {b}')
        return (np.all(a <= b) and np.any(a < b))

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
                        action_input = f'miu_{self.rng_mutate.integers(*self.action_bounds[0])}|sr_{self.rng_mutate.uniform(*self.action_bounds[1])}|irstp_{self.rng_mutate.uniform(*self.action_bounds[2])}'
                        item.value = action_input

        return P

    def bounded_gaussian(self, x, bounds):
        # do mutation in normalized [0,1] to avoid sigma scaling issues
        lb, ub = bounds
        xnorm = (x - lb) / (ub - lb)
        x_trial = np.clip(xnorm + self.rng_gauss.normal(0, scale=0.1), 0, 1)

        return lb + x_trial * (ub - lb)

    def same_box(self, a, b):
        if np.any(self.epsilons):
            a = a // self.epsilons
            b = b // self.epsilons
        return np.all(a == b)

    def distance(self, P1, P2):
        # Input is list
        num_dimensions = len(P1)
        dist = []
        for dimension in range(num_dimensions):
            dist_ = (P2[dimension] - P1[dimension]) ** 2
            dist.append(dist_)
        distance = math.sqrt(sum(dist))
        return distance


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None


Cluster(20, 80, master_rng=master_rng)
