import numpy as np
from POT.tree import PTree
from RICE_model.IAM_RICE import RICE
import copy
import pandas as pd
import math
import time

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from statistics import mean
# master_rng = np.random.default_rng(42)  # Master RNG


class Cluster_:
    def __init__(self, num_parents, num_children, master_rng,
                 years_10,
                 regions,
                 metrics,
                 metrics_choice,
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
                 P_ref,
                 pareto_front=None,
                 ):
        # start = time.time()
        self.graveyard = {}
        self.VIPs = {}
        self.non_dominated = []

        self.holding_hands = []
        self.parents = []
        self.children = []
        self.family = []
        self.num_parents = num_parents
        self.num_children = num_children

        self.nfe = 0
        self.generation = 0

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

        self.model = RICE(years_10, regions)

        # self.metrics = metrics
        self.metrics_choice = metrics_choice
        self.metrics = []
        for idx in self.metrics_choice:
            self.metrics = self.metrics + [metrics[idx]]
        print(self.metrics)
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
        self.P_ref = P_ref

        self.central_point_tracker = []
        self.spread_tracker = []

        self.pareto_front = []
        if pareto_front:
            for item in pareto_front:
                item_ = Organism()
                item_.dna = item.dna
                item_.fitness = self.policy_tree_RICE_fitness(item_.dna)
                self.pareto_front.append(item_)

    def run(self):
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
        convergence_dict = {'convergence': self.convergence, 'generation': [x for x in range(len(self.convergence))]}
        df_convergence = pd.DataFrame(data=convergence_dict)

        self.plot(df_graveyard[self.metrics[0]], df_graveyard[self.metrics[1]], df_VIPs[self.metrics[0]], df_VIPs[self.metrics[1]], df_pareto_front[self.metrics[0]], df_pareto_front[self.metrics[1]])
        # self.indicators_actions_analysis(df_graveyard)

        # print(self.central_point_tracker)
        # print(self.spread_tracker)

        # Write dictionaries to an excel file
        # writer = pd.ExcelWriter(f'{self.save_location}/{self.file_name}.xlsx')
        # writer = pd.ExcelWriter("test_run_increasing_objectives.xlsx")
        # df_graveyard.to_excel(writer, sheet_name='graveyard')
        # df_VIPs.to_excel(writer, sheet_name='VIPs')
        # df_pareto_front.to_excel(writer, sheet_name='pareto_front')
        # df_convergence.to_excel(writer, sheet_name='convergence_ref_point')
        # writer.close()
        return self.pareto_front

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

        plt.subplot(2, 3, 1)
        plt.scatter(x1, y1)
        plt.title('graveyard')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 3, 2)
        plt.scatter(x2, y2)
        plt.title('VIPs')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 3, 3)
        plt.scatter(x3, y3)
        plt.title('pareto front')
        plt.ylabel(self.metrics[1])
        plt.xlabel(self.metrics[0])

        plt.subplot(2, 3, 4)
        y4 = self.convergence
        x4 = [x for x in range(len(y4))]
        plt.plot(x4, y4)
        plt.title('convergence to reference point')
        plt.ylabel('mean generational distance to reference point')
        plt.xlabel('generation')

        # 3d plot -------------
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plt.subplot(2, 3, 4, projection='3d')
        # ax = plt.axes(projection='3d')
        # # ax.scatter([s.objectives[0] for s in algorithm.result],
        # #            [s.objectives[1] for s in algorithm.result],
        # #            [s.objectives[2] for s in algorithm.result])
        # x = []
        # y = []
        # z = []
        # for item in self.central_point_tracker:
        #     x.append(item[0])
        #     y.append(item[1])
        #     z.append(item[2])
        #
        # ax.scatter(x, y, z, color='b')
        # # ax.text(x, y, z, '%s' % (label), size=20, zorder=1, color='k')
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        #
        # # ax.set_xlim([-43000, -42000])
        # # ax.set_ylim([-10000000, -1000000])
        # # ax.set_zlim([2, 6])
        # ax.view_init(elev=30.0, azim=15)
        # plt.show()
        # --------------

        plt.subplot(2, 3, 5)
        dist_ = [0]
        for i in range(1, len(self.central_point_tracker)):
            dist_.append(self.distance(self.central_point_tracker[i - 1], self.central_point_tracker[i]))
        x5 = [x for x in range(len(dist_))]
        y5 = dist_
        plt.plot(x5, y5, color='r')
        plt.xlabel("generation")
        plt.ylabel("central point")
        plt.title("generational distance tracker")

        plt.subplot(2, 3, 6)
        x6 = [x for x in range(len(self.spread_tracker))]
        y6 = self.spread_tracker
        plt.plot(x6, y6, color='g')
        plt.xlabel("generation")
        plt.ylabel("spread")
        plt.title("spread tracker")

        plt.legend()
        plt.show()
        plt.close()

        # plt.show()
        plt.savefig(f'{str(self.metrics_choice)}.png')
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

            # Take average position of a pareto_front
            # Calculate difference from each point on the pareto_front to the average point
            # Track change in averages position and change in sum of differences
            central_point = self.get_central_point([x.fitness for x in self.pareto_front])
            dist_list = []
            for item in self.pareto_front:
                dist_ = self.distance(central_point, item.fitness)
                dist_list.append(dist_)
            self.spread_tracker.append(sum(dist_list))
            self.central_point_tracker.append(central_point)
            # When spread decreases (below 10%, arbitrarily chose): take the most extreme solutions from the non_dominated solutions and add them to the pareto front
            # Find the most extreme solution for each objective
            if generation >= 1:
                if self.spread_tracker[-1] < (self.spread_tracker[-2]*0.9):
                    for idx_m in range(len(self.metrics_choice)):
                        ar = np.array([item.fitness[idx_m] for item in self.non_dominated])
                        extreme_idx = np.argmax(ar)
                        # print(np.max(ar))
                        extreme_solution = self.non_dominated[extreme_idx]
                        self.pareto_front.append(extreme_solution)
                        # print(self.non_dominated[extreme_value].fitness[idx_m])


            # Calculate distance between reference point and non_dominated solutions to track convergence
            # x1 = -10
            # x2 = 0.01
            # x3 = 1
            # P_ref = [x1, x2, x3]
            # combine distances from non_dominated_solutions
            dist_list = []
            for solution in self.pareto_front:
                dist = self.distance(self.P_ref, solution.fitness)
                dist_list.append(dist)
            self.convergence.append(mean(dist_list))

            print(f'end of generation: {generation}')
            generation += 1
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
        for idx, m in enumerate(self.metrics_choice):
            metrics.append(fitness[idx])
        return metrics

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
        return np.all(a <= b) and np.any(a < b)

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

    def get_central_point(self, data_list):
        # This function calculates the average of all positions in a list, so its return value need not be a value in the original list
        return [sum(x)/len(x) for x in zip(*data_list)]


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None

# start = time.time()
# years_10 = []
# for i in range(2005, 2315, 10):
#     years_10.append(i)
#
# regions = [
#     "US",
#     "OECD-Europe",
#     "Japan",
#     "Russia",
#     "Non-Russia Eurasia",
#     "China",
#     "India",
#     "Middle East",
#     "Africa",
#     "Latin America",
#     "OHI",
#     "Other non-OECD Asia",
# ]

# run = Cluster(20, 80, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'utility', 'temp_overshoots'],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=1000,
#         epsilons=np.array([0.01, 0.01, 0.01]),
#         ).run()

# run12 = Cluster(20, 80, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'damages', 'temp_overshoots'],
#         metrics_choice=[0, 1],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=3000,
#         epsilons=np.array([0.01, 0.01]),
#         P_ref=[-10, 0.01],).run()
#
# run23 = Cluster(20, 80, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'damages', 'temp_overshoots'],
#         metrics_choice=[1, 2],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=3000,
#         epsilons=np.array([0.01, 0.01]),
#         P_ref=[0.01, 1],
#         pareto_front=None,).run()
#
# run31 = Cluster(20, 80, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'damages', 'temp_overshoots'],
#         metrics_choice=[2, 0],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=30000,
#         epsilons=np.array([0.01, 0.01]),
#         P_ref=[1, -10],
#         pareto_front=None,).run()
#
# pareto_front_combined = run12 + run23 + run31
#
# run123 = Cluster(20, 80, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'damages', 'temp_overshoots'],
#         metrics_choice=[0, 1, 2],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=3000,
#         epsilons=np.array([0.01, 0.01, 0.01]),
#         P_ref=[-10, 0.01, 1],
#         pareto_front=pareto_front_combined,
#                  ).run()
# end = time.time()
# print(f'Elapsed time: {(end-start)/60} minutes.')
# ---------------------------------
# run123 = Cluster(2, 8, master_rng=master_rng,
#         years_10=years_10,
#         regions=regions,
#         metrics=['period_utility', 'damages', 'temp_overshoots'],
#         metrics_choice=[0, 1, 2],
#         # Tree variables
#         action_names=['miu', 'sr', 'irstp'],
#         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#         feature_names=['mat', 'net_output', 'year'],
#         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#         max_depth=4,
#         discrete_actions=False,
#         discrete_features=False,
#         # Optimization variables
#         mutation_prob=0.5,
#         max_nfe=100,
#         epsilons=np.array([0.01, 0.01, 0.01]),
#         P_ref=[-10, 0.01, 1],
#                  ).run()

# run123[2].to_excel('run123_pareto_front_1000x3.xlsx')


# run12[2].to_excel('run12_pareto_front_20000.xlsx')
# run23[2].to_excel('run23_pareto_front_20000.xlsx')
# run31[2].to_excel('run31_pareto_front_20000.xlsx')

## VERSION BEFORE 12-09-2023, BEFORE CORRECT IMPLEMENTATION OF RANDOMNESS ----------------------------------------
# import numpy as np
# rng = np.random.default_rng(seed=44)
# from POT.tree import PTree
# from RICE_model.IAM_RICE import RICE
# # import random
# import time
# import copy
# import itertools
# import pandas as pd
# import math
#
# import matplotlib.pyplot as plt
#
# from statistics import mean
#
# # np.random.seed(42)
#
#
# # rng = np.random.default_rng(0)
# # out = [rng.choice([0, 1], p=[0.5, 0.5]) for _ in range(10)]
# # print(out)
#
#
# class Cluster:
#     def __init__(self, num_parents, num_children, rng):
#         self.graveyard = {}
#         self.VIPs = {}
#         # self.overall_pareto_front = {}
#         self.non_dominated = []
#         self.pareto_front = []
#         self.parents = []
#         self.children = []
#         self.family = []
#         self.num_parents = num_parents
#         self.num_children = num_children
#
#         self.nfe = 0
#         self.generation = 0
#
#         if rng is None:
#             rng = np.random.default_rng()
#         self.rng = rng  # np.random.default_rng(seed=seed)
#         # self.rng = np.random.default_rng(seed=44)
#
#         self.convergence = []
#
#         # The center position takes the average dist
#         self.center_position = []
#
#         # Model variables
#         self.years_10 = []
#         for i in range(2005, 2315, 10):
#             self.years_10.append(i)
#
#         self.regions = [
#             "US",
#             "OECD-Europe",
#             "Japan",
#             "Russia",
#             "Non-Russia Eurasia",
#             "China",
#             "India",
#             "Middle East",
#             "Africa",
#             "Latin America",
#             "OHI",
#             "Other non-OECD Asia",
#         ]
#         self.model = RICE(self.years_10, self.regions)
#         # Tree variables
#         # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#         self.action_names = ['miu', 'sr', 'irstp']
#         self.action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
#         self.feature_names = ['mat', 'net_output', 'year']
#         self.feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#         self.max_depth = 4
#         self.discrete_actions = False
#         self.discrete_features = False
#         # Optimization variables
#         self.mutation_prob = 0.5
#         # self.max_nfe = 100
#
#     def run_one_gen(self):
#         # Start optimization
#         self.iterate()
#         # Determine final pareto front
#         self.pareto_front = self.natural_selection(self.pareto_front)
#
#         # Record the overall pareto front
#         pareto_front_dict = {}
#         for member in self.pareto_front:
#             pareto_front_dict[str(member.dna)] = member.fitness
#         df_pareto_front = pd.DataFrame.from_dict(pareto_front_dict, orient='index', columns=['ofv1', 'ofv2'])
#
#         df_graveyard = self.turn_to_dataframe(self.graveyard)
#         df_VIPs = self.turn_to_dataframe(self.VIPs)
#
#         # df_graveyard.to_excel('graveyard_tests_10.xlsx')
#         # df_VIPs.to_excel('VIPs_tests_10.xlsx')
#         # df_pareto_front.to_excel('pareto_front_tests_10.xlsx')
#
#         # self.rng = np.random.default_rng(42)
#         # print(self.rng.integers(10, size=10))
#         # self.rng_1 = np.random.default_rng(42)
#         # print(self.rng_1.integers(10, size=10))
#
#         self.plot(df_graveyard['ofv1'], df_graveyard['ofv2'], df_VIPs['ofv1'], df_VIPs['ofv2'], df_pareto_front['ofv1'], df_pareto_front['ofv2'])
#         # self.indicators_actions_analysis(df_graveyard)
#         return
#
#         # # Start optimization
#         # self.iterate()
#         # # Determine final pareto front
#         # self.pareto_front = self.natural_selection(self.pareto_front)
#         #
#         # # Record the overall pareto front
#         # pareto_front_dict = {}
#         # for member in self.pareto_front:
#         #     pareto_front_dict[str(member.dna)] = member.fitness
#         # df_pareto_front = pd.DataFrame.from_dict(pareto_front_dict, orient='index', columns=['ofv1', 'ofv2'])
#         #
#         # df_graveyard = self.turn_to_dataframe(self.graveyard)
#         # df_VIPs = self.turn_to_dataframe(self.VIPs)
#         #
#         # df_graveyard.to_excel('graveyard_tests_10.xlsx')
#         # df_VIPs.to_excel('VIPs_tests_10.xlsx')
#         # df_pareto_front.to_excel('pareto_front_tests_10.xlsx')
#         #
#         # # self.rng = np.random.default_rng(42)
#         # # print(self.rng.integers(10, size=10))
#         # # self.rng_1 = np.random.default_rng(42)
#         # # print(self.rng_1.integers(10, size=10))
#         #
#         # self.plot(df_graveyard['ofv1'], df_graveyard['ofv2'], df_VIPs['ofv1'], df_VIPs['ofv2'], df_pareto_front['ofv1'], df_pareto_front['ofv2'])
#         # # self.indicators_actions_analysis(df_graveyard)
#
#     def turn_to_dataframe(self, dict_obj):
#         dfs = []
#         for i in range(len(dict_obj.keys())):
#             # df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
#             df = pd.DataFrame.from_dict(dict_obj[i], orient='index', columns=['ofv1', 'ofv2'])
#             df['policy'] = df.index
#             df['generation'] = i
#             dfs.append(df)
#         df = pd.concat(dfs)
#         df.reset_index(drop=True, inplace=True)
#         # print(df.head)
#         return df
#
#     def plot(self, x1, y1, x2, y2, x3, y3):
#         # print(self.rng.integers(10, size=10))
#         # ys = dist_dict_diff
#         # xs = [x for x in range(len(ys))]
#
#         plt.subplot(2, 2, 1)
#         plt.scatter(x1, y1)
#         plt.title('graveyard')
#         plt.ylabel('ofv2 - temp_overshoots')
#         plt.xlabel('ofv1 - period utility')
#
#         plt.subplot(2, 2, 2)
#         plt.scatter(x2, y2)
#         plt.title('VIPs')
#         plt.ylabel('ofv2 - temp_overshoots')
#         plt.xlabel('ofv1 - period utility')
#
#         plt.subplot(2, 2, 3)
#         plt.scatter(x3, y3)
#         plt.title('pareto front')
#         plt.ylabel('ofv2 - temp_overshoots')
#         plt.xlabel('ofv1 - period utility')
#
#         plt.subplot(2, 2, 4)
#         y4 = self.convergence
#         x4 = [x for x in range(len(y4))]
#         plt.plot(x4, y4)
#         plt.title('convergence to reference point')
#         plt.ylabel('mean generational distance to reference point')
#         plt.xlabel('generation')
#
#         plt.show()
#         # Make sure to close the plt object once done
#         plt.close()
#         # --------------------------
#
#         # t = np.arange(1000) / 100.
#         # x = np.sin(2 * np.pi * 10 * t)
#         # y = np.cos(2 * np.pi * 10 * t)
#         #
#         # fig = plt.figure()
#         # ax1 = plt.subplot(211)
#         # ax2 = plt.subplot(212)
#         #
#         # ax1.plot(t, x)
#         # ax2.plot(t, y)
#         #
#         # ax1.get_shared_x_axes().join(ax1, ax2)
#         # ax1.set_xticklabels([])
#         # # ax2.autoscale() ## call autoscale if needed
#         #
#         # plt.show()
#         pass
#
#     def indicators_actions_analysis(self, df):
#         action_dict = {}
#         indicator_dict = {}
#         for policy_in_df in df['policy']:
#             policy = policy_in_df.split(',')
#
#             indicators = []
#             actions = []
#             for pol in policy:
#                 if '<' in pol:
#                     indicators.append(pol)
#                 elif '|' in pol:
#                     actions.append(pol)
#
#             # Separate actions
#             actions = [action.split(' ')[1] for action in actions]
#             actions = [actions.split('|') for actions in actions]
#
#             for act_ in actions:
#                 for act in act_:
#                     name, value = act.split('_')
#                     if name in action_dict.keys():
#                         action_dict[name].append(float(value))
#                     else:
#                         action_dict[name] = [float(value)]
#
#             # Separate indicators
#             indicators = [indicator.strip(' ') for indicator in indicators]
#             indicators = [indicator.split(' < ') for indicator in indicators]
#
#             for ind in indicators:
#                 if ind[0] in indicator_dict.keys():
#                     indicator_dict[ind[0]].append(float(ind[1]))
#                 else:
#                     indicator_dict[ind[0]] = [float(ind[1])]
#         # This assumes all indicator names from the input are also present in the indicator_dict (i.e. they were used in at least one policy tree). Dito fro the actions
#         for idx, indicator_name in enumerate(self.feature_names):
#             plt.subplot(2, 3, idx+1)
#             y = np.array(indicator_dict[indicator_name])
#             plt.hist(y)
#             plt.title(indicator_name)
#
#         for idx, action_name in enumerate(self.action_names):
#             plt.subplot(2, 3, idx+4)
#             y = np.array(action_dict[action_name])
#             plt.hist(y)
#             plt.title(action_name)
#
#         plt.show()
#         plt.close()
#
#     def iterate(self):
#         # nfe = 0
#         # generation = 0
#         # while nfe < self.max_nfe:
#
#         # Populate a generation, depending on the number of available parents
#         self.populate()
#         # Find the non dominated solutions in a generation
#         self.non_dominated = self.natural_selection(self.family)
#         # Add these solutions also to the pareto front if they are non_dominated throughout the generations
#         if self.generation == 0:
#             self.pareto_front = self.non_dominated
#         self.add_to_pareto_front()
#
#         print(f'pareto front: {[value.fitness for value in self.pareto_front]}')
#         # # determine position of parents in solution space
#         # self.center_position.append(self.determine_center_position(self.parents[0].fitness, self.parents[1].fitness))
#
#         self.nfe += len(self.family)
#
#         # Record all organisms per generation
#         graveyard_dict = {}
#         for member in self.family:
#             graveyard_dict[str(member.dna)] = member.fitness
#         self.graveyard[self.generation] = graveyard_dict
#
#         # Record the non dominated organisms per generation
#         VIPs_dict = {}
#         for member in self.non_dominated:
#             VIPs_dict[str(member.dna)] = member.fitness
#         self.VIPs[self.generation] = VIPs_dict
#
#         print(f'end of generation: {self.generation}')
#         self.generation += 1
#
#         # Calculate distance between reference point and non_dominated solutions to track convergence
#         x1 = -10
#         x2 = 2
#         P_ref = [x1, x2]
#         # combine distances from non_dominated_solutions
#         dist_list = []
#         for solution in self.non_dominated:
#             dist = self.distance(P_ref, solution.fitness)
#             dist_list.append(dist)
#         self.convergence.append(mean(dist_list))
#         # nfe = 0
#         # generation = 0
#         # while nfe < self.max_nfe:
#         #
#         #     # Populate a generation, depending on the number of available parents
#         #     self.populate()
#         #     # Find the non dominated solutions in a generation
#         #     self.non_dominated = self.natural_selection(self.family)
#         #     # Add these solutions also to the pareto front if they are non_dominated throughout the generations
#         #     if generation == 0:
#         #         self.pareto_front = self.non_dominated
#         #     self.add_to_pareto_front()
#         #
#         #     print(f'pareto front: {[value.fitness for value in self.pareto_front]}')
#         #     # # determine position of parents in solution space
#         #     # self.center_position.append(self.determine_center_position(self.parents[0].fitness, self.parents[1].fitness))
#         #
#         #     nfe += len(self.family)
#         #
#         #     # Record all organisms per generation
#         #     graveyard_dict = {}
#         #     for member in self.family:
#         #         graveyard_dict[str(member.dna)] = member.fitness
#         #     self.graveyard[generation] = graveyard_dict
#         #
#         #     # Record the non dominated organisms per generation
#         #     VIPs_dict = {}
#         #     for member in self.non_dominated:
#         #         VIPs_dict[str(member.dna)] = member.fitness
#         #     self.VIPs[generation] = VIPs_dict
#         #
#         #     print(f'end of generation: {generation}')
#         #     generation += 1
#         #
#         #     # Calculate distance between reference point and non_dominated solutions to track convergence
#         #     x1 = -10
#         #     x2 = 2
#         #     P_ref = [x1, x2]
#         #     # combine distances from non_dominated_solutions
#         #     dist_list = []
#         #     for solution in self.non_dominated:
#         #         dist = self.distance(P_ref, solution.fitness)
#         #         dist_list.append(dist)
#         #     self.convergence.append(mean(dist_list))
#         return
#
#     def random_tree(self, terminal_ratio=0.5):
#         num_features = len(self.feature_names)
#
#         depth = self.rng.integers(1, self.max_depth + 1)
#         # print(f'a new tree is made')
#         L = []
#         S = [0]
#
#         while S:
#             current_depth = S.pop()
#
#             # action node
#             if current_depth == depth or (current_depth > 0 and \
#                                           self.rng.random() < terminal_ratio):
#                 if self.discrete_actions:
#                     L.append([str(self.rng.choice(self.action_names))])
#                 else:
#                     # a = np.random.choice(num_actions)  # SD changed
#                     # action_name = self.action_names[a]
#                     # action_value = np.random.uniform(*self.action_bounds[a])
#                     # action_input = f'{action_name}_{action_value}'
#                     # # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     # L.append([action_input])  # SD changed
#
#                     # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                     # number_of_actions_on_leaf_node = np.random.randint(1, num_actions+1)
#                     # actions = np.random.choice(num_actions, number_of_actions_on_leaf_node, replace=False)
#                     # collect_actions = []
#                     # for a in actions:
#                     #     action_name = self.action_names[a]
#                     #     action_value = np.random.uniform(*self.action_bounds[a])
#                     #     action_input = f'{action_name}_{action_value}'
#                     #     collect_actions.append(action_input)
#                     #     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     #     # L.append([action_input])  # SD changed
#                     # L.append(['|'.join(collect_actions)])  # SD changed
#
#                     # # Now all actions are part of a leaf
#                     # collect_actions = []
#                     # for a in range(num_actions):
#                     #     action_name = self.action_names[a]
#                     #     action_value = np.random.uniform(*self.action_bounds[a])
#                     #     action_input = f'{action_name}_{action_value}'
#                     #     collect_actions.append(action_input)
#                     #     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     #     # L.append([action_input])  # SD changed
#                     # L.append(['|'.join(collect_actions)])  # SD changed
#                     # # print(['|'.join(collect_actions)])
#
#                     action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
#                     L.append([action_input])
#             else:
#                 x = self.rng.choice(num_features)
#                 v = self.rng.uniform(*self.feature_bounds[x])
#                 L.append([x, v])
#                 # print(self.feature_names[x])
#                 S += [current_depth + 1] * 2
#
#         T = PTree(L, self.feature_names, self.discrete_features)
#         T.prune()
#         return T
#
#     # def policy_tree_RICE_fitness(self, T):
#     #     m1, m2, m3 = self.model.POT_control(T)
#     #     # print(m1, m2, m3, T)
#     #     return [m1, m2, m3]
#
#     def policy_tree_RICE_fitness(self, T):
#         m1, m2 = self.model.POT_control(T)
#         # print(m1, m2, m3, T)
#         return [m1, m2]
#
#     def populate(self):
#         # Swipe generational variables
#         self.parents = []
#         self.children = []
#         self.family = []
#         self.non_dominated = []
#
#         # If there are no parents, create them
#         if not self.pareto_front:
#             for _ in range(self.num_parents):
#                 parent = Organism()
#                 parent.dna = self.random_tree()
#                 parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
#                 self.parents.append(parent)
#
#         # If there are possible parents, choose two and let the rest be children
#         elif self.pareto_front:
#             if len(self.pareto_front) >= self.num_parents:
#                 while len(self.parents) < self.num_parents:
#                     P1, P2 = self.rng.choice(
#                         self.pareto_front, 2, replace=False)
#                     self.parents.append(P1)
#                     self.parents.append(P2)
#
#                 # Take the other non_dominated solutions as children
#                 self.children = [i for i in self.pareto_front if i not in self.parents]
#                 # Chuck two(?)/ half out if every solution was non-dominated i.e. len(self.non_dominated) >= 8
#                 if len(self.pareto_front) >= self.num_children+self.num_parents:  # 8
#                     idx = self.num_children-2
#                     self.children = self.children[idx:]  # 6:
#             # Else if there are less pareto solutions than number of parents required, take the pareto solutions as parents and create a random other parent
#             else:
#                 # self.parents = [self.pareto_front[0]]
#                 self.parents = self.pareto_front
#                 while len(self.parents) < self.num_parents:
#                     # Create a random other parent
#                     parent = Organism()
#                     parent.dna = self.random_tree()
#                     parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
#                     self.parents.append(parent)
#
#             # TODO:: In case every family member is non dominated, there is no more progression, what to do?
#         # print(f'parent_dna: {[str(value.dna) for value in self.parents]}')
#         self.family.extend(self.parents)
#         self.family.extend(self.children)
#
#         # print(f'parents: {len(self.parents)}')
#         # print(f'children: {len(self.children)}')
#         # print(f'family: {len(self.family)}')
#
#         # for _ in range(self.num_children):
#         while len(self.children) < self.num_children:
#             # print(len(self.children))
#             child = Organism()
#             P1, P2 = self.rng.choice(
#                 self.parents, 2, replace=False)
#             child.dna = self.crossover(P1.dna, P2.dna)[0]
#
#             # bloat control
#             while child.dna.get_depth() > self.max_depth:
#                 child.dna = self.crossover(P1.dna, P2.dna)[0]
#
#             # Mutate (with probability of mutation accounted for in function)
#             child.dna = self.mutate(child.dna)
#             child.dna.prune()
#
#             # Welcome child to family
#             child.fitness = self.policy_tree_RICE_fitness(child.dna)
#             self.children.append(child)
#             self.family.append(child)
#
#     def natural_selection(self, list_obj):
#         A = np.array(list_obj)
#         N = len(list_obj)
#         keep = np.ones(N, dtype=bool)
#
#         for i in range(N):
#             for j in range(i + 1, N):
#                 if keep[j] and self.dominates(A[i].fitness, A[j].fitness):
#                     keep[j] = False
#
#                 elif keep[i] and self.dominates(A[j].fitness, A[i].fitness):
#                     keep[i] = False
#
#         return list(A[keep])
#
#     def add_to_pareto_front(self):
#         for i, member_candidate in enumerate(self.non_dominated):
#             for j, member_established in enumerate(self.pareto_front):
#                 if self.dominates(member_candidate.fitness, member_established.fitness):
#                     self.pareto_front.pop(j)
#                     self.pareto_front.append(member_candidate)
#
#         # Remove duplicates from pareto_front if present:
#         self.pareto_front = list(set(self.pareto_front))
#         return
#
#
#
#         # # self.non_dominated = []
#         #
#         # # for member in self.family:
#         # #     print(member.fitness)
#         # # Get all possible combinations in family
#         # for organism_combo in itertools.permutations(list_obj, 2):
#         # # for organism_combo in itertools.permutations(self.family, 2):
#         #     # print(organism_combo)
#         #     if self.dominates(organism_combo[0].fitness, organism_combo[1].fitness):
#         #         # print(f'{organism_combo[0]} dominates {organism_combo[1]}')
#         #         self.non_dominated.append(organism_combo[0])
#         # return
#
#     def dominates(self, a, b):
#         # assumes minimization
#         # a dominates b if it is <= in all objectives and < in at least one
#         # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
#         # So to circumvent this but still allow multiobjective optimisation in different directions under the
#         # constraint that every number is positive, just add a large number to every index.
#         large_number = 1000000000
#
#         a = np.array(a)
#         a = a + large_number
#
#         b = np.array(b)
#         b = b + large_number
#         # print(f'a: {a}')
#         # print(f'b: {b}')
#         return (np.all(a <= b) and np.any(a < b))
#
#     def crossover(self, P1, P2):
#         P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
#         # should use indices of ONLY feature nodes
#         feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
#         feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
#         index1 = self.rng.choice(feature_ix1)
#         index2 = self.rng.choice(feature_ix2)
#         slice1 = P1.get_subtree(index1)
#         slice2 = P2.get_subtree(index2)
#         P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
#         P1.build()
#         P2.build()
#         return (P1, P2)
#
#     def mutate(self, P, mutate_actions=True):
#         P = copy.deepcopy(P)
#
#         for item in P.L:
#             if self.rng.random() < self.mutation_prob:
#                 if item.is_feature:
#                     low, high = self.feature_bounds[item.index]
#                     if item.is_discrete:
#                         item.threshold = self.rng.integers(low, high + 1)
#                     else:
#                         item.threshold = self.bounded_gaussian(
#                             item.threshold, [low, high])
#                 elif mutate_actions:
#                     if self.discrete_actions:
#                         item.value = str(self.rng.choice(self.action_names))
#                     else:
#                         # print(item)
#                         # print(self.action_bounds)
#                         # print(item.value)
#                         # item.value = self.bounded_gaussian(
#                         #     item.value, self.action_bounds)
#
#                         # --------
#                         # a = np.random.choice(len(self.action_names))  # SD changed
#                         # action_name = self.action_names[a]
#                         # action_value = np.random.uniform(*self.action_bounds[a])
#                         # action_input = f'{action_name}_{action_value}'
#                         # # print(action_input)
#                         # item.value = action_input
#
#                         action_input = f'miu_{self.rng.integers(*self.action_bounds[0])}|sr_{self.rng.uniform(*self.action_bounds[1])}|irstp_{self.rng.uniform(*self.action_bounds[2])}'
#                         item.value = action_input
#
#         return P
#
#     def bounded_gaussian(self, x, bounds):
#         # do mutation in normalized [0,1] to avoid sigma scaling issues
#         lb, ub = bounds
#         xnorm = (x - lb) / (ub - lb)
#         x_trial = np.clip(xnorm + self.rng.normal(0, scale=0.1), 0, 1)
#
#         return lb + x_trial * (ub - lb)
#
#     def determine_center_position(self, P1, P2):
#         num_dimensions = len(P1)
#         center_position = []
#         for dimension in range(num_dimensions):
#             avg_pos = P1[dimension] + ((P2[dimension]-P1[dimension])/2)
#             center_position.append(avg_pos)
#         return center_position
#
#
#         # Step 1: select best performing parent -> create dominates() function
#         # Step 2: let other parents mutate -> if mutation dominates over
#
#         # -----------
#         # create children by cross-over
#         # let all within a population mutate -> keep the better performing version
#         # Choose the best number of organisms equal to num_parents and choose them as parent clusters -> repeat process
#
#         # ----------
#         # When a cluster is not performing better, keep its solutions and create a new (random) cluster.
#         # When the fitness of two clusters starts to approach each other, freeze one and let the other grow
#
#     def distance(self, P1, P2):
#         # Input is list
#         num_dimensions = len(P1)
#         dist = []
#         for dimension in range(num_dimensions):
#             dist_ = (P2[dimension] - P1[dimension]) ** 2
#             dist.append(dist_)
#         distance = math.sqrt(sum(dist))
#         return distance
#
#
# class Organism:
#     def __init__(self):
#         self.dna = None
#         self.fitness = None
#
#
# # # rng = np.random.default_rng(seed=44)
# # Cluster(2, 8, rng=rng)
# # Cluster(2, 8, rng=rng)
#
#
# class ClusterOpt:
#     def __init__(self, rng=None):
#         if rng is None:
#             rng = np.random.default_rng()
#         self.rng = rng  # np.random.default_rng(seed=seed)
#
#         # np.random.seed(time.perf_counter())
#         self.C1 = Cluster(20, 80, rng=self.rng)
#         # np.random.seed(time.perf_counter())
#         self.C2 = Cluster(20, 80, rng=self.rng)
#
#         self.C1.run_one_gen()
#         self.C2.run_one_gen()
#
#         self.C1.run_one_gen()
#         self.C2.run_one_gen()
#
#         print(mean([organism.fitness] for organism in self.C1.pareto_front))
#         print(mean([organism.fitness] for organism in self.C2.pareto_front))
#
#         self.C1.run_one_gen()
#         self.C2.run_one_gen()
#
#         print(mean([organism.fitness] for organism in self.C1.pareto_front))
#         print(mean([organism.fitness] for organism in self.C2.pareto_front))
#
#         self.C1.run_one_gen()
#         self.C2.run_one_gen()
#
#         print(mean([organism.fitness] for organism in self.C1.pareto_front))
#         print(mean([organism.fitness] for organism in self.C2.pareto_front))
#
#         # print(self.C1.center_position)
#         # print(self.C1.center_position)
#         #
#         # print(self.distance(self.C1.center_position[0], self.C2.center_position[0]))
#
#     def distance(self, P1, P2):
#         # Input is list
#         num_dimensions = len(P1)
#         dist = []
#         for dimension in range(num_dimensions):
#             dist_ = (P2[dimension] - P1[dimension]) ** 2
#             dist.append(dist_)
#         distance = math.sqrt(sum(dist))
#         return distance
#         # return math.sqrt(((P2[0] - P1[0]) ** 2) + ((P2[1] - P1[1]) ** 2) + ((P2[2] - P1[2]) ** 2))
#
#
# ClusterOpt(rng=rng)



# ----- VERSION BEFORE 04-09-2023 BEFORE CHANGING SELF.NON_DOMINATED STRUCTURE -> CHANGING FROM RESET PER FAMILY TO KEEPING AS VARIABLE OVER ENTIRE SIMULATION ---------------------
# import numpy as np
# from POT.tree import PTree
# from RICE_model.IAM_RICE import RICE
# import random
# import time
# import copy
# import itertools
# import pandas as pd
# import math
#
# import matplotlib.pyplot as plt
#
# random.seed(42)
#
#
# class Cluster:
#     def __init__(self, num_parents, num_children):
#         self.graveyard = {}
#         self.VIPs = {}
#         self.non_dominated = ()
#         self.parents = []
#         self.children = []
#         self.family = []
#         self.num_parents = num_parents
#         self.num_children = num_children
#
#         # The center position takes the average dist
#         self.center_position = []
#
#         # Model variables
#         self.years_10 = []
#         for i in range(2005, 2315, 10):
#             self.years_10.append(i)
#
#         self.regions = [
#             "US",
#             "OECD-Europe",
#             "Japan",
#             "Russia",
#             "Non-Russia Eurasia",
#             "China",
#             "India",
#             "Middle East",
#             "Africa",
#             "Latin America",
#             "OHI",
#             "Other non-OECD Asia",
#         ]
#         self.model = RICE(self.years_10, self.regions)
#         # Tree variables
#         # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#         self.action_names = ['miu', 'sr', 'irstp']
#         self.action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
#         self.feature_names = ['mat', 'net_output', 'year']
#         self.feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#         self.max_depth = 4
#         self.discrete_actions = False
#         self.discrete_features = False
#         # Optimization variables
#         self.mutation_prob = 0.5
#         self.max_nfe = 1000
#
#         # O1 = Organism()
#         # O2 = Organism()
#         # O1.dna = self.random_tree()
#         # O1.fitness = self.policy_tree_RICE_fitness(O1.dna)
#         #
#         # O2.dna = self.random_tree()
#         # O2.fitness = self.policy_tree_RICE_fitness(O2.dna)
#         #
#         # print(O1.dna)
#         # print(O2.dna)
#
#         self.iterate()
#
#         df_graveyard = self.turn_to_dataframe(self.graveyard)
#         df_VIPs = self.turn_to_dataframe(self.VIPs)
#
#         # # Create a pandas dataframe out of the graveyard records
#         # dfs = []
#         # for i in range(len(self.graveyard.keys())):
#         #     # df = pd.DataFrame.from_dict(self.graveyard[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
#         #     df = pd.DataFrame.from_dict(self.graveyard[i], orient='index', columns=['ofv1', 'ofv2'])
#         #     df['policy'] = df.index
#         #     df['generation'] = i
#         #     dfs.append(df)
#         # df = pd.concat(dfs)
#         # df.reset_index(drop=True, inplace=True)
#         # print(df.head)
#         # # df.to_excel('generational_test_single_cluster_graveyard_max_nfe_100_1.xlsx')
#         #
#         # # Create a pandas dataframe out of the VIPs records
#         # dfs = []
#         # for i in range(len(self.VIPs.keys())):
#         #     # df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
#         #     df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2'])
#         #     df['policy'] = df.index
#         #     df['generation'] = i
#         #     dfs.append(df)
#         # df = pd.concat(dfs)
#         # df.reset_index(drop=True, inplace=True)
#         # print(df.head)
#         # # df.to_excel('generational_test_single_cluster_VIPs_max_nfe_100_1.xlsx')
#
#         self.plot(df_graveyard['ofv1'], df_graveyard['ofv2'], df_VIPs['ofv1'], df_VIPs['ofv2'])
#
#     def turn_to_dataframe(self, dict_obj):
#         dfs = []
#         for i in range(len(dict_obj.keys())):
#             # df = pd.DataFrame.from_dict(self.VIPs[i], orient='index', columns=['ofv1', 'ofv2', 'ofv3'])
#             df = pd.DataFrame.from_dict(dict_obj[i], orient='index', columns=['ofv1', 'ofv2'])
#             df['policy'] = df.index
#             df['generation'] = i
#             dfs.append(df)
#         df = pd.concat(dfs)
#         df.reset_index(drop=True, inplace=True)
#         # print(df.head)
#         return df
#
#     def plot(self, x1, y1, x2, y2):
#         # ys = dist_dict_diff
#         # xs = [x for x in range(len(ys))]
#
#         plt.subplot(1, 2, 1)
#         plt.scatter(x1, y1)
#         plt.title('graveyard')
#         plt.ylabel('ofv2 - temp_overshoots')
#         plt.xlabel('ofc1 - period utility')
#
#         plt.subplot(1, 2, 2)
#         plt.scatter(x2, y2)
#         plt.title('VIPs')
#         plt.ylabel('ofv2 - temp_overshoots')
#         plt.xlabel('ofc1 - period utility')
#
#         plt.show()
#         # Make sure to close the plt object once done
#         plt.close()
#         # --------------------------
#
#         # t = np.arange(1000) / 100.
#         # x = np.sin(2 * np.pi * 10 * t)
#         # y = np.cos(2 * np.pi * 10 * t)
#         #
#         # fig = plt.figure()
#         # ax1 = plt.subplot(211)
#         # ax2 = plt.subplot(212)
#         #
#         # ax1.plot(t, x)
#         # ax2.plot(t, y)
#         #
#         # ax1.get_shared_x_axes().join(ax1, ax2)
#         # ax1.set_xticklabels([])
#         # # ax2.autoscale() ## call autoscale if needed
#         #
#         # plt.show()
#         pass
#
#     def iterate(self):
#         nfe = 0
#         generation = 0
#         while nfe < self.max_nfe:
#
#             # Swipe generational variables
#             self.parents = []
#             self.children = []
#             self.family = []
#
#             self.populate()
#             if generation == 0:
#                 self.natural_selection(self.family)
#             else:
#                 print(f'number of non_dominated solutions: {len(self.non_dominated)}')
#                 self.natural_selection(self.non_dominated)
#
#             # determine position of parents in solution space
#             self.center_position.append(self.determine_center_position(self.parents[0].fitness, self.parents[1].fitness))
#
#             nfe += len(self.family)
#
#             # Record all organisms per generation
#             graveyard_dict = {}
#             for member in self.family:
#                 graveyard_dict[str(member.dna)] = member.fitness
#             self.graveyard[generation] = graveyard_dict
#
#             # Record the non dominated organisms per generation
#             VIPs_dict = {}
#             for member in self.non_dominated:
#                 VIPs_dict[str(member.dna)] = member.fitness
#             self.VIPs[generation] = VIPs_dict
#
#             generation += 1
#
#             # print(f'family: {len(self.family)}')
#             # print(f'nfe: {nfe}')
#             # print(f'generation: {generation}')
#         # print(self.graveyard.keys())
#         return
#
#     def random_tree(self, terminal_ratio=0.5,
#                     # discrete_actions=True,
#                     # discrete_features=None,
#                     ):
#
#         num_features = len(self.feature_names)
#         num_actions = len(self.action_names)  # SD changed
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
#                     # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                     a = np.random.choice(num_actions)  # SD changed
#                     action_name = self.action_names[a]
#                     action_value = np.random.uniform(*self.action_bounds[a])
#                     action_input = f'{action_name}_{action_value}'
#                     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     L.append([action_input])  # SD changed
#
#             else:
#                 x = np.random.choice(num_features)
#                 v = np.random.uniform(*self.feature_bounds[x])
#                 L.append([x, v])
#                 S += [current_depth + 1] * 2
#
#         T = PTree(L, self.feature_names, self.discrete_features)
#         T.prune()
#         return T
#
#     # def policy_tree_RICE_fitness(self, T):
#     #     m1, m2, m3 = self.model.POT_control(T)
#     #     # print(m1, m2, m3, T)
#     #     return [m1, m2, m3]
#
#     def policy_tree_RICE_fitness(self, T):
#         m1, m2 = self.model.POT_control(T)
#         # print(m1, m2, m3, T)
#         return [m1, m2]
#
#     def populate(self):
#         # If there are no parents, create them
#         if not self.non_dominated:
#             for _ in range(self.num_parents):
#                 parent = Organism()
#                 parent.dna = self.random_tree()
#                 parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
#                 self.parents.append(parent)
#
#         # If there are possible parents, choose two and let the rest be children
#         elif self.non_dominated:
#             if len(self.non_dominated) >= 2:
#                 P1, P2 = np.random.choice(
#                     self.non_dominated, 2, replace=False)
#                 self.parents.append(P1)
#                 self.parents.append(P2)
#
#                 # Take the other non_dominated solutions as children
#                 self.children = [i for i in self.non_dominated if i not in self.parents]
#                 # Chuck two(?)/ half out if every solution was non-dominated i.e. len(self.non_dominated) >= 8
#                 if len(self.non_dominated) >= self.num_children+self.num_parents:  # 8
#                     idx = self.num_children-2
#                     self.children = self.children[idx:]  # 6:
#             # Else if there is only 1 suitable parent, choose it (ofcourse) and create a random other parent
#             else:
#                 self.parents = [self.non_dominated[0]]
#                 # Create a random other parent
#                 parent = Organism()
#                 parent.dna = self.random_tree()
#                 parent.fitness = self.policy_tree_RICE_fitness(parent.dna)
#                 self.parents.append(parent)
#
#             # TODO:: In case every family member is non dominated, there is no more progression, what to do?
#         self.family.extend(self.parents)
#         self.family.extend(self.children)
#
#         # print(f'parents: {len(self.parents)}')
#         # print(f'children: {len(self.children)}')
#         # print(f'family: {len(self.family)}')
#
#         # for _ in range(self.num_children):
#         while len(self.children) < self.num_children:
#             # print(len(self.children))
#             child = Organism()
#             P1, P2 = np.random.choice(
#                 self.parents, 2, replace=False)
#             child.dna = self.crossover(P1.dna, P2.dna)[0]
#
#             # bloat control
#             while child.dna.get_depth() > self.max_depth:
#                 child.dna = self.crossover(P1.dna, P2.dna)[0]
#
#             # Mutate (with probability of mutation accounted for in function)
#             child.dna = self.mutate(child.dna)
#             child.dna.prune()
#
#             # Welcome child to family
#             child.fitness = self.policy_tree_RICE_fitness(child.dna)
#             self.children.append(child)
#             self.family.append(child)
#
#     def natural_selection(self, list_obj):
#         self.non_dominated = []
#
#         # for member in self.family:
#         #     print(member.fitness)
#         # Get all possible combinations in family
#         for organism_combo in itertools.permutations(list_obj, 2):
#         # for organism_combo in itertools.permutations(self.family, 2):
#             # print(organism_combo)
#             if self.dominates(organism_combo[0].fitness, organism_combo[1].fitness):
#                 # print(f'{organism_combo[0]} dominates {organism_combo[1]}')
#                 self.non_dominated.append(organism_combo[0])
#         return
#
#     def dominates(self, a, b):
#         # assumes minimization
#         # a dominates b if it is <= in all objectives and < in at least one
#         # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
#         # So to circumvent this but still allow multiobjective optimisation in different directions under the
#         # constraint that every number is positive, just add a large number to every index.
#         large_number = 1000000000
#
#         a = np.array(a)
#         a = a + large_number
#
#         b = np.array(b)
#         b = b + large_number
#         # print(f'a: {a}')
#         # print(f'b: {b}')
#         return (np.all(a <= b) and np.any(a < b))
#
#     def crossover(self, P1, P2):
#         P1, P2 = [copy.deepcopy(P) for P in (P1, P2)]
#         # should use indices of ONLY feature nodes
#         feature_ix1 = [i for i in range(P1.N) if P1.L[i].is_feature]
#         feature_ix2 = [i for i in range(P2.N) if P2.L[i].is_feature]
#         index1 = np.random.choice(feature_ix1)
#         index2 = np.random.choice(feature_ix2)
#         slice1 = P1.get_subtree(index1)
#         slice2 = P2.get_subtree(index2)
#         P1.L[slice1], P2.L[slice2] = P2.L[slice2], P1.L[slice1]
#         P1.build()
#         P2.build()
#         return (P1, P2)
#
#     def mutate(self, P, mutate_actions=True):
#         P = copy.deepcopy(P)
#
#         for item in P.L:
#             if np.random.rand() < self.mutation_prob:
#                 if item.is_feature:
#                     low, high = self.feature_bounds[item.index]
#                     if item.is_discrete:
#                         item.threshold = np.random.randint(low, high + 1)
#                     else:
#                         item.threshold = self.bounded_gaussian(
#                             item.threshold, [low, high])
#                 elif mutate_actions:
#                     if self.discrete_actions:
#                         item.value = str(np.random.choice(self.action_names))
#                     else:
#                         # print(item)
#                         # print(self.action_bounds)
#                         # print(item.value)
#                         # item.value = self.bounded_gaussian(
#                         #     item.value, self.action_bounds)
#
#                         a = np.random.choice(len(self.action_names))  # SD changed
#                         action_name = self.action_names[a]
#                         action_value = np.random.uniform(*self.action_bounds[a])
#                         action_input = f'{action_name}_{action_value}'
#                         # print(action_input)
#                         item.value = action_input
#
#         return P
#
#     def bounded_gaussian(self, x, bounds):
#         # do mutation in normalized [0,1] to avoid sigma scaling issues
#         lb, ub = bounds
#         xnorm = (x - lb) / (ub - lb)
#         x_trial = np.clip(xnorm + np.random.normal(0, scale=0.1), 0, 1)
#
#         return lb + x_trial * (ub - lb)
#
#     def determine_center_position(self, P1, P2):
#         num_dimensions = len(P1)
#         center_position = []
#         for dimension in range(num_dimensions):
#             avg_pos = P1[dimension] + ((P2[dimension]-P1[dimension])/2)
#             center_position.append(avg_pos)
#         return center_position
#
#
#         # Step 1: select best performing parent -> create dominates() function
#         # Step 2: let other parents mutate -> if mutation dominates over
#
#         # -----------
#         # create children by cross-over
#         # let all within a population mutate -> keep the better performing version
#         # Choose the best number of organisms equal to num_parents and choose them as parent clusters -> repeat process
#
#         # ----------
#         # When a cluster is not performing better, keep its solutions and create a new (random) cluster.
#         # When the fitness of two clusters starts to approach each other, freeze one and let the other grow
#
#
# class Organism:
#     def __init__(self):
#         self.dna = None
#         self.fitness = None
#
#
# Cluster(2, 8)
#
#
# class ClusterOpt:
#     def __init__(self):
#         # np.random.seed(time.perf_counter())
#         self.C1 = Cluster(2, 8)
#         # np.random.seed(time.perf_counter())
#         self.C2 = Cluster(2, 8)
#
#         print(self.C1.center_position)
#         print(self.C1.center_position)
#
#         print(self.distance(self.C1.center_position[0], self.C2.center_position[0]))
#
#     def distance(self, P1, P2):
#         # Input is list
#         num_dimensions = len(P1)
#         dist = []
#         for dimension in range(num_dimensions):
#             dist_ = (P2[dimension] - P1[dimension]) ** 2
#             dist.append(dist_)
#         distance = math.sqrt(sum(dist))
#         return distance
#         # return math.sqrt(((P2[0] - P1[0]) ** 2) + ((P2[1] - P1[1]) ** 2) + ((P2[2] - P1[2]) ** 2))
#
#
# # ClusterOpt()

# -------------------------------------------------------------------------------------------------------
# O1 = Organism()
# O2 = Organism()
#
# print(O1.dna)
# print(O2.dna)

    # def create(self):
    #     self.dna = self.genetic_blueprint.random_tree()
    #     pass
    #
    # def live_and_die(self):
    #     self.fitness = self.genetic_blueprint.policy_tree_RICE_fitness(self.dna)
    #     pass


# class Parent(Organism):
#     pass
#
#
# class Child(Organism):
#     pass
#
#
# def random_tree(self, terminal_ratio=0.5,
#                 # discrete_actions=True,
#                 # discrete_features=None,
#                 ):
#
#     num_features = len(feature_names)
#     num_actions = len(action_names) # SD changed
#
#     depth = np.random.randint(1, max_depth + 1)
#     L = []
#     S = [0]
#
#     while S:
#         current_depth = S.pop()
#
#         # action node
#         if current_depth == depth or (current_depth > 0 and \
#                                       np.random.rand() < terminal_ratio):
#             if self.discrete_actions:
#                 L.append([str(np.random.choice(action_names))])
#             else:
#                 # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                 a = np.random.choice(num_actions)  # SD changed
#                 action_name = action_names[a]
#                 action_value = np.random.uniform(*action_bounds[a])
#                 action_input = f'{action_name}_{action_value}'
#                 # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                 L.append([action_input])  # SD changed
#
#         else:
#             x = np.random.choice(num_features)
#             v = np.random.uniform(*feature_bounds[x])
#             L.append([x, v])
#             S += [current_depth + 1] * 2
#
#     T = PTree(L, feature_names, discrete_features)
#     T.prune()
#     return T
#
#
# class PolicyTree:
#     def __init__(self, model, action_names,  action_bounds, discrete_actions, feature_names, feature_bounds, discrete_features, epsilon=0.1, max_nfe=1000, max_depth=4, population_size=100):
#         self.model = model
#         # Tree variables
#         self.action_names = action_names
#         self.action_bounds = action_bounds
#         self.discrete_actions = discrete_actions
#         self.feature_names = feature_names
#         self.feature_bounds = feature_bounds
#         self.discrete_features = discrete_features
#         self.max_depth = max_depth
#         # Optimization variables
#         self.epsilon = epsilon
#         self.max_nfe = max_nfe
#         self.population_size = population_size
#
#         T1 = self.random_tree()
#         T2 = self.random_tree()
#         print(T1)
#         print(T2)
#
#     def random_tree(self, terminal_ratio=0.5,
#                     # discrete_actions=True,
#                     # discrete_features=None,
#                     ):
#
#         num_features = len(self.feature_names)
#         num_actions = len(self.action_names) # SD changed
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
#                     # TODO:: actions are not mutually exclusive, make it so that multiple actions can be activated by the same leaf node
#                     a = np.random.choice(num_actions)  # SD changed
#                     action_name = self.action_names[a]
#                     action_value = np.random.uniform(*self.action_bounds[a])
#                     action_input = f'{action_name}_{action_value}'
#                     # L.append([np.random.uniform(*self.action_bounds[a])])  # SD changed
#                     L.append([action_input])  # SD changed
#
#             else:
#                 x = np.random.choice(num_features)
#                 v = np.random.uniform(*self.feature_bounds[x])
#                 L.append([x, v])
#                 S += [current_depth + 1] * 2
#
#         T = PTree(L, self.feature_names, self.discrete_features)
#         T.prune()
#         return T
#
#     def policy_tree_RICE_fitness(self, T):
#         m1, m2, m3 = self.model.POT_control(T)
#         # print(m1, m2, m3, T)
#         return m1, m2, m3
#
#
# # Cluster(num_parents=2, num_children=5).populate()
#
# # Model variables
# years_10 = []
# for i in range(2005, 2315, 10):
#     years_10.append(i)
#
# regions = [
#     "US",
#     "OECD-Europe",
#     "Japan",
#     "Russia",
#     "Non-Russia Eurasia",
#     "China",
#     "India",
#     "Middle East",
#     "Africa",
#     "Latin America",
#     "OHI",
#     "Other non-OECD Asia",
# ]
# # Tree variables
# # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
# action_names = ['miu', 'sr', 'irstp']
# action_bounds = [[2100, 2250], [0.2, 0.5], [0.01, 0.1]]
# feature_names = ['mat', 'net_output', 'year']
# feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
# # Save variables
# # database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
# # table_name_POT = 'Test1_couplingborg_not_edited_borg'
#
# PolicyTree(model=RICE(years_10, regions),
#                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
#                 action_names=action_names,
#                 action_bounds=action_bounds,
#                 discrete_actions=False,
#                 feature_names=feature_names,
#                 feature_bounds=feature_bounds,
#                 discrete_features=False,
#                 epsilon=0.1,
#                 max_nfe=6,
#                 max_depth=4,
#                 population_size=3)
#
# # T1 = PolicyTree(model=RICE(years_10, regions),
# #                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
# #                 action_names=action_names,
# #                 action_bounds=action_bounds,
# #                 discrete_actions=False,
# #                 feature_names=feature_names,
# #                 feature_bounds=feature_bounds,
# #                 discrete_features=False,
# #                 epsilon=0.1,
# #                 max_nfe=6,
# #                 max_depth=4,
# #                 population_size=3).random_tree()
# # T2 = PolicyTree(model=RICE(years_10, regions),
# #                 # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
# #                 action_names=action_names,
# #                 action_bounds=action_bounds,
# #                 discrete_actions=False,
# #                 feature_names=feature_names,
# #                 feature_bounds=feature_bounds,
# #                 discrete_features=False,
# #                 epsilon=0.1,
# #                 max_nfe=6,
# #                 max_depth=4,
# #                 population_size=3).random_tree()
# #
# # print(T1)
# # print(T2)

