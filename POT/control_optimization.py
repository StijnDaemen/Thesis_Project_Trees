from POT.tree import PTree
import numpy as np
import pandas as pd


class PolicyTreeOptimizerControl:
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
        self.pareto_front = []
        self.pareto_front_trees = []
        # self.epsilon = epsilon
        self.max_nfe = max_nfe
        # self.population_size = population_size

    def random_tree(self, terminal_ratio=0.5,
                    # discrete_actions=True,
                    # discrete_features=None,
                    ):
        num_features = len(self.feature_names)
        num_actions = len(self.action_names)  # SD changed

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

    def RICE_evaluation(self):
        T = self.random_tree()
        m1, m2, m3 = self.model.POT_control_Herman(T)
        # print(m1, m2, m3, T)
        sol_dict = {T: [m1, m2, m3]}
        return sol_dict

    def add_to_pareto_front(self, candidate, tree):
        for i, member_candidate in enumerate(candidate):
            for j, member_established in enumerate(self.pareto_front):
                if self.dominates(member_candidate.fitness, member_established.fitness):
                    self.pareto_front.pop(j)
                    self.pareto_front.append(member_candidate)
                    self.pareto_front_trees.pop(j)
                    self.pareto_front_trees.append(tree)

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

    def run(self):
        nfe = 0
        while nfe <= self.max_nfe:
            evaluation = self.RICE_evaluation()
            solution = evaluation.values()
            tree = evaluation.keys()
            if nfe == 0:
                self.pareto_front = solution
            else:
                self.add_to_pareto_front(solution, tree)
            nfe += 1

        df = pd.DataFrame(self.pareto_front, index=self.pareto_front_trees,
                          columns=['pareto_front'])
        # df.to_excel('control_run_pareto_front.xlsx')
        return df
