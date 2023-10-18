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
        self.pareto_front = np.array([self.spawn()])

    def random_tree(self, terminal_ratio=0.5,
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

    def spawn(self):
        organism = Organism()
        organism.dna = self.random_tree()
        organism.fitness = self.policy_tree_RICE_fitness(organism.dna)
        return organism

    def policy_tree_RICE_fitness(self, T):
        metrics = np.array(self.model.POT_control_Herman(T))
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

    def selection(self, candidate):
        for member in self.pareto_front:
            if self.dominates(candidate.fitness, member.fitness):
                # Remove member from pareto front
                self.pareto_front = self.pareto_front[~np.isin(self.pareto_front, member)]
            elif self.dominates(member.fitness, candidate.fitness):
                return
        self.pareto_front = np.append(self.pareto_front, candidate)
        return

    def run(self):
        nfe = 0
        while nfe < self.max_nfe:
            organism = self.spawn()
            self.selection(organism)
            if nfe%1000 == 0:
                print(f'nfe: {nfe}')
            nfe += 1
        pf_dict = {}
        for member in self.pareto_front:
            pf_dict[member.dna] = [member.fitness[0], member.fitness[1], member.fitness[2]]
        df = pd.DataFrame.from_dict(pf_dict, orient='index')
        return df


class Organism:
    def __init__(self):
        self.dna = None
        self.fitness = None
        self.operator = None
