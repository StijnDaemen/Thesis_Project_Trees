import math
import matplotlib.pyplot as plt
import hvwfg
# import pygmo as pg

# Import the generative model by SD
# Import the policy tree optimizer by Herman
# Import the policy tree optimizer with borg
# Import the control - random search - 'optimizer'
# Import the ema workbench by professor Kwakkel
# Import the homemade POT optimizer

import pickle

import pandas as pd
import numpy as np
import sqlite3
import os

import networkx as nx
import matplotlib.patches as patches

from RICE_model.IAM_RICE import RICE

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def Pickle(file_path):
        with open(file_path, "rb") as file:
            snapshots = pickle.load(file)
        return snapshots

def calculate_generational_hypervolume(pareto_front_generations, reference_point):
        # def hypervolume(front, reference_point):
        #     """
        #     Calculate the hypervolume metric for a set of solutions in multi-objective optimization.
        #
        #     Parameters:
        #     front (list of lists): A list of objective vectors for each solution.
        #     reference_point (list): The reference point for the hypervolume calculation.
        #
        #     Returns:
        #     hypervolume_value (float): The hypervolume metric value.
        #     """
        #
        #     # Convert the input to NumPy arrays for efficient calculations
        #     front = np.array(front)
        #     reference_point = np.array(reference_point)
        #
        #     # Initialize the hypervolume value
        #     hypervolume_value = 0.0
        #
        #     # Iterate through each solution in the front
        #     for solution in front:
        #         # Calculate the hypervolume contribution of each solution
        #         contribution = np.prod(np.maximum(reference_point - solution, 0))
        #
        #         # Update the total hypervolume value
        #         hypervolume_value += contribution
        #
        #     return hypervolume_value

        # hypervolume_metric = np.array([])
        # for generation in pareto_front_generations:
        #     hypervolume_metric = np.append(hypervolume_metric, hypervolume(generation, reference_point))

        # def calculate_hypervolume(solutions, reference_point):
        #     """
        #     Calculate the hypervolume of a non-dominated set of solutions.
        #
        #     :param solutions: A list of solutions, where each solution is itself a list of objectives.
        #     :param reference_point: A worst-case point that is dominated by all the solutions.
        #     :return: The hypervolume enclosed by the solution set.
        #     """
        #     # Sort the solution set by the first objective in descending order
        #     sorted_solutions = sorted(solutions, key=lambda x: x[0], reverse=True)
        #
        #     # Initialize hypervolume
        #     hypervolume = 0.0
        #
        #     # Process each solution
        #     for i, solution in enumerate(sorted_solutions):
        #         # Distance to the next solution in the first objective or to the reference point if it's the last solution
        #         if i < len(sorted_solutions) - 1:
        #             width = sorted_solutions[i + 1][0] - solution[0]
        #         else:
        #             width = reference_point[0] - solution[0]
        #
        #         # Calculate the contribution of the current solution
        #         contribution = width
        #         for obj_index in range(1, len(solution)):
        #             contribution *= solution[obj_index] - reference_point[obj_index]
        #
        #         hypervolume += contribution
        #
        #     return hypervolume

        hypervolume_metric = np.array([])
        for generation in pareto_front_generations:
            generation = np.array(generation)
            hypervolume_metric = np.append(hypervolume_metric, hvwfg.wfg(generation, reference_point))
            # hypervolume_metric = np.append(hypervolume_metric, calculate_hypervolume(generation, reference_point))


        # hypervolume_metric = np.array([])
        # for generation in pareto_front_generations:
        #     hv = pg.hypervolume(generation)
        #     hypervolume_metric = np.append(hv.compute(reference_point, hv_algo=pg.hvwfg()))

        return hypervolume_metric


def find_last_occurrence(arr):
    indices = np.where(arr == arr[0])[0]  # Get all indices where the value matches the first element
    if indices.size > 1:  # If there are more occurrences than just the first one
        return indices[-1]  # Return the last occurrence
    return None


def indicators_actions_analysis(df, feature_names, action_names, save_name=None):
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
                if act: # extra if statement because last split can be empty
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
    for idx, indicator_name in enumerate(feature_names):
        plt.subplot(2, 3, idx+1)
        y = np.array(indicator_dict[indicator_name])
        plt.hist(y)
        plt.title(indicator_name)

    for idx, action_name in enumerate(action_names):
        plt.subplot(2, 3, idx+4)
        y = np.array(action_dict[action_name])
        plt.hist(y)
        plt.title(action_name)

    if save_name:
        plt.savefig(save_name, dpi=300)

    # plt.show()
    plt.close()
    print('closed plot')
    return


def visualize_organisms_objective_space(organisms, title=None, x_label=None, y_label=None, z_label=None, save=False):
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

    # return plt


def visualize_tree(root):
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
    return


if __name__ == '__main__':
    # # file_name = 'ForestborgRICE_100000nfe_seed_42_snapshots'
    # # print(path_to_dir+f'\output_data\{file_name}.pkl')
    # # snapshots = Pickle(path_to_dir+f'\output_data\{file_name}.pkl')
    # # print(snapshots.keys())
    # # print(snapshots['Archive_solutions'][-1])
    # # print(len(snapshots['Archive_solutions'][-1]))
    #
    # # file_name = 'TESTFB_Archive_platypus_snapshots'
    # # file_name = 'TESTFB_Archive_platypus50000nfe_eps_050101_snapshots'
    # # file_name = 'TESTFB_Archive_platypus50000nfe_eps_005005005_snapshots'
    # # file_name = 'TRY_FB_RICE_large_random_init_pop_10000_plus_20000fe_snapshots'
    # # file_name = 'TRY_FB_RICE_large_random_init_pop_10000_plus_20000fe_discrete_actions_snapshots'
    # # file_name = 'TRY_FB_RICE_30000fe_discrete_actions_restart_copy_snapshots'
    # file_name = 'TRY_FB_RICE_30000fe_discrete_actions_restart_copy_no_escape_latch_snapshots'
    # # file_name = 'TESTFB_Archive_platypus50000nfe_eps_005005005_no_restart_snapshots'
    # # file_name = 'TESTFB_Archive_platypus_no_restart_1000nfe_snapshots'
    # # file_name = 'HermanRICE_100000nfe_random_other_levers_seed_5_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #
    # # print(len(snapshots['Archive_solutions'][28]))
    # # print(len(snapshots['Archive_solutions'][29]))
    # #
    # # PF_size = []
    # # for gen in snapshots['Archive_solutions']:
    # #     PF_size.append(len(gen))
    # #
    # # plt.figure(figsize=(10, 6))
    # # plt.plot(PF_size)
    # # plt.show()
    # #
    # # # ---
    # # def arrays_not_in_list(array_list1, array_list2, tol=1e-5):
    # #     # Initialize an empty list to store unique arrays
    # #     unique_arrays = []
    # #
    # #     # Check if each NumPy array from array_list1 exists in array_list2 within a given tolerance
    # #     for arr1 in array_list1:
    # #         if not any(np.allclose(arr1, arr2, atol=tol) for arr2 in array_list2):
    # #             unique_arrays.append(arr1)
    # #
    # #     return unique_arrays
    # #
    # #
    # # # Example usage with NumPy arrays:
    # # list1 = snapshots['Archive_solutions'][28]
    # # list2 = snapshots['Archive_solutions'][29]
    # #
    # # unique_in_list1 = arrays_not_in_list(list1, list2)
    # # print("Arrays in 28 that are not in 29:")
    # # for arr in unique_in_list1:
    # #     print(arr)
    # #
    # # unique_in_list2 = arrays_not_in_list(list2, list1)
    # # print("Arrays in 29 that are not in 28:")
    # # for arr in unique_in_list2:
    # #     print(arr)
    #
    # def dominates(a, b):
    #     # assumes minimization
    #     # a dominates b if it is <= in all objectives and < in at least one
    #     # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
    #     # So to circumvent this but still allow multiobjective optimisation in different directions under the
    #     # constraint that every number is positive, just add a large number to every index.
    #
    #     large_number = 1000000000
    #     a = a + large_number
    #     b = b + large_number
    #
    #     return np.all(a <= b) and np.any(a < b)
    #
    # # for idx in range(len(snapshots['Archive_solutions'])-1):
    # #     for sol0 in snapshots['Archive_solutions'][idx]:
    # #         for sol1 in snapshots['Archive_solutions'][idx+1]:
    # #             if dominates(sol0, sol1):
    # #                 print(sol0, sol1)
    # #             else:
    # #                 print('yep!')
    #
    # # for sol0 in snapshots['Archive_solutions'][0]:
    # #     print(sol0)
    #
    #
    #
    # #
    # # start_last = find_last_occurrence(snapshots['epsilon_progress'])
    # # print(len(snapshots['epsilon_progress'][start_last:]))
    # # snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
    # #
    # # # plt.plot(snapshots['epsilon_progress_metric'], color='blue')
    # # # plt.title('Epsilon Progress Metric')
    # # # plt.xlabel('Index')
    # # # plt.ylabel('Value')
    # # # plt.grid(True)
    # # # plt.show()
    # #
    # reference_point = np.array([-0.001, 5, 10])
    # # Archive_solutions
    # hypervolume_metric = calculate_generational_hypervolume(snapshots['Archive_solutions'], reference_point)
    # # hypervolume_metric = calculate_generational_hypervolume(snapshots['best_f'], reference_point)
    # snapshots['hypervolume_metric'] = hypervolume_metric
    # #
    # # print(len(hypervolume_metric))
    # # print(len(snapshots['nfe']))
    # # print(hypervolume_metric)
    # #
    # # Create a 2x1 grid of subplots (2 rows, 1 column)
    # fig, axs = plt.subplots(1, 2, figsize=(8, 10))
    # #
    # # # Plotting the 'epsilon' series in the first subplot
    # # axs[0].plot(snapshots['epsilon_progress_metric'], color='blue')
    # # axs[0].set_title('Epsilon Series')
    # # axs[0].set_xlabel('Index')
    # # axs[0].set_ylabel('Value')
    # # axs[0].grid(True)
    # #
    # # Plotting the 'hyper' series in the second subplot
    # axs[1].plot(snapshots['hypervolume_metric'], color='red')
    # axs[1].set_title('Hyper Series')
    # axs[1].set_xlabel('Index')
    # axs[1].set_ylabel('Value')
    # axs[1].grid(True)
    # #
    # # Adjust spacing between plots
    # plt.tight_layout()
    # plt.show()
    #
    # # df = pd.DataFrame(data=snapshots['Archive_trees'][-1], columns=['policy'])
    # # # print(df['policy'])
    # # # print(df.info())
    # # # print(df.head())
    # #
    # # feature_names = ['mat', 'net_output', 'year']
    # # action_names = ['miu', 'sr', 'irstp']
    # # indicators_actions_analysis(df, feature_names, action_names)




    # -- Metrics figure -----------

    # # file_names = ['ForestborgRICE_100000nfe_seed_42_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_5_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_17_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_26_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_55_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_104_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_303_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_506_snapshots',
    # #               'ForestborgRICE_100000nfe_seed_902_snapshots']
    #
    # # file_names = [
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_17_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_42_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_104_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_303_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_902_snapshots',
    # # ]
    #
    # # file_names = [
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_5_seed_17_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_5_seed_42_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_5_seed_104_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_5_seed_303_snapshots',
    # #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_5_seed_902_snapshots',
    # # ]
    #
    # file_names = [
    #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_6_seed_17_snapshots',
    #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_6_seed_42_snapshots',
    #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_6_seed_104_snapshots',
    #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_6_seed_303_snapshots',
    #     'FB_RICE_Archive_platypus30000nfe_eps_005005005_with_restart_gamma_4_depth_6_seed_902_snapshots',
    # ]
    # dicts = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     # Metrics analysis
    #     print(snapshots['epsilon_progress'])
    #     # reference_point = np.array([10, 100, 100])
    #     reference_point = np.array([-0.001, 5, 10])
    #     snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions'], reference_point)
    #     snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    #     # start_last = find_last_occurrence(snapshots['epsilon_progress'])
    #     # snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
    #     dicts.append(snapshots)
    #     # # Trees analysis
    #     # df = pd.DataFrame(data=snapshots['Archive_trees'][-1], columns=['policy'])
    #     # feature_names = ['mat', 'net_output', 'year']
    #     # action_names = ['miu', 'sr', 'irstp']
    #     # indicators_actions_analysis(df, feature_names, action_names, save_name=f'{file_name}_ind_act_analysis.png')
    #
    # fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    #
    # for idx, d in enumerate(dicts):
    #     axs[0].plot(d['epsilon_progress_metric'], color='#92D1C3')
    #     # axs[0].set_title('Epsilon progress', fontsize=16)
    #     axs[0].set_xlabel('Index', fontsize=14)
    #     axs[0].set_ylabel('Epsilon progress', fontsize=14)
    #     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    #     axs[1].plot(d['nfe'], d['hypervolume_metric'], color='#A2CFFE')
    #     # axs[1].set_title('Hypervolume', fontsize=16)
    #     axs[1].set_xlabel('nfe', fontsize=14)
    #     axs[1].set_ylabel('Hypervolume', fontsize=14)
    #     axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # plt.title('Hypervolume ForestBORG applied to the RICE model, seed analysis', fontsize=16)
    # # plt.savefig('Hypervolume ForestBORG applied to the RICE model, seed analysis.png', dpi=300)
    #
    # plt.tight_layout()
    # plt.show()

    # ------------

    # # file_name = 'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_17_copyfixrestart_discrete_actions_snapshots'
    # # file_name = 'TESTFB_1000nfe_snapshots'
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_4_seed_42_continuous_actions_scenario_SSP_1_negemissions_no_snapshots'
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_5_seed_17_discrete_actions_snapshots'
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_5_seed_42_discrete_actions_snapshots'
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_4_seed_42_continuous_actions_depth_analysis_snapshots'
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_3_seed_42_continuous_actions_scenario_SSP_1_negemissions_no_fixed_prune_snapshots'
    # file_name = 'HermanRICE_20000nfe_random_other_levers_seed_42_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')

    # # print(snapshots['Archive_solutions'][-1])
    # print(len(snapshots['Archive_solutions'][-1]))
    # # print(snapshots['Archive_trees'][-1][0])
    # print(len(snapshots['Archive_trees'][-1]))

    # visualize_organisms_objective_space(snapshots['Archive_solutions'][-1])
    #
    # for tree in snapshots['Archive_trees'][-1]:
    #     # print(str(tree))
    #     visualize_tree(tree.root)

    # print(snapshots['best_f'][-1])
    # a = snapshots['best_f']
    # b = -snapshots['best_f']

    # a = snapshots['Archive_solutions'][-1]
    # b = snapshots['Archive_solutions'][-1] + snapshots['Archive_solutions'][-1]

    # print(a)
    # print(b)


    # print(np.hstack(snapshots['Archive_solutions'][-1]))
    # print(np.vstack(snapshots['Archive_solutions'][-1]))
    # print(np.dstack(snapshots['Archive_solutions'][-1]))
    # print(np.column_stack(snapshots['Archive_solutions'][-1]))
    #
    # stacked = np.dstack(snapshots['Archive_solutions'][-1])

    # ----------

    # min_vals = []
    # for gen in snapshots['Archive_solutions']:
    #     stacked = np.column_stack(gen)
    #     min_vals.append([np.min(stacked, axis=1)])
    # reference_point = np.min(min_vals, axis=0)[0]
    #
    # snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions'], reference_point)



    #------------------
    # file_name_Herman = 'Folsom_Herman_20000nfe_eps_001_1000_001_10_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
    # seed_PF = snapshots['best_f'][-1]
    # min_vals = []
    # max_vals = []
    # # for file_name in file_names:
    # #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    # #     # seed_PF = seed_PF + snapshots['Archive_solutions'][-1]
    # #     seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
    # stacked = np.column_stack(seed_PF)
    # min_vals.append([np.min(stacked, axis=1)])
    # min_val = np.min(min_vals, axis=0)[0]
    # # reference_point = np.min(min_vals, axis=0)[0]
    # max_vals.append([np.max(stacked, axis=1)])
    # max_val = np.max(max_vals, axis=0)[0]
    # # reference_point = np.max(max_vals, axis=0)[0]
    # # print(reference_point)
    # print(min_val)
    # print(max_val)
    # snapshots['Archive_solutions_normalized'] = []
    # for gen in snapshots['best_f']:
    #     gen_PF = []
    #     for sol in gen:
    #         sol_norm = np.array([(sol - min_val) / (max_val - min_val)])
    #         gen_PF.append(sol_norm)
    #     snapshots['Archive_solutions_normalized'].append(gen_PF)

    # ----------

    # # OFFICIAL SCENARIO ANALYSIS WORST CASE
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
    #
    # file_name = 'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_3_seed_42_continuous_actions_scenario_SSP_5_negemissions_no_fixed_prune_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #
    # Nordhaus = []
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    # dicts = []
    # SSPs = [1, 2, 3, 4, 5]
    # neg_ems = ['no', 'yes']
    # for neg_em in neg_ems:
    #     for SSP in SSPs:
    #         scenario = {'SSP_scenario': SSP,  # 1, 2, 3, 4, 5
    #                      'fosslim': 9000,  # range(4000, 13650), depending on SSP scenario
    #                      'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
    #                      'elasticity_climate_impact': 0,  # -1, 0, 1
    #                      'price_backstop_tech': 1.470,  # [1.260, 1.470, 1.680, 1.890]
    #                      'negative_emissions_possible': neg_em,  # 'yes' or 'no'
    #                      't2xco2_index': 500, }  # 0, 999
    #         Nordhaus.append(RICE(years_10, regions, scenario=scenario, levers=levers).ema_workbench_control())
    #         # RICE = RICE(years_10, regions, scenario=scenario)
    #         performance = []
    #         for tree in snapshots['Archive_trees'][-1]:
    #             performance.append(RICE(years_10, regions, scenario=scenario).POT_control(tree))
    #         dicts.append(performance)
    #
    # fig = plt.figure()
    # xmin = 0
    # xmax = -10
    # ymin = 0
    # ymax = 32
    # zmin = 3
    # zmax = 45
    # axes = [fig.add_subplot(2, 5, i, projection='3d') for i in range(1, 11)]
    #
    # names = ['SSP 1 - 0% neg. emissions',
    #      'SSP 2 - 0% neg. emissions',
    #      'SSP 3 - 0% neg. emissions',
    #      'SSP 4 - 0% neg. emissions',
    #      'SSP 5 - 0% neg. emissions',
    #      'SSP 1 - 20% neg. emissions',
    #      'SSP 2 - 20% neg. emissions',
    #      'SSP 3 - 20% neg. emissions',
    #      'SSP 4 - 20% neg. emissions',
    #      'SSP 5 - 20% neg. emissions'
    #      ]
    # for index, ax in enumerate(axes):
    #     d = dicts[index]
    #     # visualize_organisms_objective_space(_['Archive_solutions'][-1])
    #
    #     organisms = d
    #     data_dict = {}
    #     for objective in range(1, len(organisms[0]) + 1):
    #         data_dict[f'ofv{objective}'] = []
    #
    #     for idx__, item in enumerate(organisms):
    #         for key_idx, key in enumerate(data_dict.keys()):
    #             data_dict[key].append(item[key_idx])
    #
    #     ax.scatter(data_dict['ofv1'],
    #               data_dict['ofv2'],
    #               data_dict['ofv3'])
    #     ax.scatter(Nordhaus[index][0], Nordhaus[index][1], Nordhaus[index][2], color='red', marker='D')
    #     ax.set_xlim([xmin, xmax])
    #     ax.set_ylim([ymin, ymax])
    #     ax.set_zlim([zmin, zmax])
    #     ax.set_title(names[index], fontsize=11)
    #     ax.set_xlabel('utility', fontsize=7)
    #     ax.set_ylabel('damages', fontsize=7)
    #     ax.set_zlabel('temp. overshoots', fontsize=7)
    #     ax.view_init(elev=20.0, azim=145)
    #
    # # plt.tight_layout()
    # # plt.savefig('scenario_analysis_worst_case_10_RICE_FB_after_prune_fix.png', dpi=300)
    # plt.show()
    # # END OFFICIAL

    # --------------

    # # OFFICIAL SCENARIO ANALYSIS 3D PLOTS
    # file_names = []
    # SSPs = [1, 2, 3, 4, 5]
    # neg_ems = ['no', 'yes']
    # for neg_em in neg_ems:
    #     for SSP in SSPs:
    #         file_names.append(f'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_3_seed_42_continuous_actions_scenario_SSP_{SSP}_negemissions_{neg_em}_fixed_prune_snapshots')
    #
    #
    # dicts = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     # Metrics analysis
    #     # reference_point = np.array([10, 100, 100])
    #     reference_point = np.array([-0.001, 5, 10]).astype(np.float64) #  reference_point = np.array([10, 500000, 10, -100])
    #     snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions'],
    #                                                                          reference_point)
    #     snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    #     dicts.append(snapshots)
    #
    # Nordhaus = []
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
    # levers = {'mu_target': 2135,
    #           'sr': 0.248,
    #           'irstp': 0.015}
    # SSPs = [1, 2, 3, 4, 5]
    # neg_ems = ['no', 'yes']
    # for neg_em in neg_ems:
    #     for SSP in SSPs:
    #         scenario = {'SSP_scenario': SSP,  # 1, 2, 3, 4, 5
    #                      'fosslim': 9000,  # range(4000, 13650), depending on SSP scenario
    #                      'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
    #                      'elasticity_climate_impact': 0,  # -1, 0, 1
    #                      'price_backstop_tech': 1.470,  # [1.260, 1.470, 1.680, 1.890]
    #                      'negative_emissions_possible': neg_em,  # 'yes' or 'no'
    #                      't2xco2_index': 500, }  # 0, 999
    #         Nordhaus.append(RICE(years_10, regions, scenario=scenario, levers=levers).ema_workbench_control())
    #         # RICE = RICE(years_10, regions, scenario=scenario)
    #
    # fig = plt.figure()
    # xmin = -2
    # xmax = -12.5
    # ymin = 1
    # ymax = 40
    # zmin = 2.2
    # zmax = 40
    # axes = [fig.add_subplot(2, 5, i, projection='3d') for i in range(1, 11)]
    # # names = ['SSP 1 - 0% neg. emissions',
    # #          'SSP 1 - 20% neg. emissions',
    # #          'SSP 2 - 0% neg. emissions',
    # #          'SSP 2 - 20% neg. emissions',
    # #          'SSP 3 - 0% neg. emissions',
    # #          'SSP 3 - 20% neg. emissions',
    # #          'SSP 4 - 0% neg. emissions',
    # #          'SSP 4 - 20% neg. emissions',
    # #          'SSP 5 - 0% neg. emissions',
    # #          'SSP 5 - 20% neg. emissions'
    # #          ]
    # names = ['SSP 1 - 0% neg. emissions',
    #      'SSP 2 - 0% neg. emissions',
    #      'SSP 3 - 0% neg. emissions',
    #      'SSP 4 - 0% neg. emissions',
    #      'SSP 5 - 0% neg. emissions',
    #      'SSP 1 - 20% neg. emissions',
    #      'SSP 2 - 20% neg. emissions',
    #      'SSP 3 - 20% neg. emissions',
    #      'SSP 4 - 20% neg. emissions',
    #      'SSP 5 - 20% neg. emissions'
    #      ]
    # for index, ax in enumerate(axes):
    #     d = dicts[index]
    #     # visualize_organisms_objective_space(_['Archive_solutions'][-1])
    #
    #     organisms = d['Archive_solutions'][-1]
    #     data_dict = {}
    #     for objective in range(1, len(organisms[0]) + 1):
    #         data_dict[f'ofv{objective}'] = []
    #
    #     for idx__, item in enumerate(organisms):
    #         for key_idx, key in enumerate(data_dict.keys()):
    #             data_dict[key].append(item[key_idx])
    #
    #     ax.scatter(data_dict['ofv1'],
    #               data_dict['ofv2'],
    #               data_dict['ofv3'])
    #     ax.scatter(Nordhaus[index][0], Nordhaus[index][1], Nordhaus[index][2], color='red', marker='D')
    #     ax.set_xlim([xmin, xmax])
    #     ax.set_ylim([ymin, ymax])
    #     ax.set_zlim([zmin, zmax])
    #     ax.set_title(names[index], fontsize=11)
    #     ax.set_xlabel('utility', fontsize=7)
    #     ax.set_ylabel('damages', fontsize=7)
    #     ax.set_zlabel('temp. overshoots', fontsize=7)
    #     ax.view_init(elev=20.0, azim=145)
    #
    # # plt.tight_layout()
    # # plt.savefig('scenario_analysis_10_RICE_FB_after_prune_fix.png', dpi=300)
    # plt.show()
    # # END OFFICIAL

    # -------------

    # # OFFICIAL: figure Herman vs FB Folsom and RICE (discrete actions)
    # fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    # # Folsom ----------------------------------------
    # xminF = 0
    # xmaxF = 20500
    # yminF = 0  # 1.7e08
    # ymaxF = 1  # 8e08
    # # file_names = [
    # #     f'Folsom_ForestBorg_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_17_snapshots',
    # #     f'Folsom_ForestBorg_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_42_snapshots',
    # #     f'Folsom_ForestBorg_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_104_snapshots',
    # #     f'Folsom_ForestBorg_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_303_snapshots',
    # #     f'Folsom_ForestBorg_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_902_snapshots',
    # # ]
    # seeds = [17, 42, 104, 303, 902]
    # file_names = []
    # for seed in seeds:
    #     file_names.append(f'FB_Folsom_20000nfe_eps_001_1000_001_10_gamma_4_depth_4_seed_{seed}_FB_H_comparison_snapshots')
    #
    # # Calculation of minimum and maximum values for each objective
    # file_name_Herman = 'Folsom_Herman_20000nfe_eps_001_1000_001_10_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
    # seed_PF = snapshots['best_f'][-1]
    # min_vals = []
    # max_vals = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
    # stacked = np.column_stack(seed_PF)
    # min_vals.append([np.min(stacked, axis=1)])
    # min_val = np.min(min_vals, axis=0)[0]
    # max_vals.append([np.max(stacked, axis=1)])
    # max_val = np.max(max_vals, axis=0)[0]
    #
    # dicts = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     # Normalize values
    #     snapshots['Archive_solutions_normalized'] = []
    #     for gen in snapshots['Archive_solutions']:
    #         gen_PF = []
    #         for sol in gen:
    #             sol_norm = np.array((sol - min_val) / (max_val - min_val))
    #             gen_PF.append(sol_norm)
    #         snapshots['Archive_solutions_normalized'].append(gen_PF)
    #     # Metrics analysis
    #     # reference_point = np.array([10, 100, 100])
    #     # reference_point = np.array([3, 210000, 5, -300]).astype(np.float64) #  reference_point = np.array([10, 500000, 10, -100])
    #     reference_point = np.array([1, 1, 1, 1]).astype(np.float64)
    #     snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions_normalized'],
    #                                                                          reference_point)
    #     snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    #     dicts.append(snapshots)
    #
    # for idx, d in enumerate(dicts):
    #     if idx == 0:
    #         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#6A7FDB', label='ForestBORG')
    #     else:
    #         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#6A7FDB')
    #     axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
    #     axs[0].set_xlim([xminF, xmaxF])
    #     axs[0].set_ylim([yminF, ymaxF])
    #     axs[0].set_title(f'Folsom model', fontsize=12)
    #     axs[0].set_xlabel('nfe', fontsize=11)
    #     axs[0].set_ylabel('Hypervolume', fontsize=11)
    #     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # file_name_Herman = 'Folsom_Herman_20000nfe_eps_001_1000_001_10_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
    # # reference_point = np.array([3, 210000, 5, -300]).astype(np.float64)
    # # Normalize values
    # snapshots['best_f_normalized'] = []
    # for gen in snapshots['best_f']:
    #     gen_PF = []
    #     for sol in gen:
    #         sol_norm = np.array((sol - min_val) / (max_val - min_val))
    #         gen_PF.append(sol_norm)
    #     snapshots['best_f_normalized'].append(gen_PF)
    #
    # reference_point = np.array([1, 1, 1, 1]).astype(np.float64)
    # snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['best_f_normalized'],
    #                                                                      reference_point)
    # axs[0].plot(snapshots['nfe'], snapshots['hypervolume_metric'], color='#A04550', label='Original POT')
    # axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
    # axs[0].set_xlim([xminF, xmaxF])
    # axs[0].set_ylim([yminF, ymaxF])
    # axs[0].set_title(f'Folsom model', fontsize=12)
    # axs[0].set_xlabel('nfe', fontsize=11)
    # axs[0].set_ylabel('Hypervolume', fontsize=11)
    # axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # # RICE ------------------------------------------
    # xminR = 0
    # xmaxR = 20500
    # yminR = 0  # 360
    # ymaxR = 1  # 420
    # # file_names = [
    # #     f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_17_copyfixrestart_discrete_actions_snapshots',
    # #     f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_42_copyfixrestart_discrete_actions_snapshots',
    # #     f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_104_copyfixrestart_discrete_actions_snapshots',
    # #     f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_303_copyfixrestart_discrete_actions_snapshots',
    # #     f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_4_seed_902_copyfixrestart_discrete_actions_snapshots',
    # # ]
    # seeds = [17, 42, 104, 303, 902]
    # file_names = []
    # for seed in seeds:
    #     file_names.append(
    #         f'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_4_seed_{seed}_discrete_actions_FB_H_comparison_snapshots')
    #
    # # Calculation of minimum and maximum values for each objective
    # file_name_Herman = 'HermanRICE_20000nfe_random_other_levers_seed_42_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
    # seed_PF = snapshots['best_f'][-1]
    # min_vals = []
    # max_vals = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
    # stacked = np.column_stack(seed_PF)
    # min_vals.append([np.min(stacked, axis=1)])
    # min_val = np.min(min_vals, axis=0)[0]
    # max_vals.append([np.max(stacked, axis=1)])
    # max_val = np.max(max_vals, axis=0)[0]
    #
    # dicts = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     # Normalize values
    #     snapshots['Archive_solutions_normalized'] = []
    #     for gen in snapshots['Archive_solutions']:
    #         gen_PF = []
    #         for sol in gen:
    #             sol_norm = np.array((sol - min_val) / (max_val - min_val))
    #             gen_PF.append(sol_norm)
    #         snapshots['Archive_solutions_normalized'].append(gen_PF)
    #     # Metrics analysis
    #     # reference_point = np.array([10, 100, 100])
    #     # reference_point = np.array([-0.001, 5, 10])
    #     # reference_point = np.array([-15.50304053, 0.32949249, 3.71813423])
    #     reference_point = np.array([1, 1, 1]).astype(np.float64)
    #     snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions_normalized'],
    #                                                                          reference_point)
    #     snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    #     dicts.append(snapshots)
    #
    # for idx, d in enumerate(dicts):
    #     if idx == 0:
    #         axs[1].plot(d['nfe'], d['hypervolume_metric'], color='#6A7FDB', label='ForestBORG')
    #     else:
    #         axs[1].plot(d['nfe'], d['hypervolume_metric'], color='#6A7FDB')
    #     axs[1].legend(loc='lower right', fontsize='medium', frameon=True)
    #     # axs[1].legend(loc=(0.70, 0.45), fontsize='medium', frameon=True)
    #     axs[1].set_xlim([xminR, xmaxR])
    #     axs[1].set_ylim([yminR, ymaxR])
    #     axs[1].set_title(f'RICE model', fontsize=12)
    #     axs[1].set_xlabel('nfe', fontsize=11)
    #     axs[1].set_ylabel('Hypervolume', fontsize=11)
    #     axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # file_name_Herman = 'HermanRICE_20000nfe_random_other_levers_seed_42_depth_4_snapshots'
    # snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
    # # Normalize values
    # snapshots['best_f_normalized'] = []
    # for gen in snapshots['best_f']:
    #     gen_PF = []
    #     for sol in gen:
    #         sol_norm = np.array((sol - min_val) / (max_val - min_val))
    #         gen_PF.append(sol_norm)
    #     snapshots['best_f_normalized'].append(gen_PF)
    # # reference_point = np.array([-0.001, 5, 10])
    # reference_point = np.array([1, 1, 1]).astype(np.float64)
    # snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['best_f_normalized'],
    #                                                                      reference_point)
    # axs[1].plot(snapshots['nfe'], snapshots['hypervolume_metric'], color='#A04550', label='Original POT')
    # axs[1].legend(loc='lower right', fontsize='medium', frameon=True)
    # axs[1].set_xlim([xminR, xmaxR])
    # axs[1].set_ylim([yminR, ymaxR])
    # axs[1].set_title(f'RICE model', fontsize=12)
    # axs[1].set_xlabel('nfe', fontsize=11)
    # axs[1].set_ylabel('Hypervolume', fontsize=11)
    # axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # plt.tight_layout()
    # # plt.savefig('Hypervolume ForestBORG and Original POT on Folsom and RICE_after_prune_fix_normalized.png', dpi=300)
    # plt.show()
    # # END OFFICIAL


    # OFFICIAL: Figure hypervolume depth-seed analysis ForestBORG on RICE
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    depths = [2, 3, 4, 5, 6, 7]
    xmin = 0
    xmax = 20500
    ymin = 0  # 120
    ymax = 1  # 220

    for index, depth in enumerate(depths):
        file_names = []
        seeds = [17, 42, 104, 303, 902]
        for seed in seeds:
            file_names.append(f'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_{depth}_seed_{seed}_continuous_actions_depth_analysis_snapshots')

        # Calculation of minimum and maximum values for each objective
        min_vals = []
        max_vals = []
        seed_PF = np.empty((0, 3))
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
        stacked = np.column_stack(seed_PF)
        min_vals.append([np.min(stacked, axis=1)])
        min_val = np.min(min_vals, axis=0)[0]
        max_vals.append([np.max(stacked, axis=1)])
        max_val = np.max(max_vals, axis=0)[0]

    # for index, depth in enumerate(depths):
    #     file_names = [
    #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_17_copyfixrestart_snapshots',
    #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_42_copyfixrestart_snapshots',
    #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_104_copyfixrestart_snapshots',
    #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_303_copyfixrestart_snapshots',
    #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_902_copyfixrestart_snapshots',
    #     ]

    for index, depth in enumerate(depths):
        file_names = []
        seeds = [17, 42, 104, 303, 902]
        for seed in seeds:
            file_names.append(f'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_{depth}_seed_{seed}_continuous_actions_depth_analysis_snapshots')

        # # Calculation of minimum and maximum values for each objective
        # min_vals = []
        # max_vals = []
        # seed_PF = np.empty((0, 3))
        # for file_name in file_names:
        #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
        #     seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
        # stacked = np.column_stack(seed_PF)
        # min_vals.append([np.min(stacked, axis=1)])
        # min_val = np.min(min_vals, axis=0)[0]
        # max_vals.append([np.max(stacked, axis=1)])
        # max_val = np.max(max_vals, axis=0)[0]

        dicts = []
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            # Normalize values
            snapshots['Archive_solutions_normalized'] = []
            for gen in snapshots['Archive_solutions']:
                gen_PF = []
                for sol in gen:
                    sol_norm = np.array((sol - min_val) / (max_val - min_val))
                    # sol_norm = np.negative((sol - min_val) / (max_val - min_val))
                    gen_PF.append(sol_norm)
                snapshots['Archive_solutions_normalized'].append(gen_PF)
            # Metrics analysis
            # reference_point = np.array([10, 100, 100])
            # reference_point = np.array([-0.001, 5, 10])
            reference_point = np.array([1, 1, 1]).astype(np.float64)
            # reference_point = np.array([0, 0, 0]).astype(np.float64)
            snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions_normalized'],
                                                                                 reference_point)
            snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
            # start_last = find_last_occurrence(snapshots['epsilon_progress'])
            # snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
            dicts.append(snapshots)

        pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for idx, d in enumerate(dicts):
            axs[pos[index]].plot(d['nfe'], d['hypervolume_metric'], color='#6A7FDB')  # #A2CFFE
            axs[pos[index]].set_xlim([xmin, xmax])
            axs[pos[index]].set_ylim([ymin, ymax])
            axs[pos[index]].set_title(f'depth {depth}', fontsize=12)
            axs[pos[index]].set_xlabel('nfe', fontsize=11)
            axs[pos[index]].set_ylabel('Hypervolume', fontsize=11)
            axs[pos[index]].grid(True, which='both', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    # plt.savefig('Hypervolume ForestBORG applied to the RICE model seed and depth analysis_fixed_prune_normalized.png', dpi=300)
    plt.show()
    # END OFFICIAL

    # # OFFICIAL: Figure depth-seed analysis ForestBORG on RICE epsilon progress
    # fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    # depths = [2, 3, 4, 5, 6, 7]
    # xmin = 0
    # xmax = 20500
    # ymin = 0
    # ymax = 700
    # # for index, depth in enumerate(depths):
    # #     file_names = [
    # #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_17_copyfixrestart_snapshots',
    # #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_42_copyfixrestart_snapshots',
    # #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_104_copyfixrestart_snapshots',
    # #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_303_copyfixrestart_snapshots',
    # #         f'FB_RICE_Archive_platypus20000nfe_eps_005005005_with_restart_gamma_4_depth_{depth}_seed_902_copyfixrestart_snapshots',
    # #     ]
    # for index, depth in enumerate(depths):
    #     file_names = []
    #     seeds = [17, 42, 104, 303, 902]
    #     for seed in seeds:
    #         file_names.append(
    #             f'FB_RICE_20000nfe_eps_005005005_gamma_4_depth_{depth}_seed_{seed}_continuous_actions_depth_analysis_snapshots')
    #     dicts = []
    #     for file_name in file_names:
    #         snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #         # Metrics analysis
    #         # reference_point = np.array([-0.001, 5, 10])
    #         # snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions'],
    #         #                                                                      reference_point)
    #         snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    #         # start_last = find_last_occurrence(snapshots['epsilon_progress'])
    #         # snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
    #         dicts.append(snapshots)
    #
    #     pos = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    #     for idx, d in enumerate(dicts):
    #         axs[pos[index]].plot(d['nfe'], d['epsilon_progress_metric'], color='#6A7FDB')
    #         axs[pos[index]].set_xlim([xmin, xmax])
    #         axs[pos[index]].set_ylim([ymin, ymax])
    #         axs[pos[index]].set_title(f'depth {depth}', fontsize=12)
    #         axs[pos[index]].set_xlabel('nfe', fontsize=11)
    #         axs[pos[index]].set_ylabel('Epsilon progress', fontsize=11)
    #         axs[pos[index]].grid(True, which='both', linestyle='--', linewidth=0.8)
    #
    # plt.tight_layout()
    # # plt.savefig('Epsilon progress ForestBORG applied to the RICE model seed and depth analysis_fixed_prune.png', dpi=300)
    # plt.show()
    # # END OFFICIAL


# ----------------------------------------------------------------------

# # Calculation of reference point
#     file_name_Herman = 'Folsom_Herman_20000nfe_eps_001_1000_001_10_depth_4_snapshots'
#     snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
#     seed_PF = -snapshots['best_f'][-1]
#     min_vals = []
#     max_vals = []
#     for file_name in file_names:
#         snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
#         # seed_PF = seed_PF + snapshots['Archive_solutions'][-1]
#         seed_PF = np.concatenate((seed_PF, np.negative(snapshots['Archive_solutions'][-1])), axis=0)
#     stacked = np.column_stack(seed_PF)
#     min_vals.append([np.min(stacked, axis=1)])
#     min_val = np.min(min_vals, axis=0)[0]
#     # reference_point = np.min(min_vals, axis=0)[0]
#     max_vals.append([np.max(stacked, axis=1)])
#     max_val = np.max(max_vals, axis=0)[0]
#     # reference_point = np.max(max_vals, axis=0)[0]
#     # print(reference_point)
#     print(min_val)
#     print(max_val)


# class ProcessResults:
#     def __init__(self):
#         return
#
#     def Pickle(self, file_path):
#         # file_path = fr'{filepath}\{file_name}'
#         with open(file_path, "rb") as file:
#             snapshots = pickle.load(file)
#         return snapshots
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
#     def calculate_centroid(self, points):
#         # if not points:
#         #     raise ValueError("Points list is empty")
#
#         # Determine the dimensionality by checking the length of the first point
#         dimension = len(points[0])
#
#         # Initialize sums for each dimension
#         sums = [0.0] * dimension
#
#         # Calculate the sum of each dimension for all points
#         for point in points:
#             if len(point) != dimension:
#                 raise ValueError("All points must have the same dimension")
#             sums = [sums[i] + point[i] for i in range(dimension)]
#
#         # Calculate the average for each dimension
#         centroid = [sums[i] / len(points) for i in range(dimension)]
#
#         return centroid
#
#     def visualize_generational_series(self, series, title='centroid distance', x_label='snapshot', y_label='distance (-)', save=False):
#         x = [x for x in range(len(series))]
#         y = series
#         plt.plot(x, y)
#         plt.title(title)
#         plt.ylabel(y_label)
#         plt.xlabel(x_label)
#
#         if save:
#             plt.savefig(f'{self.save_location}/{title}.png', bbox_inches='tight')
#         else:
#             plt.show()
#         # Make sure to close the plt object once done
#         plt.close()
#
#     def view_sqlite_database(self, database, table_name):
#         # df = pd.DataFrame()
#         conn = sqlite3.connect(database)
#         # c = conn.cursor()
#
#         # c.execute(f"""SELECT count(*) FROM sqlite_master WHERE type='table' AND name={table_name}""")
#         df = pd.read_sql_query(f'''SELECT * FROM {table_name}''', conn)
#
#         conn.commit()
#         conn.close()
#
#         return df
#
#     def list_table_names_in_db(self, database):
#         try:
#
#             # Making a connection between sqlite3
#             # database and Python Program
#             sqliteConnection = sqlite3.connect(database)
#
#             # If sqlite3 makes a connection with python
#             # program then it will print "Connected to SQLite"
#             # Otherwise it will show errors
#             print("Connected to SQLite")
#
#             # Getting all tables from sqlite_master
#             sql_query = """SELECT name FROM sqlite_master
#             WHERE type='table';"""
#
#             # Creating cursor object using connection object
#             cursor = sqliteConnection.cursor()
#
#             # executing our sql query
#             cursor.execute(sql_query)
#             print("List of tables\n")
#
#             # printing all tables list
#             print(cursor.fetchall())
#
#         except sqlite3.Error as error:
#             print("Failed to execute the above query", error)
#
#         finally:
#
#             # Inside Finally Block, If connection is
#             # open, we need to close it
#             if sqliteConnection:
#                 # using close() method, we will close
#                 # the connection
#                 sqliteConnection.close()
#
#                 # After closing connection object, we
#                 # will print "the sqlite connection is
#                 # closed"
#                 print("the sqlite connection is closed")
#
#     def dominates(self, a, b):
#         # assumes minimization
#         # a dominates b if it is <= in all objectives and < in at least one
#         # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
#         # So to circumvent this but still allow multiobjective optimisation in different directions under the
#         # constraint that every number is positive, just add a large number to every index.
#
#         large_number = 1000000000
#         a = a + large_number
#         b = b + large_number
#
#         return np.all(a <= b) and np.any(a < b)
#
#     def calculate_generational_distance(self, pareto_front_generations):
#         # Input: array/ array of arrays
#         def closest_distance_to_points(target_point, points_collection):
#             """
#             Calculate the closest distance from a target point to a collection of other points in multi-dimensional space.
#
#             Parameters:
#             target_point (numpy.ndarray): The target point for which you want to find the closest distance.
#             points_collection (numpy.ndarray): A 2D numpy array where each row represents a point in multi-dimensional space.
#
#             Returns:
#             float: The closest distance from the target_point to any point in the collection.
#             """
#             # Calculate the Euclidean distances between the target_point and all points in the collection
#             distances = np.linalg.norm(points_collection - target_point, axis=1)
#
#             # Find the index of the point with the minimum distance
#             closest_point_index = np.argmin(distances)
#
#             # Get the closest distance
#             closest_distance = distances[closest_point_index]
#
#             return closest_distance
#
#         generational_distance = np.array([])
#         for idx, generation in enumerate(pareto_front_generations[:-1]):
#             gen_distance = 0
#             gen_0 = generation
#             gen_1 = pareto_front_generations[idx + 1]
#             for sol in gen_0:
#                 gen_distance += closest_distance_to_points(sol, gen_1)
#             generational_distance = np.append(generational_distance, gen_distance)
#         return generational_distance
#
#     def calculate_generational_hypervolume(self, pareto_front_generations, reference_point):
#         def hypervolume(front, reference_point):
#             """
#             Calculate the hypervolume metric for a set of solutions in multi-objective optimization.
#
#             Parameters:
#             front (list of lists): A list of objective vectors for each solution.
#             reference_point (list): The reference point for the hypervolume calculation.
#
#             Returns:
#             hypervolume_value (float): The hypervolume metric value.
#             """
#
#             # Convert the input to NumPy arrays for efficient calculations
#             front = np.array(front)
#             reference_point = np.array(reference_point)
#
#             # Initialize the hypervolume value
#             hypervolume_value = 0.0
#
#             # Iterate through each solution in the front
#             for solution in front:
#                 # Calculate the hypervolume contribution of each solution
#                 contribution = np.prod(np.maximum(reference_point - solution, 0))
#
#                 # Update the total hypervolume value
#                 hypervolume_value += contribution
#
#             return hypervolume_value
#
#         hypervolume_metric = np.array([])
#         for generation in pareto_front_generations:
#             hypervolume_metric = np.append(hypervolume_metric, hypervolume(generation, reference_point))
#         return hypervolume_metric
#
#     def visualize_generational_metrics(self, series, x_data=None, ax=None, color=None, title='centroid distance', x_label='snapshot', y_label='distance (-)', save=False):
#         if not x_data:
#             x = [x for x in range(len(series))]
#         else:
#             x = x_data
#         y = series
#         ax.plot(x, y, color=color)
#         ax.set_title(title)
#         ax.set_ylabel(y_label)
#         ax.set_xlabel(x_label)
#
#
# if __name__ == '__main__':
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_25000nfe_snapshots.pkl'
#     # data_H = ProcessResults().Pickle(file_path)
#     #
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_w_snapshots_snapshots.pkl'
#     # data_FB = ProcessResults().Pickle(file_path)
#     # centroid_list = []
#     # for generation_f in data_H['best_f']:
#     #     centroid = ProcessResults().calculate_centroid(generation_f)
#     #     centroid_list.append(centroid)
#     #
#     # distance_list = []
#     # for i in range(len(centroid_list)-1):
#     #     dist = ProcessResults().distance(centroid_list[i+1], centroid_list[i])
#     #     distance_list.append(dist)
#     #
#     # print(distance_list)
#     #
#     # ProcessResults().visualize_generational_series(distance_list, title='centroid distance H', x_label='snapshot', y_label='distance (-)')
#     # ----------------------------
#     #
#     # centroid_list = []
#     # for generation_f in data_FB['Archive_solutions']:
#     #     centroid = ProcessResults().calculate_centroid(generation_f)
#     #     centroid_list.append(centroid)
#     #
#     # distance_list = []
#     # for i in range(len(centroid_list)-1):
#     #     dist = ProcessResults().distance(centroid_list[i+1], centroid_list[i])
#     #     distance_list.append(dist)
#     #
#     # print(distance_list)
#     #
#     # ProcessResults().visualize_generational_series(distance_list, title='centroid distance FB', x_label='snapshot', y_label='distance (-)')
#     #
#     #
#     # # ------------------------------------------
#     # print(len(data_H['best_f'][-1]))
#     # print(len(data_FB['Archive_solutions'][-1]))
#     #
#     # non_dominated_FB = []
#     # for sol_FB in data_FB['Archive_solutions'][-1]:
#     #     for sol_H in data_H['best_f'][-1]:
#     #         if ProcessResults().dominates(sol_FB, sol_H):
#     #             non_dominated_FB.append(sol_FB)
#     #
#     # print(len(non_dominated_FB))
#     #
#     # non_dominated_H = []
#     # for sol_FB in data_FB['Archive_solutions'][-1]:
#     #     for sol_H in data_H['best_f'][-1]:
#     #         if ProcessResults().dominates(sol_H, sol_FB):
#     #             non_dominated_H.append(sol_H)
#     #
#     # unique_array = np.unique(non_dominated_H)
#     #
#     # print(len(unique_array))
#
#     # --------------------------------------
#
#     # PROPER generational distance
#     # Calculated by taking the closest solution in the next generation to a point in the pervious generation, for every point in the PF
#
#     # Generational Distance
#
#     # generational_distance = ProcessResults().calculate_generational_distance(data_H['best_f'])
#     # print(generational_distance)
#     # ProcessResults().visualize_generational_series(generational_distance, title='Generational Distance H', x_label='snapshot', y_label='distance (-)')
#     # print(len(generational_distance))
#     #
#     # generational_distance = ProcessResults().calculate_generational_distance(data_FB['Archive_solutions'])
#     # print(generational_distance)
#     # ProcessResults().visualize_generational_series(generational_distance, title='Generational Distance FB', x_label='snapshot', y_label='distance (-)')
#     # print(len(generational_distance))
#     #
#     # # Very spicky, not good
#     #
#     # # Try hypervolume
#     #
#     # reference_point = np.array([10, 500000, 10, -100])
#     #
#     # hypervolume_metric = ProcessResults().calculate_generational_hypervolume(data_H['best_f'], reference_point)
#     # print(hypervolume_metric)
#     # ProcessResults().visualize_generational_series(hypervolume_metric, title='Hypervolume H', x_label='snapshot', y_label='volume (-)')
#     # print(len(hypervolume_metric))
#     #
#     # hypervolume_metric = ProcessResults().calculate_generational_hypervolume(data_FB['Archive_solutions'], reference_point)
#     # print(hypervolume_metric)
#     # ProcessResults().visualize_generational_series(hypervolume_metric, title='Hypervolume FB', x_label='snapshot', y_label='volume (-)')
#     # print(len(hypervolume_metric))
#
#
#     # -- All figures as subplots ----------
#     # file_path = r'/output_data/Folsom_Herman_25000nfe_snapshots.pkl'
#     file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\HermanRICE_100000nfe_random_other_levers_seed_5_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\HermanRICE_100000nfe_random_other_levers_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_100000nfe_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_25000nfe_snapshots.pkl'
#     data_H_5 = ProcessResults().Pickle(file_path)
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\HermanRICE_100000nfe_random_other_levers_seed_26_snapshots.pkl'
#     # data_H_26 = ProcessResults().Pickle(file_path)
#
#     # file_path = r'/output_data/Folsom_ForestBorg_100000nfe_snapshots.pkl'
#     file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\ForestborgRICE_100000nfe_seed_26_snapshots.pkl'
#     data_FB_26 = ProcessResults().Pickle(file_path)
#     file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\ForestborgRICE_100000nfe_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_new_numpy_operators_full_restart_and_restart_escape_latch_fixed_Archive_duplicates_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_100000nfe_new_numpy_operators_full_restart_and_restart_escape_latch_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_50000nfe_new_numpy_operators_no_restart_mech_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\TEST_Folsom_ForestBorg_10000nfe_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_new_numpy_operators_no_gamma_restart_mech_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_100000nfe_new_add_Archive_add_population_no_gamma_restart_more_snapshots_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_new_add_Archive_add_population_no_gamma_restart_more_snapshots_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_new_add_Archive_no_restart_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_new_add_Archive_snapshots.pkl'
#     # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_w_snapshots_snapshots.pkl'
#     data_FB = ProcessResults().Pickle(file_path)
#     file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\ForestborgRICE_100000nfe_seed_5_snapshots.pkl'
#     data_FB_5 = ProcessResults().Pickle(file_path)
#
#     # print('loaded')
#     # print(data_FB.keys())
#     # print(len(data_FB['Archive_solutions']))
#     # print(data_FB['nfe'])
#     #
#     # gen_0 = data_FB['Archive_solutions'][0]
#     # gen_1 = data_FB['Archive_solutions'][1]
#     #
#     # print(gen_0)
#     # print(gen_1)
#     #
#     # print(f'size gen_0: {len(gen_0)}')
#     # print(f'size gen_1: {len(gen_1)}')
#     #
#     def get_unique_array(gen):
#         # Convert the list of arrays to a 2D array
#         arr_2d = np.vstack(gen)
#         # Find the unique rows
#         unique_rows = np.unique(arr_2d, axis=0)
#         return unique_rows
#     #
#     # gen_0 = get_unique_array(gen_0)
#     # print(f'size gen_0: {len(gen_0)}')
#     #
#     # gen_1 = get_unique_array(gen_1)
#     # print(f'size gen_1: {len(gen_1)}')
#     #
#     #
#     # def check_not_present(list_of_arrays, d_array):
#     #     # Iterate through the list and check each array
#     #     for arr in list_of_arrays:
#     #         if np.array_equal(arr, d_array):
#     #             return False
#     #     return True
#     #
#     # new_sol = []#np.array([])
#     # for sol in gen_1:
#     #     if check_not_present(gen_0, sol):
#     #         # new_sol = np.append(new_sol, sol)
#     #         new_sol.append(sol)
#     #
#     # print(new_sol)
#     # print(f'size new_sol: {len(new_sol)}')
#     #
#     # data_FB['unique_Archive_solutions'] = []
#     # for gen in data_FB['Archive_solutions']:
#     #     if not len(gen) == len(get_unique_array(gen)):
#     #         print(len(gen))
#     #         print(len(get_unique_array(gen)))
#     #         print(f'gen {gen} has duplicates')
#
#     # print('before check')
#     # for gen in data_FB['Archive_solutions']:
#     #     gen_duplicates = 0
#     #     for sol in gen:
#     #         if np.any(np.array_equal(sol, x) for x in gen):
#     #             gen_duplicates += 1
#     #     if not len(gen) == gen_duplicates:
#     #         print('duplicate found')
#     #
#     # print('after check')
#     # #
#     # print('before check')
#     # for gen in data_H['best_f']:
#     #     gen_duplicates = 0
#     #     for sol in gen:
#     #         if np.any(np.array_equal(sol, x) for x in gen):
#     #             gen_duplicates += 1
#     #     if not len(gen) == gen_duplicates:
#     #         print('duplicate found')
#     #
#     # print('after check')
#
#     # best_H = data_H['best_f'][-1]
#     # best_FB = data_FB['Archive_solutions'][-1]
#     #
#     # print(len(best_H))
#     # print(len(best_FB))
#     #
#     # H_dominates_FB = []
#     # for sol_H in best_H:
#     #     for sol_FB in best_FB:
#     #         if ProcessResults().dominates(sol_H, sol_FB):
#     #             H_dominates_FB.append(sol_H)
#     # # print(H_dominates_FB)
#     # print(len(H_dominates_FB))
#     #
#     # unique_dom = np.unique(H_dominates_FB, axis=0)
#     # print(len(unique_dom))
#     # print(unique_dom)
#     #
#     # FB_dominates_H = []
#     # for sol_FB in best_FB:
#     #     for sol_H in best_H:
#     #         if ProcessResults().dominates(sol_FB, sol_H):
#     #             FB_dominates_H.append(sol_FB)
#     # # print(H_dominates_FB)
#     # print(len(FB_dominates_H))
#     #
#     # unique_dom = np.unique(FB_dominates_H, axis=0)
#     # print(len(unique_dom))
#     # print(unique_dom)
#
#     #
#     # size_diff = []
#     # nr_new_sol = []
#     # for idx in range(len(data_FB['unique_Archive_solutions'])-1):
#     #     print(f'size gen_{idx}: {len(data_FB["unique_Archive_solutions"][idx])}')
#     #     print(f'size gen_{idx+1}: {len(data_FB["unique_Archive_solutions"][idx+1])}')
#     #     new_sol = []  # np.array([])
#     #     for sol in data_FB["unique_Archive_solutions"][idx+1]:
#     #         if check_not_present(data_FB["unique_Archive_solutions"][idx], sol):
#     #             # new_sol = np.append(new_sol, sol)
#     #             new_sol.append(sol)
#     #     print(f'size new_sol: {len(new_sol)}')
#     #     print('-----------------------------------------------------')
#     #     size_diff.append(len(data_FB["unique_Archive_solutions"][idx+1])-len(data_FB["unique_Archive_solutions"][idx]))
#     #     nr_new_sol.append(len(new_sol))
#     #
#     # ProcessResults().visualize_generational_series(size_diff)
#     # ProcessResults().visualize_generational_series(nr_new_sol)
#
#     # for gen in data_FB['Archive_solutions']:
#     #     print(len(gen))
#     #     unique_gen = get_unique_array(gen)
#     #     print(len(unique_gen))
#     #     print('-----------------')
#
#  # ---------------------
#     # print(len(data_FB['Archive_solutions'][76]))
#     # print(len(np.unique(data_FB['Archive_solutions'][86], axis=0)))
#     #
#     # # peak_dominates_list = []
#     # # for sol_1 in data_FB['Archive_solutions'][76]:
#     # #     for sol_2 in data_FB['Archive_solutions'][86]:
#     # #         if ProcessResults().dominates(sol_1, sol_2):
#     # #             peak_dominates_list.append(sol_1)
#     # # print(f'sol_1 dominates: {len(peak_dominates_list)}')
#     # peak_dominated_list = []
#     # for sol_1 in data_FB['Archive_solutions'][86]:
#     #     for sol_2 in data_FB['Archive_solutions'][76]:
#     #         if ProcessResults().dominates(sol_1, sol_2):
#     #             peak_dominated_list.append(sol_1)
#     # print(f'sol_1 dominates: {len(peak_dominated_list)}')
#     # unique_rows = np.unique(peak_dominated_list, axis=0)
#     # # unique_values = list(set(peak_dominated_list))
#     # print(f'unique: {len(unique_rows)}')
#
#     # for idx, pf in enumerate(data_FB['Archive_solutions']):
#     #     print(idx, len(pf))
#
#     # print(data_H['best_f'][-1])
#     # print(data_FB['Archive_solutions'][-1])
#
#     # data_FB = {k: data_FB[k] for k in list(data_FB)[:1009]}
#     # data_FB_5 = {k: data_FB_5[k] for k in list(data_FB_5)[:1009]}
#     #
#     # print(len(data_FB['Archive_solutions']))
#     # print(len(data_FB_5['Archive_solutions']))
#     # print(len(data_FB_26['Archive_solutions']))
#
#     # Create a 2x2 subplot grid
#     # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
#     fig, axs = plt.subplots(1, 2, figsize=(10, 8))
#
#     # # Generational Distance
#     # generational_distance_H = ProcessResults().calculate_generational_distance(data_H['best_f'])
#     # ProcessResults().visualize_generational_metrics(generational_distance_H, ax=axs[0, 0], title='Generational Distance Herman POT',
#     #                                                x_label='snapshot', y_label='distance (-)')
#     #
#     # generational_distance_FB = ProcessResults().calculate_generational_distance(data_FB['Archive_solutions'])
#     # ProcessResults().visualize_generational_metrics(generational_distance_FB, ax=axs[0, 1], title='Generational Distance ForestBORG',
#     #                                                x_label='snapshot', y_label='distance (-)')
#     # Hypervolume
#     # reference_point = np.array([10, 500000, 10, -100])
#     # reference_point = np.array([0, ])
#     reference_point = np.array([10, 100, 100])
#
#     hypervolume_metric_H_5 = ProcessResults().calculate_generational_hypervolume(data_H_5['best_f'], reference_point)
#     ProcessResults().visualize_generational_metrics(hypervolume_metric_H_5, x_data=data_H_5['nfe'], ax=axs[0],
#                                                     title='Hypervolume Herman POT', x_label='nfe',
#                                                     y_label='volume (-)')
#     hypervolume_metric_FB_5 = ProcessResults().calculate_generational_hypervolume(data_FB_5['Archive_solutions'][:1009],
#                                                                                   reference_point)
#     ProcessResults().visualize_generational_metrics(hypervolume_metric_FB_5, x_data=data_FB['nfe'][:1009], ax=axs[1], color='blue',
#                                                     title='Hypervolume ForestBorg', x_label='nfe',
#                                                     y_label='volume (-)')
#     hypervolume_metric_FB_26 = ProcessResults().calculate_generational_hypervolume(data_FB_26['Archive_solutions'][:1009],
#                                                                                   reference_point)
#     ProcessResults().visualize_generational_metrics(hypervolume_metric_FB_26, x_data=data_FB_26['nfe'][:1009], ax=axs[1], color='blue',
#                                                     title='Hypervolume ForestBorg', x_label='nfe',
#                                                     y_label='volume (-)')
#     hypervolume_metric_FB = ProcessResults().calculate_generational_hypervolume(data_FB['Archive_solutions'][:1009],
#                                                                                   reference_point)
#     ProcessResults().visualize_generational_metrics(hypervolume_metric_FB, x_data=data_FB['nfe'][:1009], ax=axs[1], color='blue',
#                                                     title='Hypervolume ForestBorg', x_label='nfe',
#                                                     y_label='volume (-)')
#
#
#     # hypervolume_metric_H_5 = ProcessResults().calculate_generational_hypervolume(data_H_5['best_f'], reference_point)
#     # ProcessResults().visualize_generational_metrics(hypervolume_metric_H_5, x_data=data_H_5['nfe'], ax=axs[0, 0],
#     #                                                 title='Hypervolume Herman POT', x_label='nfe',
#     #                                                 y_label='volume (-)')
#     # hypervolume_metric_FB_5 = ProcessResults().calculate_generational_hypervolume(data_FB_5['Archive_solutions'],
#     #                                                                             reference_point)
#     # ProcessResults().visualize_generational_metrics(hypervolume_metric_FB_5, x_data=data_FB_5['nfe'], ax=axs[1, 1],
#     #                                                 title='Hypervolume ForestBorg', x_label='nfe',
#     #                                                 y_label='volume (-)')
#     #
#     # hypervolume_metric_H_26 = ProcessResults().calculate_generational_hypervolume(data_H_26['best_f'], reference_point)
#     # ProcessResults().visualize_generational_metrics(hypervolume_metric_H_26, x_data=data_H_26['nfe'], ax=axs[1, 0], title='Hypervolume Herman POT', x_label='nfe',
#     #                                                y_label='volume (-)')
#     # hypervolume_metric_FB_26 = ProcessResults().calculate_generational_hypervolume(data_FB_26['Archive_solutions'],
#     #                                                                          reference_point)
#     # ProcessResults().visualize_generational_metrics(hypervolume_metric_FB_26, x_data=data_FB_26['nfe'], ax=axs[1, 1], title='Hypervolume ForestBorg', x_label='nfe',
#     #                                                y_label='volume (-)')
#     # Add a title for the entire plot
#     plt.suptitle('Convergence metrics Herman POT and ForestBORG on RICE model')
#
#     # Adjust spacing between subplots
#     plt.tight_layout()
#
#     # Display the plot
#     plt.show()
#
#     # for idx, value in enumerate(hypervolume_metric_FB):
#     #     print(idx, value)






