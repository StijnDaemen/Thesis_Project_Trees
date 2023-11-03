import math
import matplotlib.pyplot as plt

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

        def calculate_hypervolume(solutions, reference_point):
            """
            Calculate the hypervolume of a non-dominated set of solutions.

            :param solutions: A list of solutions, where each solution is itself a list of objectives.
            :param reference_point: A worst-case point that is dominated by all the solutions.
            :return: The hypervolume enclosed by the solution set.
            """
            # Sort the solution set by the first objective in descending order
            sorted_solutions = sorted(solutions, key=lambda x: x[0], reverse=True)

            # Initialize hypervolume
            hypervolume = 0.0

            # Process each solution
            for i, solution in enumerate(sorted_solutions):
                # Distance to the next solution in the first objective or to the reference point if it's the last solution
                if i < len(sorted_solutions) - 1:
                    width = sorted_solutions[i + 1][0] - solution[0]
                else:
                    width = reference_point[0] - solution[0]

                # Calculate the contribution of the current solution
                contribution = width
                for obj_index in range(1, len(solution)):
                    contribution *= solution[obj_index] - reference_point[obj_index]

                hypervolume += contribution

            return hypervolume

        hypervolume_metric = np.array([])
        for generation in pareto_front_generations:
            hypervolume_metric = np.append(hypervolume_metric, calculate_hypervolume(generation, reference_point))

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


if __name__ == '__main__':
    # file_name = 'ForestborgRICE_100000nfe_seed_42_snapshots'
    # print(path_to_dir+f'\output_data\{file_name}.pkl')
    # snapshots = Pickle(path_to_dir+f'\output_data\{file_name}.pkl')
    # print(snapshots.keys())
    # print(snapshots['Archive_solutions'][-1])
    # print(len(snapshots['Archive_solutions'][-1]))

    file_name = 'TESTFB_Archive_platypus_snapshots'
    snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')

    # PF_size = []
    # for gen in snapshots['Archive_solutions']:
    #     PF_size.append(len(gen))
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(PF_size)
    # plt.show()

    def dominates(a, b):
        # assumes minimization
        # a dominates b if it is <= in all objectives and < in at least one
        # Note SD: somehow the logic with np.all() breaks down if there are positive and negative numbers in the array
        # So to circumvent this but still allow multiobjective optimisation in different directions under the
        # constraint that every number is positive, just add a large number to every index.

        large_number = 1000000000
        a = a + large_number
        b = b + large_number

        return np.all(a <= b) and np.any(a < b)

    for idx in range(len(snapshots['Archive_solutions'])-1):
        for sol0 in snapshots['Archive_solutions'][idx]:
            for sol1 in snapshots['Archive_solutions'][idx+1]:
                if dominates(sol0, sol1):
                    print(sol0, sol1)
                else:
                    print('yep!')

    # for sol0 in snapshots['Archive_solutions'][0]:
    #     print(sol0)



    #
    # start_last = find_last_occurrence(snapshots['epsilon_progress'])
    # print(len(snapshots['epsilon_progress'][start_last:]))
    # snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
    #
    # # plt.plot(snapshots['epsilon_progress_metric'], color='blue')
    # # plt.title('Epsilon Progress Metric')
    # # plt.xlabel('Index')
    # # plt.ylabel('Value')
    # # plt.grid(True)
    # # plt.show()
    #
    # reference_point = np.array([10, 100, 100])
    #
    # hypervolume_metric = calculate_generational_hypervolume(snapshots['Archive_solutions'], reference_point)
    # snapshots['hypervolume_metric'] = hypervolume_metric
    #
    # print(len(hypervolume_metric))
    # print(len(snapshots['nfe']))
    # print(hypervolume_metric)
    #
    # # Create a 2x1 grid of subplots (2 rows, 1 column)
    # fig, axs = plt.subplots(1, 2, figsize=(8, 10))
    #
    # # Plotting the 'epsilon' series in the first subplot
    # axs[0].plot(snapshots['epsilon_progress_metric'], color='blue')
    # axs[0].set_title('Epsilon Series')
    # axs[0].set_xlabel('Index')
    # axs[0].set_ylabel('Value')
    # axs[0].grid(True)
    #
    # # Plotting the 'hyper' series in the second subplot
    # axs[1].plot(snapshots['hypervolume_metric'], color='red')
    # axs[1].set_title('Hyper Series')
    # axs[1].set_xlabel('Index')
    # axs[1].set_ylabel('Value')
    # axs[1].grid(True)
    #
    # # Adjust spacing between plots
    # plt.tight_layout()
    # plt.show()

    # df = pd.DataFrame(data=snapshots['Archive_trees'][-1], columns=['policy'])
    # # print(df['policy'])
    # # print(df.info())
    # # print(df.head())
    #
    # feature_names = ['mat', 'net_output', 'year']
    # action_names = ['miu', 'sr', 'irstp']
    # indicators_actions_analysis(df, feature_names, action_names)




    # -- Metrics figure -----------

    # file_names = ['ForestborgRICE_100000nfe_seed_42_snapshots',
    #               'ForestborgRICE_100000nfe_seed_5_snapshots',
    #               'ForestborgRICE_100000nfe_seed_17_snapshots',
    #               'ForestborgRICE_100000nfe_seed_26_snapshots',
    #               'ForestborgRICE_100000nfe_seed_55_snapshots',
    #               'ForestborgRICE_100000nfe_seed_104_snapshots',
    #               'ForestborgRICE_100000nfe_seed_303_snapshots',
    #               'ForestborgRICE_100000nfe_seed_506_snapshots',
    #               'ForestborgRICE_100000nfe_seed_902_snapshots']
    # dicts = []
    # for file_name in file_names:
    #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
    #     # Metrics analysis
    #     reference_point = np.array([10, 100, 100])
    #     snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['Archive_solutions'], reference_point)
    #     start_last = find_last_occurrence(snapshots['epsilon_progress'])
    #     snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress'][start_last:]
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
    # plt.savefig('Hypervolume ForestBORG applied to the RICE model, seed analysis.png', dpi=300)
    #
    # plt.tight_layout()
    # plt.show()

    # -------------



# # Plotting
# plt.figure(figsize=(10, 6))
# for idx, d in enumerate(dicts):
#     plt.plot(d['nfe'], d['hypervolume_metric'], color='blue')#, label=f'Dict {idx + 1}', color='blue')#, cmap=plt.get_cmap(colormap))#, marker='o')
#
# plt.xlabel('nfe', fontsize=14)
# plt.ylabel('hypervolume (-)', fontsize=14)
# plt.title('Hypervolume ForestBORG applied to the RICE model, seed analysis', fontsize=16)
# plt.legend(loc='upper left', fontsize=12)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # plt.xlim([0, 50])
# # plt.ylim([0, 40])
# plt.show()


# ----------------------------------------------------------------------


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






