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


class ProcessResults:
    def __init__(self):
        return

    def Pickle(self, file_path):
        # file_path = fr'{filepath}\{file_name}'
        with open(file_path, "rb") as file:
            snapshots = pickle.load(file)
        return snapshots

    def distance(self, P1, P2):
        # Input is list
        num_dimensions = len(P1)
        dist = []
        for dimension in range(num_dimensions):
            dist_ = (P2[dimension] - P1[dimension]) ** 2
            dist.append(dist_)
        distance = math.sqrt(sum(dist))
        return distance

    def calculate_centroid(self, points):
        # if not points:
        #     raise ValueError("Points list is empty")

        # Determine the dimensionality by checking the length of the first point
        dimension = len(points[0])

        # Initialize sums for each dimension
        sums = [0.0] * dimension

        # Calculate the sum of each dimension for all points
        for point in points:
            if len(point) != dimension:
                raise ValueError("All points must have the same dimension")
            sums = [sums[i] + point[i] for i in range(dimension)]

        # Calculate the average for each dimension
        centroid = [sums[i] / len(points) for i in range(dimension)]

        return centroid

    def visualize_generational_series(self, series, title='centroid distance', x_label='snapshot', y_label='distance (-)', save=False):
        x = [x for x in range(len(series))]
        y = series
        plt.plot(x, y)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if save:
            plt.savefig(f'{self.save_location}/{title}.png', bbox_inches='tight')
        else:
            plt.show()
        # Make sure to close the plt object once done
        plt.close()

    def view_sqlite_database(self, database, table_name):
        # df = pd.DataFrame()
        conn = sqlite3.connect(database)
        # c = conn.cursor()

        # c.execute(f"""SELECT count(*) FROM sqlite_master WHERE type='table' AND name={table_name}""")
        df = pd.read_sql_query(f'''SELECT * FROM {table_name}''', conn)

        conn.commit()
        conn.close()

        return df

    def list_table_names_in_db(self, database):
        try:

            # Making a connection between sqlite3
            # database and Python Program
            sqliteConnection = sqlite3.connect(database)

            # If sqlite3 makes a connection with python
            # program then it will print "Connected to SQLite"
            # Otherwise it will show errors
            print("Connected to SQLite")

            # Getting all tables from sqlite_master
            sql_query = """SELECT name FROM sqlite_master
            WHERE type='table';"""

            # Creating cursor object using connection object
            cursor = sqliteConnection.cursor()

            # executing our sql query
            cursor.execute(sql_query)
            print("List of tables\n")

            # printing all tables list
            print(cursor.fetchall())

        except sqlite3.Error as error:
            print("Failed to execute the above query", error)

        finally:

            # Inside Finally Block, If connection is
            # open, we need to close it
            if sqliteConnection:
                # using close() method, we will close
                # the connection
                sqliteConnection.close()

                # After closing connection object, we
                # will print "the sqlite connection is
                # closed"
                print("the sqlite connection is closed")

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

    def calculate_generational_distance(self, pareto_front_generations):
        # Input: array/ array of arrays
        def closest_distance_to_points(target_point, points_collection):
            """
            Calculate the closest distance from a target point to a collection of other points in multi-dimensional space.

            Parameters:
            target_point (numpy.ndarray): The target point for which you want to find the closest distance.
            points_collection (numpy.ndarray): A 2D numpy array where each row represents a point in multi-dimensional space.

            Returns:
            float: The closest distance from the target_point to any point in the collection.
            """
            # Calculate the Euclidean distances between the target_point and all points in the collection
            distances = np.linalg.norm(points_collection - target_point, axis=1)

            # Find the index of the point with the minimum distance
            closest_point_index = np.argmin(distances)

            # Get the closest distance
            closest_distance = distances[closest_point_index]

            return closest_distance

        generational_distance = np.array([])
        for idx, generation in enumerate(pareto_front_generations[:-1]):
            gen_distance = 0
            gen_0 = generation
            gen_1 = pareto_front_generations[idx + 1]
            for sol in gen_0:
                gen_distance += closest_distance_to_points(sol, gen_1)
            generational_distance = np.append(generational_distance, gen_distance)
        return generational_distance

    def calculate_generational_hypervolume(self, pareto_front_generations, reference_point):
        def hypervolume(front, reference_point):
            """
            Calculate the hypervolume metric for a set of solutions in multi-objective optimization.

            Parameters:
            front (list of lists): A list of objective vectors for each solution.
            reference_point (list): The reference point for the hypervolume calculation.

            Returns:
            hypervolume_value (float): The hypervolume metric value.
            """

            # Convert the input to NumPy arrays for efficient calculations
            front = np.array(front)
            reference_point = np.array(reference_point)

            # Initialize the hypervolume value
            hypervolume_value = 0.0

            # Iterate through each solution in the front
            for solution in front:
                # Calculate the hypervolume contribution of each solution
                contribution = np.prod(np.maximum(reference_point - solution, 0))

                # Update the total hypervolume value
                hypervolume_value += contribution

            return hypervolume_value

        hypervolume_metric = np.array([])
        for generation in pareto_front_generations:
            hypervolume_metric = np.append(hypervolume_metric, hypervolume(generation, reference_point))
        return hypervolume_metric

    def visualize_generational_metrics(self, series, ax, title='centroid distance', x_label='snapshot', y_label='distance (-)', save=False):
        x = [x for x in range(len(series))]
        y = series
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)


if __name__ == '__main__':
    # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_25000nfe_snapshots.pkl'
    # data_H = ProcessResults().Pickle(file_path)
    #
    # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_w_snapshots_snapshots.pkl'
    # data_FB = ProcessResults().Pickle(file_path)
    # centroid_list = []
    # for generation_f in data_H['best_f']:
    #     centroid = ProcessResults().calculate_centroid(generation_f)
    #     centroid_list.append(centroid)
    #
    # distance_list = []
    # for i in range(len(centroid_list)-1):
    #     dist = ProcessResults().distance(centroid_list[i+1], centroid_list[i])
    #     distance_list.append(dist)
    #
    # print(distance_list)
    #
    # ProcessResults().visualize_generational_series(distance_list, title='centroid distance H', x_label='snapshot', y_label='distance (-)')
    # ----------------------------
    #
    # centroid_list = []
    # for generation_f in data_FB['Archive_solutions']:
    #     centroid = ProcessResults().calculate_centroid(generation_f)
    #     centroid_list.append(centroid)
    #
    # distance_list = []
    # for i in range(len(centroid_list)-1):
    #     dist = ProcessResults().distance(centroid_list[i+1], centroid_list[i])
    #     distance_list.append(dist)
    #
    # print(distance_list)
    #
    # ProcessResults().visualize_generational_series(distance_list, title='centroid distance FB', x_label='snapshot', y_label='distance (-)')
    #
    #
    # # ------------------------------------------
    # print(len(data_H['best_f'][-1]))
    # print(len(data_FB['Archive_solutions'][-1]))
    #
    # non_dominated_FB = []
    # for sol_FB in data_FB['Archive_solutions'][-1]:
    #     for sol_H in data_H['best_f'][-1]:
    #         if ProcessResults().dominates(sol_FB, sol_H):
    #             non_dominated_FB.append(sol_FB)
    #
    # print(len(non_dominated_FB))
    #
    # non_dominated_H = []
    # for sol_FB in data_FB['Archive_solutions'][-1]:
    #     for sol_H in data_H['best_f'][-1]:
    #         if ProcessResults().dominates(sol_H, sol_FB):
    #             non_dominated_H.append(sol_H)
    #
    # unique_array = np.unique(non_dominated_H)
    #
    # print(len(unique_array))

    # --------------------------------------

    # PROPER generational distance
    # Calculated by taking the closest solution in the next generation to a point in the pervious generation, for every point in the PF

    # Generational Distance

    # generational_distance = ProcessResults().calculate_generational_distance(data_H['best_f'])
    # print(generational_distance)
    # ProcessResults().visualize_generational_series(generational_distance, title='Generational Distance H', x_label='snapshot', y_label='distance (-)')
    # print(len(generational_distance))
    #
    # generational_distance = ProcessResults().calculate_generational_distance(data_FB['Archive_solutions'])
    # print(generational_distance)
    # ProcessResults().visualize_generational_series(generational_distance, title='Generational Distance FB', x_label='snapshot', y_label='distance (-)')
    # print(len(generational_distance))
    #
    # # Very spicky, not good
    #
    # # Try hypervolume
    #
    # reference_point = np.array([10, 500000, 10, -100])
    #
    # hypervolume_metric = ProcessResults().calculate_generational_hypervolume(data_H['best_f'], reference_point)
    # print(hypervolume_metric)
    # ProcessResults().visualize_generational_series(hypervolume_metric, title='Hypervolume H', x_label='snapshot', y_label='volume (-)')
    # print(len(hypervolume_metric))
    #
    # hypervolume_metric = ProcessResults().calculate_generational_hypervolume(data_FB['Archive_solutions'], reference_point)
    # print(hypervolume_metric)
    # ProcessResults().visualize_generational_series(hypervolume_metric, title='Hypervolume FB', x_label='snapshot', y_label='volume (-)')
    # print(len(hypervolume_metric))


    # -- All figures as subplots ----------
    # file_path = r'/output_data/Folsom_Herman_25000nfe_snapshots.pkl'
    file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_25000nfe_snapshots.pkl'
    data_H = ProcessResults().Pickle(file_path)

    # file_path = r'/output_data/Folsom_ForestBorg_100000nfe_snapshots.pkl'
    file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\TEST_Folsom_ForestBorg_25000nfe_new_add_Archive_snapshots.pkl'
    # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_w_snapshots_snapshots.pkl'
    data_FB = ProcessResults().Pickle(file_path)

    # Create a 2x2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Generational Distance
    generational_distance_H = ProcessResults().calculate_generational_distance(data_H['best_f'])
    ProcessResults().visualize_generational_metrics(generational_distance_H, ax=axs[0, 0], title='Generational Distance Herman POT',
                                                   x_label='snapshot', y_label='distance (-)')

    generational_distance_FB = ProcessResults().calculate_generational_distance(data_FB['Archive_solutions'])
    ProcessResults().visualize_generational_metrics(generational_distance_FB, ax=axs[0, 1], title='Generational Distance ForestBORG',
                                                   x_label='snapshot', y_label='distance (-)')
    # Hypervolume
    reference_point = np.array([10, 500000, 10, -100])

    hypervolume_metric_H = ProcessResults().calculate_generational_hypervolume(data_H['best_f'], reference_point)
    ProcessResults().visualize_generational_metrics(hypervolume_metric_H, ax=axs[1, 0], title='Hypervolume Herman POT', x_label='snapshot',
                                                   y_label='volume (-)')
    hypervolume_metric_FB = ProcessResults().calculate_generational_hypervolume(data_FB['Archive_solutions'],
                                                                             reference_point)
    ProcessResults().visualize_generational_metrics(hypervolume_metric_FB, ax=axs[1, 1], title='Hypervolume ForestBorg', x_label='snapshot',
                                                   y_label='volume (-)')
    # Add a title for the entire plot
    plt.suptitle('Convergence metrics Herman POT and ForestBORG on FOLSOM lake model')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()






