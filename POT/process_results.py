import math
import matplotlib.pyplot as plt

# Import the generative model by SD
from RICE_model.IAM_RICE import RICE
# Import the policy tree optimizer by Herman
from POT.ptreeopt import PTreeOpt
import logging
# Import the policy tree optimizer with borg
from POT.borg_optimization import PolicyTreeOptimizer
# Import the control - random search - 'optimizer'
from POT.control_optimization import PolicyTreeOptimizerControl
# Import the ema workbench by professor Kwakkel
from ema_workbench import RealParameter, ScalarOutcome, Constant, Model, IntegerParameter
from ema_workbench import SequentialEvaluator, ema_logging
from ema_workbench import save_results
# Import the homemade POT optimizer
from POT.homemade_optimization import Cluster
from POT.optimization_tryout import Cluster_
from POT.forest_borg import ForestBorg
from POT.forest_borg_Folsom import ForestBorgFolsom

from folsom import Folsom
from ptreeopt import PTreeOpt
import logging
import pickle

import pandas as pd
import numpy as np
import sqlite3
import time
import os
from ema_workbench import load_results, ema_logging
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

    def visualize_generational_series(self, series, title='centroid distance', x_label='snapshot(x100)', y_label='distance (-)', save=False):
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


if __name__ == '__main__':
    # # save_location = path_to_dir + '\\output_data'
    # # file_name = 'TEST_Folsom_Herman_5000nfe_snapshots.pkl'
    # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_Herman_25000nfe_snapshots.pkl'
    # data = ProcessResults().Pickle(file_path)
    # print(type(data))
    # print(data.keys())
    # print(data['best_f'][-1])
    # centroid1 = ProcessResults().calculate_centroid(data['best_f'][-1])
    # print(centroid1)
    #
    # centroid2 = ProcessResults().calculate_centroid(data['best_f'][-2])
    # print(centroid2)
    #
    # distance = ProcessResults().distance(centroid2, centroid1)
    # print(distance)
    #
    # centroid_list = []
    # for generation_f in data['best_f']:
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
    # ProcessResults().visualize_generational_series(distance_list)
    #

    # ----------------------------

    file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\TEST_Folsom_ForestBorg_1000nfe_snapshots.pkl'
    data = ProcessResults().Pickle(file_path)
    print(data)

    centroid_list = []
    for generation_f in data['Archive_solutions']:
        centroid = ProcessResults().calculate_centroid(generation_f)
        centroid_list.append(centroid)

    distance_list = []
    for i in range(len(centroid_list)-1):
        dist = ProcessResults().distance(centroid_list[i+1], centroid_list[i])
        distance_list.append(dist)

    print(distance_list)

    ProcessResults().visualize_generational_series(distance_list)





    # df = ProcessResults().view_sqlite_database(r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Experiments.db', 'archive_snapshots_Folsom_ForestBorg_25000nfe')
    # print(df.info())
    # print(df.head())
    # # df.to_excel('fb_folsom_.xlsx')
    #
    # # Split the 'string_column' into two parts at the '_' character
    # df[['number_part', 'text_part']] = df['index'].str.split('_', 1)
    #
    # # Convert the 'number_part' column to integers (if needed)
    # df['number_part'] = df['number_part'].astype(int)
    #
    # # Calculate the differences between consecutive values in 'existing_column'
    # diff_values = df['number_part'].diff()
    #
    # # Initialize the 'incremental_column' with zeros
    # df['incremental_column'] = 0
    #
    # # Use cumsum() to accumulate the differences
    # df['incremental_column'] = diff_values.cumsum().fillna(0).astype(int)
    #
    # print(df.head())
    #
    # # file_path = r'C:\\Users\\Stijn Daemen\\Documents\\master thesis TU Delft\\code\\a_git folder_ do not keep large files here\\IAM_RICE2\\output_data\\Folsom_ForestBorg_25000nfe_Archive_snapshots.pkl'
    # # data = ProcessResults().Pickle(file_path)
    # # print(len(data))
    # # # print(type(data))

