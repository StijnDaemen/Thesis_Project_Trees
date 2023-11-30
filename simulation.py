from RICE_model.IAM_RICE import RICE
from POT.forest_borg import ForestBorg
from folsom import Folsom
from ptreeopt.opt import PTreeOpt
import logging
import pickle
import numpy as np
import time
import os

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons):
    title_of_run = f'Folsom_Herman_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_v1'
    np.random.seed(seed)

    start = time.time()

    model = Folsom('folsom/data/folsom-daily-w2016.csv',
                   sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)

    algorithm = PTreeOpt(model.f,
                         feature_bounds=[[0, 1000], [1, 365], [0, 300]],
                         feature_names=['Storage', 'Day', 'Inflow'],
                         discrete_actions=True,
                         action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
                                       'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                         mu=20,  # number of parents per generation
                         cx_prob=0.70,  # crossover probability
                         population_size=100,
                         max_depth=depth,
                         multiobj=True,
                         epsilons=epsilons  # [0.01, 1000, 0.01, 10]
                         )

    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    # With sonly 1000 function evaluations this will not be very good
    best_solution, best_score, snapshots = algorithm.run(max_nfe=max_nfe,  # 20000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


def Folsom_ForestBorg_discrete(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval):
    title_of_run = f'Folsom_ForestBORG_discrete_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_gamma{gamma}_tau{tau}_restart{restart_interval}_v1'
    start = time.time()

    model = Folsom('folsom/data/folsom-daily-w2016.csv',
                   sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)
    master_rng = np.random.default_rng(seed)  # Master RNG
    snapshots = ForestBorg(pop_size=100,
                           model=model.f,
                           master_rng=master_rng,
                           metrics=['period_utility', 'damages', 'temp_overshoots'],
                           # Tree variables
                           action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
                                         'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                           action_bounds=None,
                           feature_bounds=[[0, 1000], [1, 365], [0, 300]],
                           feature_names=['Storage', 'Day', 'Inflow'],
                           max_depth=depth,
                           discrete_actions=True,
                           discrete_features=None,
                           # Optimization variables
                           mutation_prob=0.5,
                           max_nfe=max_nfe,
                           epsilons=epsilons,  # [0.01, 1000, 0.01, 10],
                           gamma=gamma,  # 4,
                           tau=tau,  # 0.02,
                           restart_interval=restart_interval,
                           save_location=save_location,
                           title_of_run=title_of_run,
                           ).run()
    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


def RICE_Herman_discrete(save_location, seed, max_nfe, depth, epsilons):
    title_of_run = f'RICE_Herman_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_v1'
    start = time.time()

    # model = RICE(years_10, regions, database_POT=input_path+'/ptreeopt/output_data/POT_Experiments.db', table_name_POT='indicator_groupsize_3_bin_tournament_1')
    # model = RICE(years_10, regions, save_location=save_location, file_name=title_of_run)
    np.random.seed(seed)
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
    model = RICE(years_10, regions)
    algorithm = PTreeOpt(model.POT_control_discrete,
                         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                         feature_names=['mat', 'net_output', 'year'],
                         discrete_features=None,
                         discrete_actions=True,
                         action_names=['miu_2065',
                                       'miu_2100',
                                       'miu_2180',
                                       'miu_2250',
                                       'miu_2305',
                                       'sr_01',
                                       'sr_02',
                                       'sr_03',
                                       'sr_04',
                                       'sr_05',
                                       'irstp_0005',
                                       'irstp_0015',
                                       'irstp_0025'],
                         # action_bounds=[[2065, 2305], [0.1, 0.5], [0.01, 0.1]],
                         mu=20,  # number of parents per generation, 20
                         cx_prob=0.70,  # crossover probability
                         population_size=100,  # 100
                         max_depth=depth,  # 4,
                         epsilons=epsilons,  # [0.05, 0.05, 0.05],
                         multiobj=True
                         )

    logging.basicConfig(level=logging.INFO,
                        format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')

    # With only 1000 function evaluations this will not be very good
    best_solution, best_score, snapshots = algorithm.run(max_nfe=max_nfe,  # 20000,
                                                         log_frequency=100,
                                                         snapshot_frequency=100)

    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


def RICE_ForestBorg_discrete(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval, scenario=None,
                             scenario_name=None):
    title_of_run = f'RICE_ForestBORG_discrete_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_gamma{gamma}_tau{tau}_restart{restart_interval}_v1'  # f'ForestborgRICE_100000nfe_seed_{seed}'
    start = time.time()

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
    model = RICE(years_10, regions)
    # model = RICE(years_10, regions, scenario=scenario)
    master_rng = np.random.default_rng(seed)  # Master RNG
    snapshots = ForestBorg(pop_size=100, master_rng=master_rng,
                           model=model.POT_control_discrete,
                           metrics=['period_utility', 'damages', 'temp_overshoots'],
                           # Tree variables
                           action_names=['miu_2065',
                                         'miu_2100',
                                         'miu_2180',
                                         'miu_2250',
                                         'miu_2305',
                                         'sr_01',
                                         'sr_02',
                                         'sr_03',
                                         'sr_04',
                                         'sr_05',
                                         'irstp_0005',
                                         'irstp_0015',
                                         'irstp_0025'],
                           action_bounds=None,
                           feature_names=['mat', 'net_output', 'year'],
                           feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                           max_depth=depth,
                           discrete_actions=True,
                           discrete_features=None,
                           # Optimization variables
                           mutation_prob=0.5,
                           max_nfe=max_nfe,  # 20000,
                           epsilons=epsilons,  # np.array([0.05, 0.05, 0.05]),
                           gamma=gamma,  # 4,
                           tau=tau,  # 0.02,
                           save_location=save_location,
                           title_of_run=title_of_run,
                           ).run()
    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


def RICE_ForestBorg_continuous(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval,
                               scenario=None, scenario_name=None):
    title_of_run = f'RICE_ForestBORG_continuous_seed{seed}_nfe{max_nfe}_depth{depth}_epsilons{epsilons}_gamma{gamma}_tau{tau}_restart{restart_interval}_v1'
    start = time.time()

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
    model = RICE(years_10, regions)
    # model = RICE(years_10, regions, scenario=scenario)
    master_rng = np.random.default_rng(seed)  # Master RNG
    snapshots = ForestBorg(pop_size=100, master_rng=master_rng,
                           model=model.POT_control_continuous,
                           metrics=['period_utility', 'damages', 'temp_overshoots'],
                           # Tree variables
                           action_names=['miu', 'sr', 'irstp'],
                           action_bounds=[[2065, 2305], [0.1, 0.5], [0.01, 0.1]],
                           feature_names=['mat', 'net_output', 'year'],
                           feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
                           max_depth=depth,
                           discrete_actions=False,
                           discrete_features=False,
                           # Optimization variables
                           mutation_prob=0.5,
                           max_nfe=max_nfe,  # 20000,
                           epsilons=epsilons,  # np.array([0.05, 0.05, 0.05]),
                           gamma=gamma,  # 4,
                           tau=tau,  # 0.02,
                           # restart_interval=restart_interval,
                           save_location=save_location,
                           title_of_run=title_of_run,
                           ).run()
    pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
    end = time.time()
    return print(f'Total elapsed time: {(end - start) / 60} minutes.')


if __name__ == '__main__':
    save_location = path_to_dir + '/output_data'

    # The following experiments are possible
    # - - - - - - - - - - - - - - - - - - - -
    # Folsom
    #   - Herman - discrete
    #   - FB - discrete

    # RICE
    #   - Herman - discrete
    #   - FB - discrete

    # RICE
    #   - FB - continuous
    # - - - - - - - - - - - - - - - - - - - -
    # RICE on the discrete systems is not used for any of the experiments at the moment, can do later
    # No scenarios are passed to RICE eventhough they could be. Only the standard RICE scenario is used for these experiments

    max_nfe = 4999  # 30000
    depth = 3
    epsilons = [0.01, 1000, 0.01, 10]
    gamma = 4
    tau = 0.02
    restart_interval = 5000
    seeds = [17, 42]#, 104, 303, 902]
    for seed in seeds:
        Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons)

    # # -- Folsom ------------------
    # # -- I & II -- Run time and operator dynamics - 5 seeds, other controls constant
    # max_nfe = 200 # 30000
    # depth = 3
    # epsilons = np.array([0.01, 1000, 0.01, 10])
    # gamma = 4
    # tau = 0.02
    # restart_interval = 5000
    # seeds = [17, 42, 104, 303, 902]
    # # for seed in seeds:
    # #     Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons)
    # for seed in seeds:
    #     Folsom_ForestBorg_discrete(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval)
    #
    # # -- III -- Controllability map -  control for depth, gamma and restart_interval, for now on one seed
    # max_nfe = 200 #20000
    # depths = [2, 3, 4, 5, 6]
    # epsilons = np.array([0.01, 1000, 0.01, 10])
    # gammas = [3, 4, 5]
    # tau = 0.02
    # restart_intervals = [500, 2000, 5000]
    # seed = 42
    # for depth in depths:
    #     for gamma in gammas:
    #         for restart_interval in restart_intervals:
    #             Folsom_ForestBorg_discrete(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval)
    #
    # # -- IV & V -- Use seed 42 from experiment I & II
    #
    # # -- RICE ------------------
    # # -- I & II -- Run time and operator dynamics - 5 seeds, other controls constant
    # max_nfe = 200 #30000
    # depth = 3
    # epsilons = np.array([0.05, 0.05, 0.05])
    # gamma = 4
    # tau = 0.02
    # restart_interval = 5000
    # seeds = [17, 42, 104, 303, 902]
    # # for seed in seeds:
    # #     Folsom_Herman_discrete(save_location, seed, max_nfe, depth, epsilons)
    # for seed in seeds:
    #     RICE_ForestBorg_continuous(save_location, seed, max_nfe, depth, epsilons, gamma, tau, restart_interval)
    #
    # # -- III -- Controllability map -  control for depth, gamma and restart_interval, for now on one seed
    # max_nfe = 200 #20000
    # depths = [2, 3, 4, 5, 6]
    # epsilons = np.array([0.05, 0.05, 0.05])
    # gammas = [3, 4, 5]
    # tau = 0.02
    # restart_intervals = [500, 2000, 5000]
    # seed = 42
    # for depth in depths:
    #     for gamma in gammas:
    #         for restart_interval in restart_intervals:
    #             RICE_ForestBorg_continuous(save_location, seed, max_nfe, depth, epsilons, gamma, tau,
    #                                        restart_interval)
    #
    # # -- IV & V -- Use a seed from experiment I & II




# -- BEFORE GREEN LIGHT --------------------------------------------------
#
# # Import the generative model by SD
# from RICE_model.IAM_RICE import RICE
# # Import the control - random search - 'optimizer'
# from POT.control_optimization import PolicyTreeOptimizerControl
# # Import the ema workbench by professor Kwakkel
# from ema_workbench import RealParameter, ScalarOutcome, Constant, Model, IntegerParameter
# from ema_workbench import SequentialEvaluator
# # Import the homemade POT optimizer
# from POT.homemade_optimization import Cluster
# from POT.optimization_tryout import Cluster_
# from POT.forest_borg import ForestBorg
# from POT.forest_borg_Folsom import ForestBorgFolsom
# from POT.shotgun_optimization import Shotgun
#
# from folsom import Folsom
# from ptreeopt.opt import PTreeOpt
# import logging
# import pickle
#
# import pandas as pd
# import numpy as np
# import sqlite3
# import time
# import os
# from ema_workbench import ema_logging
# package_directory = os.path.dirname(os.path.abspath(__file__))
# path_to_dir = os.path.join(package_directory)
#
#
# def view_sqlite_database(database, table_name):
#     # df = pd.DataFrame()
#     conn = sqlite3.connect(database)
#     # c = conn.cursor()
#
#     # c.execute(f"""SELECT count(*) FROM sqlite_master WHERE type='table' AND name={table_name}""")
#     df = pd.read_sql_query(f'''SELECT * FROM {table_name}''', conn)
#
#     conn.commit()
#     conn.close()
#
#     return df
#
#
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
#
#     save_location = path_to_dir + '/output_data'
#
#     # BASIC RUN ----------------------------------------
#     def basic_run_RICE(years_10, regions, save_location):
#         # RICE(years_10, regions).run(write_to_excel=False, file_name='Basic RICE - Nordhaus Policy - 2')
#         title_of_run = 'test_RICE_verification'
#         start = time.time()
#         RICE(years_10, regions, save_location=save_location, file_name=title_of_run).run(write_to_excel=True, write_to_sqlite=False)
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start)/60} minutes.')
#
#     def basic_run_RICE_with_scenarios(years_10, regions, save_location):
#         title_of_run = 'TEST_SSP1_1_scenario'
#         start = time.time()
#         levers = {'mu_target': 2135,
#                   'sr': 0.248,
#                   'irstp': 0.015}
#         scenario1 = {'SSP_scenario': 1,                                        # 1, 2, 3, 4, 5
#                     'fosslim': 11720,                                          # range(4000, 13650), depending on SSP scenario
#                     'climate_sensitivity_distribution': 'lognormal',          # 'log', 'lognormal', 'Cauchy'
#                     'elasticity_climate_impact': 0,                          # -1, 0, 1
#                     'price_backstop_tech': 1.260,                             # [1.260, 1.470, 1.680, 1.890]
#                     'negative_emissions_possible': 'no',                    # 'yes' or 'no'
#                     't2xco2_index': 800, }                                  # 0, 999
#         RICE(years_10, regions, scenario=scenario1, levers=levers, save_location=save_location, file_name=title_of_run).run(write_to_excel=True, write_to_sqlite=False)
#
#         # scenario2 = {'SSP_scenario': 2,  # 1, 2, 3, 4, 5
#         #             'fosslim': 9790,  # range(4000, 13650), depending on SSP scenario
#         #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
#         #             'elasticity_climate_impact': 0,  # -1, 0, 1
#         #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
#         #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
#         # RICE(years_10, regions, scenario=scenario2, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
#         #                                                               file_name='SSP2_Emissions.xlsx')
#         #
#         # scenario3 = {'SSP_scenario': 3,  # 1, 2, 3, 4, 5
#         #             'fosslim': 7860,  # range(4000, 13650), depending on SSP scenario
#         #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
#         #             'elasticity_climate_impact': 0,  # -1, 0, 1
#         #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
#         #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
#         # RICE(years_10, regions, scenario=scenario3, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
#         #                                                               file_name='SSP3_Emissions.xlsx')
#         #
#         # scenario4 = {'SSP_scenario': 4,  # 1, 2, 3, 4, 5
#         #             'fosslim': 5930,  # range(4000, 13650), depending on SSP scenario
#         #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
#         #             'elasticity_climate_impact': 0,  # -1, 0, 1
#         #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
#         #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
#         # RICE(years_10, regions, scenario=scenario4, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
#         #                                                               file_name='SSP4_Emissions.xlsx')
#         #
#         # scenario5 = {'SSP_scenario': 5,  # 1, 2, 3, 4, 5
#         #             'fosslim': 4000,  # range(4000, 13650), depending on SSP scenario
#         #             'climate_sensitivity_distribution': 'lognormal',  # 'log', 'lognormal', 'Cauchy'
#         #             'elasticity_climate_impact': 0,  # -1, 0, 1
#         #             'price_backstop_tech': 1.260,  # [1.260, 1.470, 1.680, 1.890]
#         #             'negative_emissions_possible': 'no'}  # 'yes' or 'no'
#         # RICE(years_10, regions, scenario=scenario5, levers=levers).run(write_to_excel=True, write_to_sqlite=False,
#         #                                                               file_name='SSP5_Emissions.xlsx')
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start)/60} minutes.')
#
#     # CONNECT TO EMA -----------------------------------
#     def connect_to_EMA(years_10, regions, save_location):
#         # # Now all parameters must be given in the experiments.py file, this function simply calls it. Must fix later.
#         # ConnectToEMA()
#         title_of_run = '200 czsnfe_reference_scenario_lever_search_period_util_global_damages_temp_overshoots'
#         start = time.time()
#         def RICE_wrapper_ema_workbench(years_10,
#                                        regions,
#                                        SSP_scenario=None,
#                                        fosslim=None,
#                                        climate_sensitivity_distribution=None,
#                                        elasticity_climate_impact=None,
#                                        price_backstop_tech=None,
#                                        negative_emissions_possible=None,  # 1 = 'no'; 1.2 = 'yes'
#                                        t2xco2_index=None,
#                                        mu_target=2135,
#                                        sr=0.248,
#                                        irstp=0.015):
#             '''
#             This wrapper connects the RICE model to the ema workbench. The ema workbench requires that the uncertainties and
#             levers of the model are given as direct inputs to the model. The RICE model instead accepts dictionaries of the
#             uncertainties and levers. This wrapper takes the input uncertainties and levers and puts them in a dictionary
#             that serves as the input to RICE.
#             '''
#             # years_10 = []
#             # for i in range(2005, 2315, 10):
#             #     years_10.append(i)
#             #
#             # regions = [
#             #     "US",
#             #     "OECD-Europe",
#             #     "Japan",
#             #     "Russia",
#             #     "Non-Russia Eurasia",
#             #     "China",
#             #     "India",
#             #     "Middle East",
#             #     "Africa",
#             #     "Latin America",
#             #     "OHI",
#             #     "Other non-OECD Asia",
#             # ]
#             # if negative_emissions_possible == 0:
#             #     negative_emissions_possible = 'no'
#             # elif negative_emissions_possible == 1:
#             #     negative_emissions_possible = 'yes'
#             # else:
#             #     print('incorrect input for negative_emissions_possible variable')
#             #
#             # if climate_sensitivity_distribution == 0:
#             #     climate_sensitivity_distribution = 'log'
#             # elif climate_sensitivity_distribution == 1:
#             #     climate_sensitivity_distribution = 'lognormal'
#             # elif climate_sensitivity_distribution == 2:
#             #     climate_sensitivity_distribution = 'Cauchy'
#             # else:
#             #     print('incorrect input for climate_sensitivity_distribution variable')
#
#             if SSP_scenario is None: # Assume that all other uncertainties are None as well
#                 scenario = None
#             else:
#                 scenario = {'SSP_scenario': SSP_scenario,
#                         'fosslim': fosslim,
#                         'climate_sensitivity_distribution': climate_sensitivity_distribution,
#                         'elasticity_climate_impact': elasticity_climate_impact,
#                         'price_backstop_tech': price_backstop_tech,
#                         'negative_emissions_possible': negative_emissions_possible,
#                         't2xco2_index': t2xco2_index}
#
#             levers = {'mu_target': mu_target,
#                       'sr': sr,
#                       'irstp': irstp}
#             utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3 = RICE(
#                 years_10, regions, scenario=scenario, levers=levers).ema_workbench_control()
#             return utilitarian_objective_function_value1, utilitarian_objective_function_value2, utilitarian_objective_function_value3
#
#         # Set up RICE model -----------------------------------------------------------------------------------------------
#         model = Model("WrapperRICE", function=RICE_wrapper_ema_workbench)
#
#         # specify model constants
#         model.constants = [
#             Constant("years_10", years_10),
#             Constant("regions", regions)
#         ]
#
#         # specify uncertainties
#         model.uncertainties = [
#             IntegerParameter("SSP_scenario", 1, 5),
#             RealParameter("fosslim", 4000, 13650),
#             IntegerParameter("climate_sensitivity_distribution", 0, 2),
#             IntegerParameter("elasticity_climate_impact", -1, 1),
#             RealParameter("price_backstop_tech", 1.260, 1.890),
#             IntegerParameter("negative_emissions_possible", 0, 1),
#             IntegerParameter("t2xco2_index", 0, 999),
#         ]
#
#         # set levers
#         model.levers = [
#             IntegerParameter("mu_target", 2055, 2305),
#             RealParameter("sr", 0.1, 0.5),
#             RealParameter("irstp", 0.001, 0.1),
#         ]
#
#         # specify outcomes
#         model.outcomes = [
#             ScalarOutcome("utilitarian_objective_function_value1", ScalarOutcome.MINIMIZE),
#             ScalarOutcome("utilitarian_objective_function_value2", ScalarOutcome.MINIMIZE),
#             ScalarOutcome("utilitarian_objective_function_value3", ScalarOutcome.MINIMIZE),
#         ]
#
#         # Set up experiments ---------------------------------------------------------------------------------------------
#         ema_logging.log_to_stderr(ema_logging.INFO)
#
#         # Uncertainty sampling ------------------------------------------------------
#
#         # with SequentialEvaluator(model) as evaluator:
#         #     # results = evaluator.perform_experiments(scenarios=100000, policies=1)
#         #     results = evaluator.perform_experiments(scenarios=20)
#
#         # Save results ----------------------------------------------------------------------------------------------------
#         # save_results(results, f'{save_location}/{title_of_run}.tar.gz')
#
#         # Lever search --------------------------------------------------------------
#
#         from ema_workbench.em_framework.optimization import EpsilonProgress
#         # from ema_workbench import MultiprocessingEvaluator
#
#         convergence_metrics = [
#             EpsilonProgress()
#         ]
#
#         with SequentialEvaluator(model) as evaluator:
#             results, convergence = evaluator.optimize(
#                 nfe=200,
#                 searchover="levers",
#                 epsilons=[
#                              0.01,
#                          ]
#                          * len(model.outcomes),
#                 convergence=convergence_metrics,
#             )
#
#         # fig, ax1 = plt.subplots(ncols=1, sharex=True, figsize=(8, 4))
#         # ax1.plot(convergence.nfe, convergence.epsilon_progress)
#         # ax1.set_ylabel("$\epsilon$-progress")
#         #
#         # ax1.set_xlabel("number of function evaluations")
#         # plt.show()
#
#         results.to_csv(f'{save_location}/{title_of_run}.csv')
#
#         # fh = f'{save_location}/{title_of_run}.tar.gz'
#         # experiments, outcomes = load_results(fh)
#         # print(experiments)
#
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start)/60} minutes.')
#
#     # POLICY TREE OPTIMIZATION WITH FOLSOM -------------
#
#     def optimization_Folsom_Herman(save_location):
#         title_of_run = 'TEST_Folsom_Herman_5000nfe'
#         np.random.seed(42)
#
#         model = Folsom('folsom/data/folsom-daily-w2016.csv',
#                        sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)
#
#         algorithm = PTreeOpt(model.f,
#                              feature_bounds=[[0, 1000], [1, 365], [0, 300]],
#                              feature_names=['Storage', 'Day', 'Inflow'],
#                              discrete_actions=True,
#                              action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
#                                            'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
#                              mu=20,  # number of parents per generation
#                              cx_prob=0.70,  # crossover probability
#                              population_size=100,
#                              max_depth=5,
#                              multiobj=True,
#                              epsilons=[0.01, 1000, 0.01, 10]
#                              )
#
#         logging.basicConfig(level=logging.INFO,
#                             format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
#
#         # With only 1000 function evaluations this will not be very good
#         best_solution, best_score, snapshots = algorithm.run(max_nfe=5000,
#                                                              log_frequency=100,
#                                                              snapshot_frequency=100)
#
#         pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         pickle.dump(best_solution, open(f'{save_location}/{title_of_run}_best_solution.pkl', 'wb'))
#         pickle.dump(best_score, open(f'{save_location}/{title_of_run}_best_score.pkl', 'wb'))
#         return
#
#     def optimization_Folsom_ForestBorg(save_location):
#         title_of_run = 'TEST_Folsom_ForestBorg_10000nfe'
#         start = time.time()
#
#         master_rng = np.random.default_rng(42)  # Master RNG
#         df_optimized_metrics, snapshots = ForestBorgFolsom(pop_size=100, master_rng=master_rng,
#                                           years_10=years_10,
#                                           regions=regions,
#                                           metrics=['period_utility', 'damages', 'temp_overshoots'],
#                                           # Tree variables
#                                           action_names=['Release_Demand', 'Hedge_90', 'Hedge_80',
#                                                         'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
#                                           action_bounds=None,
#                                           feature_bounds=[[0, 1000], [1, 365], [0, 300]],
#                                           feature_names=['Storage', 'Day', 'Inflow'],
#                                           max_depth=5,
#                                           discrete_actions=True,
#                                           discrete_features=None,
#                                           # Optimization variables
#                                           mutation_prob=0.5,
#                                           max_nfe=10000,
#                                           epsilons=[0.01, 1000, 0.01, 10],
#                                           gamma=4,
#                                           tau=0.02,
#                                           save_location=save_location,
#                                           title_of_run=title_of_run,
#                                           ).run()
#         # df_optimized_metrics.to_excel(f'{save_location}/{title_of_run}.xlsx')
#         pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start) / 60} minutes.')
#
#         # logging.basicConfig(level=logging.INFO,
#         #                     format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
#         #
#         # # With only 1000 function evaluations this will not be very good
#         # best_solution, best_score, snapshots = algorithm.run(max_nfe=5000,
#         #                                                      log_frequency=100,
#         #                                                      snapshot_frequency=100)
#         #
#         # pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         # pickle.dump(best_solution, open(f'{save_location}/{title_of_run}_best_solution.pkl', 'wb'))
#         # pickle.dump(best_score, open(f'{save_location}/{title_of_run}_best_score.pkl', 'wb'))
#         # return
#
#     def optimization_Folsom_Shotgun(save_location):
#         title_of_run = 'TEST_Folsom_Shotgun'
#         start = time.time()
#
#         model = Folsom('folsom/data/folsom-daily-w2016.csv',
#                        sd='1995-10-01', ed='2016-09-30', use_tocs=False, multiobj=True)
#         master_rng = np.random.default_rng(42)  # Master RNG
#         snapshots = Shotgun(model=model,
#                     master_rng=master_rng,
#                     metrics=['period_utility', 'damages', 'temp_overshoots'],
#                     # Tree variables
#                     action_names=['Release_Demand', 'Hedge-90', 'Hedge-80',
#                                   'Hedge-70', 'Hedge-60', 'Hedge-50', 'Flood_Control'],
#                     action_bounds=None,
#                     feature_bounds=[[0, 1000], [1, 365], [0, 300]],
#                     feature_names=['Storage', 'Day', 'Inflow'],
#                     max_depth=5,
#                     discrete_actions=True,
#                     discrete_features=None,
#                     # Optimization variables
#                     mutation_prob=0.5,
#                     pop_size=10,
#                     max_nfe=2000,
#                     epsilons=np.array([0.01, 1000, 0.01, 10]),
#                     ).run()
#         # df_optimized_metrics.to_excel(f'{save_location}/{title_of_run}.xlsx')
#         pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start) / 60} minutes.')
#
#     # POLICY TREE OPTIMIZATION WITH RICE ---------------
#     def optimization_RICE_POT_Herman(years_10, regions, save_location, seed):
#         title_of_run = f'HermanRICE_100000nfe_random_other_levers_seed_{seed}'
#         start = time.time()
#         # input_path = os.path.join(package_directory)
#         # model = RICE(years_10, regions, database_POT=input_path+'/ptreeopt/output_data/POT_Experiments.db', table_name_POT='indicator_groupsize_3_bin_tournament_1')
#         # model = RICE(years_10, regions, save_location=save_location, file_name=title_of_run)
#         np.random.seed(seed)
#         model = RICE(years_10, regions)
#         algorithm = PTreeOpt(model.POT_control_Herman,
#                              # feature_bounds=[[0.8, 2.8], [700, 900], [2005, 2305]],
#                              # feature_names=['temp_atm', 'mat', 'year'],
#                              # feature_bounds=[[2005, 2305]],
#                              # feature_names=['year'],
#                              # feature_bounds=[[0.8, 2.8]],
#                              # feature_names=['temp_atm'],
#                              feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                              feature_names=['mat', 'net_output', 'year'],
#                              discrete_features=None,
#                              discrete_actions=True,
#                              # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low',
#                              #               'miu_2100_sr_high', 'miu_2125_sr_high', 'miu_2150_sr_high'],
#                              # action_names=['miu_2100_sr_low', 'miu_2150_sr_high'],
#                              # action_names=['miu_2100_sr_low', 'miu_2125_sr_low', 'miu_2150_sr_low'],
#                              # action_names=['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05'],
#                              action_names=['miu_2065',
#                                            'miu_2100',
#                                            'miu_2180',
#                                            'miu_2250',
#                                            'miu_2305',
#                                             'sr_01',
#                                            'sr_02',
#                                            'sr_03',
#                                            'sr_04',
#                                            'sr_05',
#                                            'irstp_0005',
#                                            'irstp_0015',
#                                            'irstp_0025'],
#                              # action_bounds=[[2065, 2305], [0.1, 0.5], [0.01, 0.1]],
#                              mu=20,  # number of parents per generation, 20
#                              cx_prob=0.70,  # crossover probability
#                              population_size=100,  # 100
#                              max_depth=5,
#                              epsilons=[0.05, 0.05, 0.05],
#                              multiobj=True
#                              )
#
#         logging.basicConfig(level=logging.INFO,
#                             format='[%(processName)s/%(levelname)s:%(filename)s:%(funcName)s] %(message)s')
#
#         # With only 1000 function evaluations this will not be very good
#         best_solution, best_score, snapshots = algorithm.run(max_nfe=100000,
#                                                              log_frequency=100,
#                                                              snapshot_frequency=100)
#         # print(best_solution)
#         # print(best_score)
#         # print(snapshots)
#         #
#         # ## View POT data ---------------------------------------------------------------
#         # # df = view_sqlite_database(database=input_path + '/ptreeopt/output_data/POT_Experiments.db',
#         # #                           table_name='indicator_groupsize_3_bin_tournament_1')
#         # df = view_sqlite_database(database=save_location + '/Experiments.db', table_name=title_of_run)
#         # print(df.head())
#         # print(df.info())
#         # df.to_excel(f'{save_location}/{title_of_run}.xlsx')
#
#         pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start)/60} minutes.')
#
#     # def optimization_RICE_POT_Borg(years_10, regions, save_location):
#     #     # from POT.optimization import PolicyTreeOptimizer
#     #     #
#     #     # model = RICE(years_10, regions)
#     #     # feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#     #     # feature_names = ['mat', 'net_output', 'year']
#     #     # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#     #     # PolicyTreeOptimizer(model.POT_control, feature_bounds=feature_bounds,
#     #     #                     feature_names=feature_names,
#     #     #                     action_names=action_names,
#     #     #                     discrete_actions=True,
#     #     #                     population_size=4,
#     #     #                     mu=2).run(max_nfe=4)
#     #
#     #     # np.random.seed(1)
#     #
#     #     title_of_run = ''
#     #     start = time.time()
#     #     # Model variables
#     #
#     #     # Tree variables
#     #     # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#     #     action_names = ['miu', 'sr', 'irstp']
#     #     action_bounds = [[2065, 2305], [0.1, 0.5], [0.001, 0.1]]
#     #     feature_names = ['mat', 'net_output', 'year']
#     #     feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#     #     # Save variables
#     #     # database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
#     #     # table_name_POT = 'Test3_couplingborg_not_edited_borg'
#     #
#     #     df_optimized_metrics = PolicyTreeOptimizer(model=RICE(years_10, regions, save_location=save_location, file_name=title_of_run),
#     #                         # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
#     #                         action_names=action_names,
#     #                         action_bounds=action_bounds,
#     #                         discrete_actions=False,
#     #                         feature_names=feature_names,
#     #                         feature_bounds=feature_bounds,
#     #                         discrete_features=False,
#     #                         epsilon=0.01,
#     #                         max_nfe=100000,
#     #                         max_depth=4,
#     #                         population_size=100
#     #                         ).run()
#     #     df_optimized_metrics.to_excel(f'{save_location}/{title_of_run}.xlsx')
#     #     end = time.time()
#     #     return print(f'Total elapsed time: {(end - start)/60} minutes.')
#
#     def optimization_RICE_POT_Homemade(years_10, regions, save_location):
#         title_of_run = ''
#         start = time.time()
#         master_rng = np.random.default_rng(42)  # Master RNG
#         run = Cluster(20, 80, master_rng=master_rng,
#                 years_10=years_10,
#                 regions=regions,
#                 metrics=['period_utility', 'utility', 'temp_overshoots'],
#                 # Tree variables
#                 action_names=['miu', 'sr', 'irstp'],
#                 action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                 feature_names=['mat', 'net_output', 'year'],
#                 feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                 max_depth=4,
#                 discrete_actions=False,
#                 discrete_features=False,
#                 # Optimization variables
#                 mutation_prob=0.5,
#                 max_nfe=30000,
#                 epsilons=np.array([0.05, 0.05]),
#                 ).run()
#
#         with pd.ExcelWriter(f'{save_location}/{title_of_run}.xlsx', engine='xlsxwriter') as writer:
#             run[0].to_excel(writer, sheet_name='graveyard')
#             run[1].to_excel(writer, sheet_name='VIPs')
#             run[2].to_excel(writer, sheet_name='pareto front')
#             run[3].to_excel(writer, sheet_name='convergence')
#
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start) / 60} minutes.')
#
#     def optimization_RICE_POT_Control(years_10, regions, save_location):
#         # from POT.optimization import PolicyTreeOptimizer
#         #
#         # model = RICE(years_10, regions)
#         # feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#         # feature_names = ['mat', 'net_output', 'year']
#         # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#         # PolicyTreeOptimizer(model.POT_control, feature_bounds=feature_bounds,
#         #                     feature_names=feature_names,
#         #                     action_names=action_names,
#         #                     discrete_actions=True,
#         #                     population_size=4,
#         #                     mu=2).run(max_nfe=4)
#
#         # np.random.seed(1)
#
#         title_of_run = 'Control_500000nfe_depth_4__mat_netoutput_year_periodutility_damages_tmpovershoots'
#         start = time.time()
#         # Model variables
#
#         # Tree variables
#         # action_names = ['miu_2100', 'miu_2150', 'miu_2200', 'miu_2125', 'sr_02', 'sr_03', 'sr_04', 'sr_05']
#         action_names = ['miu', 'sr', 'irstp']
#         action_bounds = [[2065, 2305], [0.1, 0.5], [0.001, 0.1]]
#         feature_names = ['mat', 'net_output', 'year']
#         feature_bounds = [[780, 1300], [55, 2300], [2005, 2305]]
#         # Save variables
#         # database_POT = 'C:/Users/Stijn Daemen/Documents/master thesis TU Delft/code/IAM_RICE2/jupyter notebooks/Tests_Borg.db'
#         # table_name_POT = 'Test3_couplingborg_not_edited_borg'
#
#         df_optimized_metrics = PolicyTreeOptimizerControl(
#             model=RICE(years_10, regions, save_location=save_location, file_name=title_of_run),
#             # model=RICE(years_10, regions, database_POT=database_POT, table_name_POT=table_name_POT),
#             action_names=action_names,
#             action_bounds=action_bounds,
#             discrete_actions=False,
#             feature_names=feature_names,
#             feature_bounds=feature_bounds,
#             discrete_features=False,
#             epsilon=0.05,
#             max_nfe=500000,
#             max_depth=4,
#             population_size=100
#         ).run()
#         df_optimized_metrics.to_excel(f'{save_location}/{title_of_run}.xlsx')
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start) / 60} minutes.')
#
#     def optimization_RICE_POT_Homemade_advanced(years_10, regions, save_location):
#         master_rng = np.random.default_rng(42)  # Master RNG
#         title_of_run = ''
#         start = time.time()
#
#         run12 = Cluster_(20, 80, master_rng=master_rng,
#                         years_10=years_10,
#                         regions=regions,
#                         metrics=['period_utility', 'damages', 'temp_overshoots'],
#                         metrics_choice=[0, 1],
#                         # Tree variables
#                         action_names=['miu', 'sr', 'irstp'],
#                         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                         feature_names=['mat', 'net_output', 'year'],
#                         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                         max_depth=4,
#                         discrete_actions=False,
#                         discrete_features=False,
#                         # Optimization variables
#                         mutation_prob=0.5,
#                         max_nfe=3000,
#                         epsilons=np.array([0.01, 0.01]),
#                         P_ref=[-10, 0.01], ).run()
#
#         run23 = Cluster_(20, 80, master_rng=master_rng,
#                         years_10=years_10,
#                         regions=regions,
#                         metrics=['period_utility', 'damages', 'temp_overshoots'],
#                         metrics_choice=[1, 2],
#                         # Tree variables
#                         action_names=['miu', 'sr', 'irstp'],
#                         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                         feature_names=['mat', 'net_output', 'year'],
#                         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                         max_depth=4,
#                         discrete_actions=False,
#                         discrete_features=False,
#                         # Optimization variables
#                         mutation_prob=0.5,
#                         max_nfe=3000,
#                         epsilons=np.array([0.01, 0.01]),
#                         P_ref=[0.01, 1],
#                         pareto_front=None, ).run()
#
#         run31 = Cluster_(20, 80, master_rng=master_rng,
#                         years_10=years_10,
#                         regions=regions,
#                         metrics=['period_utility', 'damages', 'temp_overshoots'],
#                         metrics_choice=[2, 0],
#                         # Tree variables
#                         action_names=['miu', 'sr', 'irstp'],
#                         action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                         feature_names=['mat', 'net_output', 'year'],
#                         feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                         max_depth=4,
#                         discrete_actions=False,
#                         discrete_features=False,
#                         # Optimization variables
#                         mutation_prob=0.5,
#                         max_nfe=30000,
#                         epsilons=np.array([0.01, 0.01]),
#                         P_ref=[1, -10],
#                         pareto_front=None, ).run()
#
#         pareto_front_combined = run12 + run23 + run31
#
#         run123 = Cluster_(20, 80, master_rng=master_rng,
#                          years_10=years_10,
#                          regions=regions,
#                          metrics=['period_utility', 'damages', 'temp_overshoots'],
#                          metrics_choice=[0, 1, 2],
#                          # Tree variables
#                          action_names=['miu', 'sr', 'irstp'],
#                          action_bounds=[[2100, 2250], [0.2, 0.5], [0.01, 0.1]],
#                          feature_names=['mat', 'net_output', 'year'],
#                          feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                          max_depth=4,
#                          discrete_actions=False,
#                          discrete_features=False,
#                          # Optimization variables
#                          mutation_prob=0.5,
#                          max_nfe=3000,
#                          epsilons=np.array([0.01, 0.01, 0.01]),
#                          P_ref=[-10, 0.01, 1],
#                          pareto_front=pareto_front_combined,
#                          ).run()
#         df = pd.DataFrame(run123,
#                           columns=['pareto_front'])
#         df.to_excel(f'{save_location}/{title_of_run}.xlsx')
#         end = time.time()
#         print(f'Elapsed time: {(end - start) / 60} minutes.')
#         return
#
#     def optimization_RICE_POT_ForestBorg(years_10, regions, save_location, seed):
#         # title_of_run = 'ForestBORG_500000nfe_tree_depth_4_population_100_mat_net_output_year_continuous_period_utility_damages_tempovershoots'
#         title_of_run = 'TESTFB_1000nfe'#f'ForestborgRICE_100000nfe_seed_{seed}'
#         start = time.time()
#
#         model = RICE(years_10, regions)
#         master_rng = np.random.default_rng(seed)  # Master RNG
#         snapshots = ForestBorg(pop_size=100, master_rng=master_rng,
#                                           model=model.POT_control,
#                                           # model=model.f,
#                                           metrics=['period_utility', 'damages', 'temp_overshoots'],
#                                           # Tree variables
#                                           action_names=['miu', 'sr', 'irstp'],
#                                           action_bounds=[[2065, 2305], [0.1, 0.5], [0.01, 0.1]],
#                                           feature_names=['mat', 'net_output', 'year'],
#                                           feature_bounds=[[780, 1300], [55, 2300], [2005, 2305]],
#                                           max_depth=5,
#                                           discrete_actions=False,
#                                           discrete_features=False,
#                                           # Optimization variables
#                                           mutation_prob=0.5,
#                                           max_nfe=10000,
#                                           epsilons=np.array([0.05, 0.05, 0.05]),
#                                           gamma=2,
#                                           tau=0.02,
#                                           save_location=save_location,
#                                           title_of_run=title_of_run,
#                                           ).run()
#         pickle.dump(snapshots, open(f'{save_location}/{title_of_run}_snapshots.pkl', 'wb'))
#         end = time.time()
#         return print(f'Total elapsed time: {(end - start) / 60} minutes.')
#
#
#     # basic_run_RICE(years_10, regions, save_location)
#
#     # optimization_RICE_POT_Borg(years_10, regions, save_location)
#     # optimization_RICE_POT_Control(years_10, regions, save_location)
#     # optimization_RICE_POT_Homemade_advanced(years_10, regions, save_location)
#
#     # basic_run_RICE(years_10, regions, save_location)
#
#     # connect_to_EMA(years_10, regions, save_location)
#
#     # seeds = [5, 26, 17, 55, 104, 506]
#     seeds = [42]
#     for seed in seeds:
#         # optimization_RICE_POT_Herman(years_10, regions, save_location, seed=seed)
#         optimization_RICE_POT_ForestBorg(years_10, regions, save_location, seed=seed)
#
#     # optimization_Folsom_Herman(save_location)
#
#     # optimization_Folsom_ForestBorg(save_location)
#
#     # optimization_Folsom_Shotgun(save_location)
#
#     # basic_run_RICE_with_scenarios(years_10, regions, save_location)
