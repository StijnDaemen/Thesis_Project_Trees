import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import hvwfg
import pickle
import numpy as np
import os
import pandas as pd
import networkx as nx
import matplotlib.patches as patches
import seaborn as sns

package_directory = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(package_directory)


def Pickle(file_path):
    with open(file_path, "rb") as file:
        snapshots = pickle.load(file)
    return snapshots


def calculate_generational_hypervolume(pareto_front_generations, reference_point):
    hypervolume_metric = np.array([])
    for generation in pareto_front_generations:
        generation = np.array(generation)
        hypervolume_metric = np.append(hypervolume_metric, hvwfg.wfg(generation, reference_point))
    return hypervolume_metric


def calculate_min_max(all_snapshots):
    # Calculation of minimum and maximum values for each objective, considering pareto fronts from all seeds
    min_vals = []
    max_vals = []
    for snapshots in all_snapshots:
        try:
            seed_PF = np.concatenate((seed_PF, snapshots['Archive_solutions'][-1]), axis=0)
        except:
            seed_PF = snapshots['Archive_solutions'][-1]
    stacked = np.column_stack(seed_PF)
    min_vals.append([np.min(stacked, axis=1)])
    min_val = np.min(min_vals, axis=0)[0]
    max_vals.append([np.max(stacked, axis=1)])
    max_val = np.max(max_vals, axis=0)[0]
    return min_val, max_val


def normalize_objectives(snapshots, min_val, max_val):
    snapshots['normalized_objectives'] = []
    for gen in snapshots['Archive_solutions']:
        gen_PF = []
        for sol in gen:
            sol_norm = np.array((sol - min_val) / (max_val - min_val))
            gen_PF.append(sol_norm)
        snapshots['normalized_objectives'].append(gen_PF)
    return snapshots


def calculate_runtime_metrics(snapshots, reference_point):
    snapshots['hypervolume_metric'] = calculate_generational_hypervolume(snapshots['normalized_objectives'],
                                                                         reference_point)
    try:
        snapshots['epsilon_progress_metric'] = snapshots['epsilon_progress']
    except:
        print('No epsilon progress available')
    return snapshots


def transform_objectives(organisms):
    data_dict = {}
    for objective in range(1, len(organisms[0]) + 1):
        data_dict[f'ofv{objective}'] = []

    for idx, item in enumerate(organisms):
        for key_idx, key in enumerate(data_dict.keys()):
            data_dict[key].append(item[key_idx])
    return data_dict


def find_minimum_objective_row(df, column):
    # Find the index of the minimum value in column 'A'
    min_index = df[column].idxmin()
    # Retrieve the row with the minimum value in column 'A'
    min_row = df.loc[min_index]
    return min_row, min_index


def find_average_row(df):
    # Calculate the mean for each column
    means = df.mean()

    # Function to calculate Euclidean distance
    def euclidean_distance(row, means):
        return np.sqrt(((row - means) ** 2).sum())

    # Apply the function to each row and find the index of the row with the smallest distance
    closest_index = df.apply(euclidean_distance, axis=1, args=(means,)).idxmin()
    # Retrieve the row that is closest to the mean
    closest_row = df.loc[closest_index]

    avg_row = closest_row
    avg_index = closest_index
    return avg_row, avg_index


def graphviz_export(P, filename, colordict=None, animation=False, dpi=300):
    ''' Export policy tree P to filename (SVG or PNG)
    colordict optional. Keys must match actions. Example:
    colordict = {'Release_Demand': 'cornsilk',
            'Hedge_90': 'indianred',
            'Flood_Control': 'lightsteelblue'}
    Requires pygraphviz.'''

    import pygraphviz as pgv
    G = pgv.AGraph(directed=True)
    G.node_attr['shape'] = 'box'
    G.node_attr['style'] = 'filled'

    if animation:
        G.graph_attr['size'] = '2!,2!'  # use for animations only
        G.graph_attr['dpi'] = str(dpi)

    parent = P.root
    G.add_node(str(parent), fillcolor='lightsteelblue')
    S = []

    while parent.is_feature or len(S) > 0:
        if parent.is_feature:
            S.append(parent)
            child = parent.l
            label = 'T'

        else:
            parent = S.pop()
            child = parent.r
            label = 'F'

        if child.is_feature or not colordict:
            c = 'lightsteelblue'  # 'white'
        else:
            c = 'lightgreen' #  colordict[child.value]

        G.add_node(str(child), fillcolor=c)
        G.add_edge(str(parent), str(child), label=label)
        parent = child

    G.layout(prog='dot')
    G.draw(filename)


if __name__ == '__main__':
    def create_Folsom_Figures(save=False):
        # Collect ForestBORG files
        file_names = []
        seeds = [17, 42, 104, 303, 902]
        for seed in seeds:
            file_names.append(
                f'Folsom_ForestBORG_discrete_seed{seed}_nfe30000_depth3_epsilons[1.e-02 1.e+03 1.e-02 1.e+01]_gamma4_tau0.02_restart5000_v1_snapshots')

        # Collect Original POT files
        file_names_Herman = []
        for seed in seeds:
            file_names_Herman.append(
                f'Folsom_Herman_seed{seed}_nfe30000_depth3_epsilons[0.01, 1000, 0.01, 10]_v1_snapshots')

        # Combine all files from ForestBORG and Herman to determine the min and max value for each objective accross all available pareto fronts
        snapshots_Herman_FB = []
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            snapshots_Herman_FB.append(snapshots)
        for file_name_Herman in file_names_Herman:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
            snapshots['Archive_solutions'] = snapshots.pop('best_f')
            snapshots_Herman_FB.append(snapshots)

        # Determine min and max value over all runs
        extremes = calculate_min_max(snapshots_Herman_FB)
        min_val = extremes[0]
        max_val = extremes[1]

        # Unpack all snapshots, so for each seed (only for ForestBORG files)
        all_snapshots = []
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            all_snapshots.append(snapshots)

        # Normalize objective values
        for idx, snapshots in enumerate(all_snapshots):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            all_snapshots[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(all_snapshots):
            reference_point = np.array([1, 1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            all_snapshots[idx] = snapshots

        # Do the same for the Herman files
        # Unpack all snapshots, so for each seed
        all_snapshots_Herman = []
        for file_name_Herman in file_names_Herman:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
            snapshots['Archive_solutions'] = snapshots.pop('best_f')
            all_snapshots_Herman.append(snapshots)

        # Normalize objective values
        for idx, snapshots in enumerate(all_snapshots_Herman):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            all_snapshots_Herman[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(all_snapshots_Herman):
            reference_point = np.array([1, 1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            all_snapshots_Herman[idx] = snapshots

        # -- Figure I: Runtime Dynamics ------------------
        plt.figure(figsize=(10, 6))
        xminF = 0
        xmaxF = 30500
        yminF = 0.1  # 1.7e08
        ymaxF = 0.9  # 8e08
        for idx, d in enumerate(all_snapshots):
            if idx == 0:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
            else:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
            plt.legend(loc='lower right', fontsize='medium', frameon=True)
            plt.xlim([xminF, xmaxF])
            plt.ylim([yminF, ymaxF])
            # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
            plt.xlabel('Number of Function Evaluations', fontsize=11)
            plt.ylabel('Hypervolume', fontsize=11)
            plt.grid(True, which='both', linestyle='--', linewidth=0.8)

        for idx, d in enumerate(all_snapshots_Herman):
            if idx == 0:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#cb181d', label='Original POT')
            else:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#cb181d')
            plt.legend(loc='lower right', fontsize='medium', frameon=True)
            plt.xlim([xminF, xmaxF])
            plt.ylim([yminF, ymaxF])
            # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
            plt.xlabel('Number of Function Evaluations', fontsize=11)
            plt.ylabel('Hypervolume', fontsize=11)
            plt.grid(True, which='both', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save:
            file_name = f'Folsom_FB_runtime_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure II: Operator Dynamics ------------------

        # OPTION 1: ORIGINAL FIGURE - STACKED AREA CHART
        fig, axs = plt.subplots(1, 5, figsize=(20, 8))
        xminF = 0
        xmaxF = 28500
        yminF = 0  # 1.7e08
        ymaxF = 280  # 8e08
        for idx, d in enumerate(all_snapshots):
            index_rng = [x for x in range(len(list(d['mutation_operators'].values())[0]))]
            data = d['mutation_operators']
            df = pd.DataFrame(data, index=index_rng)
            # Stacked area chart
            axs[idx].stackplot(df.index, df.T, labels=df.columns, alpha=0.5)
            axs[idx].set_xlim([xminF, xmaxF])
            axs[idx].set_ylim([yminF, ymaxF])
            axs[idx].set_title(f'seed {idx+1}', fontsize=12)
            axs[2].set_xlabel('Number of Function Evaluations', fontsize=11)
            axs[0].set_ylabel('Operator Occurance', fontsize=11)
            axs[idx].grid(True, which='both', linestyle='--', linewidth=0.8)
            # axs[0].legend(loc='lower left')
            axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        # plt.tight_layout()

        if save:
            file_name = f'Folsom_FB_operator_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure IV: Parallel Axis Plots --------------
        snapshots = all_snapshots[3]
        data = transform_objectives(snapshots['normalized_objectives'][-1])
        df = pd.DataFrame(data)

        # Find the solutions that stand out: the minimum solution per objective and one average solutino across all objectives
        selected_rows = []
        selected_indeces = []
        selected_names = ['Best Cost', 'Best Reliability', 'Best Carryover Storage', 'Best Flooding', 'Compromise Solution']
        for col in df.columns:
            row, idx = find_minimum_objective_row(df, col)
            selected_rows.append(row)
            selected_indeces.append(idx)
        average_solution = find_average_row(df)
        selected_rows.append(average_solution[0])
        selected_indeces.append(average_solution[1])

        # Add the tree structures
        df['trees'] = snapshots['Archive_trees'][-1]

        # Edit df for the Parallel Axis Plot
        df_PAP = df[['ofv1', 'ofv2', 'ofv3', 'ofv4']]

        # PAP function needs to have the differently colored lines to be at the back of the dataframe otherwise they will be pushed to the background
        for idx in selected_indeces:
            df_PAP = pd.concat([df_PAP, df_PAP.loc[[idx]]], axis=0, ignore_index=False)

        # Custom names, change per model
        df_PAP.rename(columns={'ofv1': 'Cost ($billion, discounted)', 'ofv2': 'Reliability (volumetric)', 'ofv3': 'Carryover storage (# below 5000 TAF)', 'ofv4': 'Flooding (cumulative)'},
                      inplace=True)
        df_PAP['Name'] = 'All Solutions'
        for idx, index in enumerate(selected_indeces):
            df_PAP.loc[index, 'Name'] = f'{selected_names[idx]}'

        # Create the parallel axis plot
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df_PAP, 'Name', color=('grey', 'blue', 'purple', 'pink', 'orange', 'green'))  # color=('grey', 'blue', 'green', 'orange', 'red', 'black')
        # plt.title('Folsom Lake Model')
        arrow = '←'  # or '\u2190'
        plt.ylabel(f'{arrow} Direction of preference')
        plt.grid(False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

        # Access the lines in the plot and set different linewidths
        ax = plt.gca()
        lines = ax.get_lines()
        linewidths = [1]*len(df_PAP['Name'])
        for index in selected_indeces:
            linewidths[index] = 5
        linewidths[-5] = 5
        linewidths[-4] = 5
        linewidths[-3] = 5
        linewidths[-2] = 5
        linewidths[-1] = 5
        for line, lw in zip(lines, linewidths):
            line.set_linewidth(lw)

        if save:
            file_name = f'Folsom_FB_ParallelAxisPlot'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure V: Individual Policy Trees --------------
        # Take the solutions from Figure IV
        # colors = {'Release_Demand': 'cornsilk',
        #           'Hedge_90': 'indianred',
        #           'Hedge_80': 'indianred',
        #           'Hedge_70': 'indianred',
        #           'Hedge_60': 'indianred',
        #           'Hedge_50': 'indianred',
        #           'Flood_Control': 'lightsteelblue'}
        colors = {'Release_Demand': 'lightgreen',
                  'Hedge_90': 'lightgreen',
                  'Hedge_80': 'lightgreen',
                  'Hedge_70': 'lightgreen',
                  'Hedge_60': 'lightgreen',
                  'Hedge_50': 'lightgreen',
                  'Flood_Control': 'lightgreen'}

        for index, idx in enumerate(selected_indeces):
            P = df.loc[idx]['trees']
            file_name = f'Folsom_FB_individual_tree_{selected_names[index]}'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            graphviz_export(P, file_path, colordict=colors, animation=False, dpi=300)

        # -- Figure III: Controllability map -------------
        # Collect ForestBORG files
        file_names = []
        depths = [2, 3, 4, 5, 6]
        gammas = [3, 4, 5]
        restarts = [500, 2000, 5000]
        for restart in restarts:
            for depth in depths:
                for gamma in gammas:
                    file_names.append(
                        f'Folsom_ForestBORG_discrete_seed42_nfe20000_depth{depth}_epsilons[1.e-02 1.e+03 1.e-02 1.e+01]_gamma{gamma}_tau0.02_restart{restart}_v1_snapshots')

        # Unpack all snapshots
        control_snapshots = []
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            control_snapshots.append(snapshots)

        # Find global min and max values
        extremes = calculate_min_max(control_snapshots)
        min_val = extremes[0]
        max_val = extremes[1]

        # Normalize objective values
        for idx, snapshots in enumerate(control_snapshots):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            control_snapshots[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(control_snapshots):
            reference_point = np.array([1, 1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            control_snapshots[idx] = snapshots

        control_snapshots_500 = control_snapshots[0:15]
        control_snapshots_2000 = control_snapshots[15:30]
        control_snapshots_5000 = control_snapshots[30:45]

        def create_array_for_controllability_map(snapshot_list):
            def take_last_hypervolume_value(data_list, amount):
                return sum(data_list[-amount:]) / amount

            array = np.zeros((len(depths), len(gammas)))
            row_counter = 0
            col_counter = 0
            for i, item in enumerate(snapshot_list):
                i = i + 1
                array[row_counter, col_counter] = take_last_hypervolume_value(item['hypervolume_metric'], 5)
                col_counter += 1
                if i % 3 == 0:
                    col_counter = 0
                    row_counter += 1
            return array

        array_500 = create_array_for_controllability_map(control_snapshots_500)
        array_2000 = create_array_for_controllability_map(control_snapshots_2000)
        array_5000 = create_array_for_controllability_map(control_snapshots_5000)

        # Determine the global min and max
        vmin = min(array_500.min(), array_2000.min(), array_5000.min())
        vmax = max(array_500.max(), array_2000.max(), array_5000.max())

        x_labels = gammas
        y_labels = depths

        # Set up a matplotlib figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False, gridspec_kw={'wspace': 0.2, 'hspace': 0})

        plot_titles = ['a', 'b', 'c']
        # Plot each heatmap with custom x and y labels
        for i, data in enumerate([array_500, array_2000, array_5000]):
            sns.heatmap(data, ax=axes[i], annot=True, fmt=".2f", cmap='viridis', vmin=vmin, vmax=vmax, cbar=False,
                        xticklabels=x_labels, yticklabels=y_labels)  # cmap='plasma'
            axes[i].set_title(f'{plot_titles[i]}) Restart interval every {restarts[i]} iterations', fontsize=12)
            if i == 1:
                axes[i].set_xlabel('archive-to-population ratio target', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('maximum tree depth', fontsize=12)

        # Adjust the tick labels if needed
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Create a colorbar
        cbar = plt.colorbar(axes[0].get_children()[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Hypervolume', fontsize=12)

        if save:
            file_name = f'Folsom_FB_controllability_map'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        return

    def create_RICE_discrete_Figures(save=False):
        # Collect ForestBORG files
        file_names_discrete = []
        seeds = [17, 42, 104, 303, 902]
        for seed in seeds:
            file_names_discrete.append(
                f'RICE_ForestBORG_discrete_seed{seed}_nfe30000_depth3_epsilons[0.05 0.05 0.05]_gamma4_tau0.02_restart5000_v1_snapshots')

        # Collect Original POT files
        file_names_Herman = []
        for seed in seeds:
            file_names_Herman.append(
                f'RICE_Herman_seed{seed}_nfe30000_depth3_epsilons[0.05, 0.05, 0.05]_v1_snapshots')

        # Combine all files from ForestBORG and Herman to determine the min and max value for each objective accross all available pareto fronts
        snapshots_Herman_FB = []
        for file_name_discrete in file_names_discrete:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_discrete}.pkl')
            snapshots_Herman_FB.append(snapshots)
        for file_name_Herman in file_names_Herman:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
            snapshots['Archive_solutions'] = snapshots.pop('best_f')
            snapshots_Herman_FB.append(snapshots)

        # Determine min and max value over all runs
        extremes = calculate_min_max(snapshots_Herman_FB)
        min_val = extremes[0]
        max_val = extremes[1]

        # Unpack all snapshots, so for each seed (only for ForestBORG files)
        all_snapshots = []
        for file_name_discrete in file_names_discrete:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_discrete}.pkl')
            all_snapshots.append(snapshots)

        # Normalize objective values
        for idx, snapshots in enumerate(all_snapshots):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            all_snapshots[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(all_snapshots):
            reference_point = np.array([1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            all_snapshots[idx] = snapshots

        # Do the same for the Herman files
        # Unpack all snapshots, so for each seed
        all_snapshots_Herman = []
        for file_name_Herman in file_names_Herman:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_Herman}.pkl')
            snapshots['Archive_solutions'] = snapshots.pop('best_f')
            all_snapshots_Herman.append(snapshots)

        # Normalize objective values
        for idx, snapshots in enumerate(all_snapshots_Herman):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            all_snapshots_Herman[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(all_snapshots_Herman):
            reference_point = np.array([1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            all_snapshots_Herman[idx] = snapshots

        # -- Figure I: Runtime Dynamics ------------------
        plt.figure(figsize=(10, 6))
        xminF = 0
        xmaxF = 30500
        yminF = 0.1  # 1.7e08
        ymaxF = 0.9  # 8e08
        for idx, d in enumerate(all_snapshots):
            if idx == 0:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
            else:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
            plt.legend(loc='lower right', fontsize='medium', frameon=True)
            # plt.xlim([xminF, xmaxF])
            # plt.ylim([yminF, ymaxF])
            # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
            plt.xlabel('Number of Function Evaluations', fontsize=11)
            plt.ylabel('Hypervolume', fontsize=11)
            plt.grid(True, which='both', linestyle='--', linewidth=0.8)

        for idx, d in enumerate(all_snapshots_Herman):
            if idx == 0:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#cb181d', label='Original POT')
            else:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#cb181d')
            plt.legend(loc='lower right', fontsize='medium', frameon=True)
            # plt.xlim([xminF, xmaxF])
            # plt.ylim([yminF, ymaxF])
            # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
            plt.xlabel('Number of Function Evaluations', fontsize=11)
            plt.ylabel('Hypervolume', fontsize=11)
            plt.grid(True, which='both', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save:
            file_name = f'RICE_FB_discrete_runtime_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure II: Operator Dynamics ------------------

        # OPTION 1: ORIGINAL FIGURE - STACKED AREA CHART
        fig, axs = plt.subplots(1, 5, figsize=(20, 8))
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4, fancybox=True, shadow=True)
        xminF = 0
        xmaxF = 28500
        yminF = 0  # 1.7e08
        ymaxF = 280  # 8e08
        for idx, d in enumerate(all_snapshots):
            index_rng = [x for x in range(len(list(d['mutation_operators'].values())[0]))]
            data = d['mutation_operators']
            df = pd.DataFrame(data, index=index_rng)
            # Stacked area chart
            axs[idx].stackplot(df.index, df.T, labels=df.columns, alpha=0.5)
            # axs[idx].set_xlim([xminF, xmaxF])
            # axs[idx].set_ylim([yminF, ymaxF])
            axs[idx].set_title(f'seed {idx+1}', fontsize=12)
            axs[2].set_xlabel('Number of Function Evaluations', fontsize=11)
            axs[0].set_ylabel('Operator Occurance', fontsize=11)
            axs[idx].grid(True, which='both', linestyle='--', linewidth=0.8)
            axs[0].legend(loc='lower left')
            # axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4, fancybox=True, shadow=True)  # ncol=4,
        plt.tight_layout()
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), fancybox=True, shadow=True)  # ncol=4,

        if save:
            file_name = f'RICE_FB_discrete_operator_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure IV: Parallel Axis Plots --------------
        snapshots = all_snapshots[3]
        data = transform_objectives(snapshots['normalized_objectives'][-1])
        df = pd.DataFrame(data)

        # Find the solutions that stand out: the minimum solution per objective and one average solutino across all objectives
        selected_rows = []
        selected_indeces = []
        selected_names = ['Best Welfare', 'Best Damages', 'Best Temp. Overshoots', 'Compromise Solution']
        for col in df.columns:
            row, idx = find_minimum_objective_row(df, col)
            selected_rows.append(row)
            selected_indeces.append(idx)
        average_solution = find_average_row(df)
        selected_rows.append(average_solution[0])
        selected_indeces.append(average_solution[1])

        # Add the tree structures
        df['trees'] = snapshots['Archive_trees'][-1]

        # Edit df for the Parallel Axis Plot
        df_PAP = df[['ofv1', 'ofv2', 'ofv3']]

        # PAP function needs to have the differently colored lines to be at the back of the dataframe otherwise they will be pushed to the background
        for idx in selected_indeces:
            df_PAP = pd.concat([df_PAP, df_PAP.loc[[idx]]], axis=0, ignore_index=False)

        # Custom names, change per model
        df_PAP.rename(columns={'ofv1': 'Welfare (utility)', 'ofv2': 'Damages ()', 'ofv3': 'Temp. Overshoots (cumulative)'},
                      inplace=True)
        df_PAP['Name'] = 'All Solutions'
        for idx, index in enumerate(selected_indeces):
            df_PAP.loc[index, 'Name'] = f'{selected_names[idx]}'

        # Create the parallel axis plot
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df_PAP, 'Name', color=('grey', 'blue', 'green', 'orange', 'red', 'black'))
        # plt.title('Folsom Lake Model')
        arrow = '←'  # or '\u2190'
        plt.ylabel(f'{arrow} Direction of preference')
        plt.grid(False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

        if save:
            file_name = f'RICE_FB_discrete_ParallelAxisPlot'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure V: Individual Policy Trees --------------
        # Take the solutions from Figure IV
        # colors = {'Release_Demand': 'cornsilk',
        #           'Hedge_90': 'indianred',
        #           'Hedge_80': 'indianred',
        #           'Hedge_70': 'indianred',
        #           'Hedge_60': 'indianred',
        #           'Hedge_50': 'indianred',
        #           'Flood_Control': 'lightsteelblue'}
        # colors = {
        #     'miu_2065': 'lightsteelblue',
        #     'miu_2100': 'cornsilk',
        #     'miu_2180': 'cornsilk',
        #     'miu_2250': 'cornsilk',
        #     'miu_2305': 'indianred',
        #     'sr_01': 'lightsteelblue',
        #     'sr_02': 'cornsilk',
        #     'sr_03': 'cornsilk',
        #     'sr_04': 'cornsilk',
        #     'sr_05': 'indianred',
        #     'irstp_0005': 'lightsteelblue',
        #     'irstp_0015': 'cornsilk',
        #     'irstp_0025': 'indianred'
        # }
        colors = {
            'miu_2065': 'lightgreen',
            'miu_2100': 'lightgreen',
            'miu_2180': 'lightgreen',
            'miu_2250': 'lightgreen',
            'miu_2305': 'lightgreen',
            'sr_01': 'lightgreen',
            'sr_02': 'lightgreen',
            'sr_03': 'lightgreen',
            'sr_04': 'lightgreen',
            'sr_05': 'lightgreen',
            'irstp_0005': 'lightgreen',
            'irstp_0015': 'lightgreen',
            'irstp_0025': 'lightgreen'
        }

        for index, idx in enumerate(selected_indeces):
            P = df.loc[idx]['trees']
            file_name = f'RICE_FB_discrete_individual_tree_{selected_names[index]}'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            graphviz_export(P, file_path, colordict=colors, animation=False, dpi=300)

        # # -- Figure III: Controllability map -------------
        # # Collect ForestBORG files
        # file_names = []
        # depths = [2, 3, 4, 5, 6]
        # gammas = [3, 4, 5]
        # restarts = [500, 2000, 5000]
        # for restart in restarts:
        #     for depth in depths:
        #         for gamma in gammas:
        #             file_names.append(
        #                 f'RICE_ForestBORG_discrete_seed42_nfe20000_depth{depth}_epsilons[0.05 0.05 0.05]_gamma{gamma}_tau0.02_restart{restart}_v1_snapshots')
        #
        # # Unpack all snapshots
        # control_snapshots = []
        # for file_name in file_names:
        #     snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
        #     control_snapshots.append(snapshots)
        #
        # # Find global min and max values
        # extremes = calculate_min_max(control_snapshots)
        # min_val = extremes[0]
        # max_val = extremes[1]
        #
        # # Normalize objective values
        # for idx, snapshots in enumerate(control_snapshots):
        #     snapshots = normalize_objectives(snapshots, min_val, max_val)
        #     control_snapshots[idx] = snapshots
        #
        # # Calculate metrics
        # for idx, snapshots in enumerate(control_snapshots):
        #     reference_point = np.array([1, 1, 1]).astype(np.float64)
        #     snapshots = calculate_runtime_metrics(snapshots, reference_point)
        #     control_snapshots[idx] = snapshots
        #
        # control_snapshots_500 = control_snapshots[0:15]
        # control_snapshots_2000 = control_snapshots[15:30]
        # control_snapshots_5000 = control_snapshots[30:45]
        #
        # def create_array_for_controllability_map(snapshot_list):
        #     def take_last_hypervolume_value(data_list, amount):
        #         return sum(data_list[-amount:]) / amount
        #
        #     array = np.zeros((len(depths), len(gammas)))
        #     row_counter = 0
        #     col_counter = 0
        #     for i, item in enumerate(snapshot_list):
        #         i = i + 1
        #         array[row_counter, col_counter] = take_last_hypervolume_value(item['hypervolume_metric'], 5)
        #         col_counter += 1
        #         if i % 3 == 0:
        #             col_counter = 0
        #             row_counter += 1
        #     return array
        #
        # array_500 = create_array_for_controllability_map(control_snapshots_500)
        # array_2000 = create_array_for_controllability_map(control_snapshots_2000)
        # array_5000 = create_array_for_controllability_map(control_snapshots_5000)
        #
        # # Determine the global min and max
        # vmin = min(array_500.min(), array_2000.min(), array_5000.min())
        # vmax = max(array_500.max(), array_2000.max(), array_5000.max())
        #
        # x_labels = gammas
        # y_labels = depths
        #
        # # Set up a matplotlib figure with 3 subplots
        # fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False, gridspec_kw={'wspace': 0.2, 'hspace': 0})
        #
        # plot_titles = ['a', 'b', 'c']
        # # Plot each heatmap with custom x and y labels
        # for i, data in enumerate([array_500, array_2000, array_5000]):
        #     sns.heatmap(data, ax=axes[i], annot=True, fmt=".2f", cmap='viridis', vmin=vmin, vmax=vmax, cbar=False,
        #                 xticklabels=x_labels, yticklabels=y_labels)  # cmap='plasma'
        #     axes[i].set_title(f'{plot_titles[i]}) Restart interval every {restarts[i]} iterations', fontsize=12)
        #     if i == 1:
        #         axes[i].set_xlabel('archive-to-population ratio target', fontsize=12)
        #     if i == 0:
        #         axes[i].set_ylabel('maximum tree depth', fontsize=12)
        #
        # # Adjust the tick labels if needed
        # for ax in axes:
        #     ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        #     ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        #
        # # Create a colorbar
        # cbar = plt.colorbar(axes[0].get_children()[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        # cbar.set_label('Hypervolume', fontsize=12)
        #
        # if save:
        #     file_name = f'RICE_FB_discrete_controllability_map'
        #     file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
        #     plt.savefig(file_path, dpi=300)
        # else:
        #     plt.show()

        return

    def create_RICE_continuous_Figures(save=False):
        # - Second for continuous action analysis ------------------------------

        # Collect ForestBORG files
        file_names_discrete = []
        seeds = [17, 42, 104, 303, 902]
        for seed in seeds:
            file_names_discrete.append(
                f'RICE_ForestBORG_continuous_seed{seed}_nfe30000_depth3_epsilons[0.05 0.05 0.05]_gamma4_tau0.02_restart5000_v1_snapshots')

        # Unpack all snapshots, so for each seed (only for ForestBORG files)
        all_snapshots = []
        for file_name_discrete in file_names_discrete:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name_discrete}.pkl')
            all_snapshots.append(snapshots)

        # Determine min and max value over all runs
        extremes = calculate_min_max(all_snapshots)
        min_val = extremes[0]
        max_val = extremes[1]

        # Normalize objective values
        for idx, snapshots in enumerate(all_snapshots):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            all_snapshots[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(all_snapshots):
            reference_point = np.array([1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            all_snapshots[idx] = snapshots

        # -- Figure I: Runtime Dynamics ------------------
        plt.figure(figsize=(10, 6))
        xminF = 0
        xmaxF = 30500
        yminF = 0.1  # 1.7e08
        ymaxF = 0.9  # 8e08
        for idx, d in enumerate(all_snapshots):
            if idx == 0:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
            else:
                plt.plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
            plt.legend(loc='lower right', fontsize='medium', frameon=True)
            # plt.xlim([xminF, xmaxF])
            # plt.ylim([yminF, ymaxF])
            # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
            plt.xlabel('Number of Function Evaluations', fontsize=11)
            plt.ylabel('Hypervolume', fontsize=11)
            plt.grid(True, which='both', linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save:
            file_name = f'RICE_FB_continuous_runtime_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure II: Operator Dynamics ------------------

        # OPTION 1: ORIGINAL FIGURE - STACKED AREA CHART
        fig, axs = plt.subplots(1, 5, figsize=(20, 8))
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=4, fancybox=True, shadow=True)
        xminF = 0
        xmaxF = 28500
        yminF = 0  # 1.7e08
        ymaxF = 280  # 8e08
        for idx, d in enumerate(all_snapshots):
            index_rng = [x for x in range(len(list(d['mutation_operators'].values())[0]))]
            data = d['mutation_operators']
            df = pd.DataFrame(data, index=index_rng)
            # Stacked area chart
            axs[idx].stackplot(df.index, df.T, labels=df.columns, alpha=0.5)
            # axs[idx].set_xlim([xminF, xmaxF])
            # axs[idx].set_ylim([yminF, ymaxF])
            axs[idx].set_title(f'seed {idx + 1}', fontsize=12)
            axs[2].set_xlabel('Number of Function Evaluations', fontsize=11)
            axs[0].set_ylabel('Operator Occurance', fontsize=11)
            axs[idx].grid(True, which='both', linestyle='--', linewidth=0.8)
            # axs[0].legend(loc='lower left')
            axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        # plt.tight_layout()

        if save:
            file_name = f'RICE_FB_continuous_operator_dynamics'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure IV: Parallel Axis Plots --------------
        snapshots = all_snapshots[3]
        data = transform_objectives(snapshots['normalized_objectives'][-1])
        df = pd.DataFrame(data)

        # Find the solutions that stand out: the minimum solution per objective and one average solutino across all objectives
        selected_rows = []
        selected_indeces = []
        selected_names = ['Best Welfare', 'Best Damages', 'Best Temp. Overshoots', 'Compromise Solution']
        for col in df.columns:
            row, idx = find_minimum_objective_row(df, col)
            selected_rows.append(row)
            selected_indeces.append(idx)
        average_solution = find_average_row(df)
        selected_rows.append(average_solution[0])
        selected_indeces.append(average_solution[1])

        # Add the tree structures
        df['trees'] = snapshots['Archive_trees'][-1]

        # Edit df for the Parallel Axis Plot
        df_PAP = df[['ofv1', 'ofv2', 'ofv3']]

        # PAP function needs to have the differently colored lines to be at the back of the dataframe otherwise they will be pushed to the background
        for idx in selected_indeces:
            df_PAP = pd.concat([df_PAP, df_PAP.loc[[idx]]], axis=0, ignore_index=False)

        # Custom names, change per model
        df_PAP.rename(
            columns={'ofv1': 'Welfare (utility)', 'ofv2': 'Damages (global)', 'ofv3': 'Temp. Overshoots (cumulative above 2 degrees)'},
            inplace=True)
        df_PAP['Name'] = 'All Solutions'
        for idx, index in enumerate(selected_indeces):
            df_PAP.loc[index, 'Name'] = f'{selected_names[idx]}'

        # Create the parallel axis plot
        plt.figure(figsize=(12, 6))
        parallel_coordinates(df_PAP, 'Name', color=('grey', 'blue', 'purple', 'pink', 'orange', 'green'))  # color=('grey', 'blue', 'green', 'orange', 'red', 'black'))
        # plt.title('Folsom Lake Model')
        arrow = '←'  # or '\u2190'
        plt.ylabel(f'{arrow} Direction of preference')
        plt.grid(False)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

        # Access the lines in the plot and set different linewidths
        ax = plt.gca()
        lines = ax.get_lines()
        linewidths = [1]*len(df_PAP['Name'])  # Specify your linewidths here
        for index in selected_indeces:
            linewidths[index] = 5
        linewidths[-4] = 5
        linewidths[-3] = 5
        linewidths[-2] = 5
        linewidths[-1] = 5
        for line, lw in zip(lines, linewidths):
            line.set_linewidth(lw)

        if save:
            file_name = f'RICE_FB_continuous_ParallelAxisPlot'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        # -- Figure V: Individual Policy Trees --------------
        # Take the solutions from Figure IV
        colors = {'miu_2200.415|sr_0.33|irstp_0.038|': 'lightgreen',
                  'miu_2075.045|sr_0.0101|irstp_0.01|': 'lightgreen',
                  'miu_2163.979|sr_0.225|irstp_0.064|': 'lightgreen',
                  'miu_2235.902|sr_0.126|irstp_0.053|': 'lightgreen',
                  'miu_2295.311|sr_0.161|irstp_0.01|': 'lightgreen',
                  'miu_2103.046|sr_0.225|irstp_0.01|': 'lightgreen',
                  'miu_2077.239|sr_0.225|irstp_0.01|': 'lightgreen',
                  'miu_2075.045|sr_0.101|irstp_0.01|': 'lightgreen',
                  'miu_2189.983|sr_0.176|irstp_0.099|': 'lightgreen',
                  'miu_2299.727|sr_0.5|irstp_0.01|': 'lightgreen',
                  'miu_2130.547|sr_0.148|irstp_0.01|': 'lightgreen',
                  'miu_2270.148|sr_0.388|irstp_0.095|': 'lightgreen',
                  'miu_2222.245|sr_0.119|irstp_0.01|': 'lightgreen',
                  'miu_2077.239|sr_0.229|irstp_0.01|': 'lightgreen',
                  }

        for index, idx in enumerate(selected_indeces):
            P = df.loc[idx]['trees']
            file_name = f'RICE_FB_continuous_individual_tree_{selected_names[index]}'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            graphviz_export(P, file_path, colordict=colors, animation=False, dpi=300)

        # -- Figure III: Controllability map -------------
        # Collect ForestBORG files
        file_names = []
        depths = [2, 3, 4, 5, 6]
        gammas = [3, 4, 5]
        restarts = [500, 2000, 5000]
        for restart in restarts:
            for depth in depths:
                for gamma in gammas:
                    file_names.append(
                        f'RICE_ForestBORG_continuous_seed42_nfe20000_depth{depth}_epsilons[0.05 0.05 0.05]_gamma{gamma}_tau0.02_restart{restart}_v1_snapshots')

        # Unpack all snapshots
        control_snapshots = []
        for file_name in file_names:
            snapshots = Pickle(path_to_dir + f'\output_data\{file_name}.pkl')
            control_snapshots.append(snapshots)

        # Find global min and max values
        extremes = calculate_min_max(control_snapshots)
        min_val = extremes[0]
        max_val = extremes[1]

        # Normalize objective values
        for idx, snapshots in enumerate(control_snapshots):
            snapshots = normalize_objectives(snapshots, min_val, max_val)
            control_snapshots[idx] = snapshots

        # Calculate metrics
        for idx, snapshots in enumerate(control_snapshots):
            reference_point = np.array([1, 1, 1]).astype(np.float64)
            snapshots = calculate_runtime_metrics(snapshots, reference_point)
            control_snapshots[idx] = snapshots

        control_snapshots_500 = control_snapshots[0:15]
        control_snapshots_2000 = control_snapshots[15:30]
        control_snapshots_5000 = control_snapshots[30:45]

        def create_array_for_controllability_map(snapshot_list):
            def take_last_hypervolume_value(data_list, amount):
                return sum(data_list[-amount:]) / amount

            array = np.zeros((len(depths), len(gammas)))
            row_counter = 0
            col_counter = 0
            for i, item in enumerate(snapshot_list):
                i = i + 1
                array[row_counter, col_counter] = take_last_hypervolume_value(item['hypervolume_metric'], 5)
                col_counter += 1
                if i % 3 == 0:
                    col_counter = 0
                    row_counter += 1
            return array

        array_500 = create_array_for_controllability_map(control_snapshots_500)
        array_2000 = create_array_for_controllability_map(control_snapshots_2000)
        array_5000 = create_array_for_controllability_map(control_snapshots_5000)

        # Determine the global min and max
        vmin = min(array_500.min(), array_2000.min(), array_5000.min())
        vmax = max(array_500.max(), array_2000.max(), array_5000.max())

        x_labels = gammas
        y_labels = depths

        # Set up a matplotlib figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False, gridspec_kw={'wspace': 0.2, 'hspace': 0})

        plot_titles = ['a', 'b', 'c']
        # Plot each heatmap with custom x and y labels
        for i, data in enumerate([array_500, array_2000, array_5000]):
            sns.heatmap(data, ax=axes[i], annot=True, fmt=".2f", cmap='viridis', vmin=vmin, vmax=vmax, cbar=False,
                        xticklabels=x_labels, yticklabels=y_labels)  # cmap='plasma'
            axes[i].set_title(f'{plot_titles[i]}) Restart interval every {restarts[i]} iterations', fontsize=12)
            if i == 1:
                axes[i].set_xlabel('archive-to-population ratio target', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('maximum tree depth', fontsize=12)

        # Adjust the tick labels if needed
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # Create a colorbar
        cbar = plt.colorbar(axes[0].get_children()[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Hypervolume', fontsize=12)

        if save:
            file_name = f'RICE_FB_continuous_controllability_map'
            file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
            plt.savefig(file_path, dpi=300)
        else:
            plt.show()

        return


    create_Folsom_Figures(save=True)
    # create_RICE_discrete_Figures(save=True)
    create_RICE_continuous_Figures(save=True)





# # OPTION 2: BOXPLOTS
# # Strategy: Sum occurances for each operator throughout a run. Then do a boxplot for each seed on this operator
#
# data_list = [[], [], [], [], [], [], []]
# for snapshots in all_snapshots:
#     index_rng = [x for x in range(len(list(snapshots['mutation_operators'].values())[0]))]
#     data = snapshots['mutation_operators']
#     df = pd.DataFrame(data, index=index_rng)
#
#     for idx, col in enumerate(df.columns):
#         # print(df[col][index_rng[-1]])
#         last_value = df[col][index_rng[-1]]
#         data_list[idx].append(last_value)
#
# print(data_list)
# fig = plt.figure(figsize=(10, 7))
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
# # Creating plot
# bp = ax.boxplot(data_list)
# # show plot
# plt.show()
# -----------------------------------------------------------------------------------------------------------------
# fig, axs = plt.subplots(1, 2, figsize=(10, 8))
# xminF = 0
# xmaxF = 30500
# yminF = 0.1  # 1.7e08
# ymaxF = 0.9  # 8e08
# for idx, d in enumerate(all_snapshots):
#     if idx == 0:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
#     else:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
#     axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
#     axs[0].set_xlim([xminF, xmaxF])
#     axs[0].set_ylim([yminF, ymaxF])
#     # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
#     axs[0].set_xlabel('Number of Function Evaluations', fontsize=11)
#     axs[0].set_ylabel('Hypervolume', fontsize=11)
#     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
#
#     # if idx == 0:
#     #     axs[1].plot(d['nfe'], d['epsilon_progress_metric'], color='#225ea8', label='ForestBORG')
#     # else:
#     #     axs[1].plot(d['nfe'], d['epsilon_progress_metric'], color='#225ea8')
#     # axs[1].legend(loc='lower right', fontsize='medium', frameon=True)
#     # # axs[1].set_xlim([xminF, xmaxF])
#     # # axs[1].set_ylim([yminF, ymaxF])
#     # # axs[1].set_title(f'Folsom Lake Model', fontsize=12)
#     # axs[1].set_xlabel('Number of Function Evaluations', fontsize=11)
#     # axs[1].set_ylabel('Epsilon progress indicator', fontsize=11)
#     # axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
#
# for idx, d in enumerate(all_snapshots_Herman):
#     if idx == 0:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#cb181d', label='Original POT')
#     else:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#cb181d')
#     axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
#     axs[0].set_xlim([xminF, xmaxF])
#     axs[0].set_ylim([yminF, ymaxF])
#     # axs[0].set_title(f'Folsom Lake Model', fontsize=12)
#     axs[0].set_xlabel('Number of Function Evaluations', fontsize=11)
#     axs[0].set_ylabel('Hypervolume', fontsize=11)
#     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
#
# plt.tight_layout()
# ----------------------------------------------------------------------------------------------
# -- Figure I: Runtime Dynamics ------------------
# fig, axs = plt.subplots(1, 2, figsize=(10, 8))
# xminD = 0
# xmaxD = 30500
# yminD = 0.1  # 1.7e08
# ymaxD = 0.9  # 8e08
# xminC = 0
# xmaxC = 30500
# yminC = 0.1  # 1.7e08
# ymaxC = 0.9  # 8e08
# for idx, d in enumerate(all_snapshots):
#     if idx == 0:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
#     else:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
#     axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
#     # axs[0].set_xlim([xminD, xmaxD])
#     # axs[0].set_ylim([yminD, ymaxD])
#     axs[0].set_title(f'Discrete actions', fontsize=12)
#     axs[0].set_xlabel('Number of Function Evaluations', fontsize=11)
#     axs[0].set_ylabel('Hypervolume', fontsize=11)
#     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
#
# for idx, d in enumerate(all_snapshots_Herman):
#     if idx == 0:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#cb181d', label='Original POT')
#     else:
#         axs[0].plot(d['nfe'], d['hypervolume_metric'], color='#cb181d')
#     axs[0].legend(loc='lower right', fontsize='medium', frameon=True)
#     # axs[0].set_xlim([xminD, xmaxD])
#     # axs[0].set_ylim([yminD, ymaxD])
#     axs[0].set_title(f'Discrete actions', fontsize=12)
#     axs[0].set_xlabel('Number of Function Evaluations', fontsize=11)
#     axs[0].set_ylabel('Hypervolume', fontsize=11)
#     axs[0].grid(True, which='both', linestyle='--', linewidth=0.8)
#
# # for idx, d in enumerate(all_snapshots):
# #     if idx == 0:
# #         axs[1].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8', label='ForestBORG')
# #     else:
# #         axs[1].plot(d['nfe'], d['hypervolume_metric'], color='#225ea8')
# #     axs[1].legend(loc='lower right', fontsize='medium', frameon=True)
# #     # axs[1].set_xlim([xminC, xmaxC])
# #     # axs[1].set_ylim([yminC, ymaxC])
# #     axs[1].set_title(f'Continuous actions', fontsize=12)
# #     axs[1].set_xlabel('Number of Function Evaluations', fontsize=11)
# #     axs[1].set_ylabel('Hypervolume', fontsize=11)
# #     axs[1].grid(True, which='both', linestyle='--', linewidth=0.8)
#
# plt.tight_layout()
#
# if save:
#     file_name = f'RICE_FB_discrete_runtime_dynamics'
#     file_path = path_to_dir + f'\output_data\Figs\{file_name}.png'
#     plt.savefig(file_path, dpi=300)
# else:
#     plt.show()
