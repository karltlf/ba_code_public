import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
target_dir_name = 'ba_code_project'
while True:
    # Check if the target directory exists in the current directory
    potential_target = os.path.join(current_dir, target_dir_name)
    if os.path.isdir(potential_target):
        code_root_dir = potential_target
        break
    # Move one level up
    parent_dir = os.path.dirname(current_dir)
    # If we're at the root of the file system and still haven't found it, stop
    if parent_dir == current_dir:
        code_root_dir = None
        break
    current_dir = parent_dir
if code_root_dir:
    # Add the found target directory to sys.path
    sys.path.append(code_root_dir)
    print(f'Found and appended to sys.path: {code_root_dir}')
else:
    print(f'Target directory not found.')

import json
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from src.features.get_x_y_tuple_list import get_x_y_tuple_list
from src.visualization.render_vehicle_track_cluster_in_notebook import render_vehicle_track_cluster_in_notebook
from src.models.optics.get_clusters_from_optics_labels import get_clusters_from_optics_labels
from src.visualization.plot_vehicle_tracks_in_notebook import plot_vehicle_tracks_in_notebook

def plot_within_cluster_optics(intersection_name, params_uid, cluster_label, eval_metric, within_cluster_params_uid=None):

    print(f'------------------------within CLUSTER #{cluster_label} for INTERSECTION {intersection_name}------------------------')

    # get data
    data_path = f'{code_root_dir}/data/processed/{intersection_name}_cuid.csv'
    df_cuid = pd.read_csv(data_path)
    df_cuid_grouped_path = data_path.replace('.csv', '_grouped.csv')
    df_cuid_grouped = pd.read_csv(df_cuid_grouped_path)
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(lambda x: ast.literal_eval(x))
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(lambda y: ast.literal_eval(y))
    list_x_y_tuples = get_x_y_tuple_list(df_cuid_grouped, ['x','y'])

    # ----------------- UNDERLYING OPTICS -----------------
    def get_optics_params(optics_params, metric):
        optics_params = optics_params[metric]
        dtw_dist_matrix_key = optics_params['dtw_key']
        max_eps = optics_params['epsilon']
        min_samples = optics_params['min_samples']
        cluster_method = optics_params['cluster_method']
        xi = optics_params['xi']
        kwargs = {
            'max_eps': max_eps,
            'min_samples': min_samples,
            'cluster_method': cluster_method,
            'xi': xi
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return dtw_dist_matrix_key, filtered_kwargs
    
    # check for params uid
    if params_uid is not None:
        opt_params_path = f'{code_root_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params_{params_uid}.json'
    else:
        opt_params_path = f'{code_root_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params.json'
    with open(opt_params_path, 'r') as f:
        optics_params = json.load(f)

    # import optimized parameters
    dtw_matrix_key, kwargs_acc_score = get_optics_params(optics_params, eval_metric)

    # import dtw distance matrix
    dtw_distance_matrix_path = f'{code_root_dir}/data/processed/{intersection_name}_diff_itakura_slope_dtw_matrices.json'
    with open(dtw_distance_matrix_path) as f:
        dtw_distance_matrices_dict = json.load(f)

    # get the distance matrix
    dtw_distance_matrix = dtw_distance_matrices_dict[dtw_matrix_key]

    # create optics model
    optics_model = OPTICS(metric='precomputed', **kwargs_acc_score).fit(dtw_distance_matrix)

    # get labels for big models
    labels = get_clusters_from_optics_labels(optics_model.labels_)

    



    # ----------------- WITHIN CLUSTER OPTICS -----------------
    # path to optimized within cluster optics params
    path_optimized_in_cluster_optics_params = f'{code_root_dir}/data/processed/within_cluster_clustering_optimization/{intersection_name}_optics_optimized_vehicle_paths_{params_uid}_{eval_metric}_within_cluster_{cluster_label}_optimized_params_{within_cluster_params_uid}.json'

    # load optimized within cluster optics params
    try:
        with open(path_optimized_in_cluster_optics_params, 'r') as f:
            within_cluster_optics_params = json.load(f)
    except FileNotFoundError as e:
        print(f'File not found: {path_optimized_in_cluster_optics_params}')
        _, ax = plt.subplots(1, 1, figsize=(10,10))
        # filter labels to include whole cluster
        render_vehicle_track_cluster_in_notebook(ax, df_cuid, df_cuid_grouped, {'0': labels[cluster_label]})
        plot_vehicle_tracks_in_notebook(ax, df_cuid, df_cuid_grouped, color='gray', alpha=0.5, title='')
        return
    
    # extract within cluster optics params according to eval metric
    within_cluster_optics_params = within_cluster_optics_params[eval_metric]

    # extract within cluster optics params
    within_cluster_max_eps = within_cluster_optics_params['epsilon']
    within_cluster_min_samples = within_cluster_optics_params['min_samples']
    within_cluster_cluster_method = within_cluster_optics_params['cluster_method']
    within_cluster_xi = within_cluster_optics_params['xi'] if within_cluster_optics_params['xi'] is not None else 0.05
    within_cluster_dtw_key = within_cluster_optics_params['dtw_key']

    # create filtered data using certain cluster
    df_grouped_within_cluster_filtered = df_cuid_grouped.loc[optics_model.labels_ == cluster_label] 

    # extract optimized within cluster dtw distance matrix
    within_cluster_dtw_distance_matrix = dtw_distance_matrices_dict[within_cluster_dtw_key]

    # create within cluster dtw distance matrices by filtering only for the cluster
    within_cluster_dtw_distance_matrix = np.array(within_cluster_dtw_distance_matrix)
    filtered_rows = within_cluster_dtw_distance_matrix[optics_model.labels_ == cluster_label]
    within_cluster_dtw_distance_matrix = filtered_rows[:, optics_model.labels_ == cluster_label]

    # fit new within cluster optics model
    optics_within_cluster = OPTICS(metric='precomputed', 
                                   max_eps=within_cluster_max_eps, 
                                   min_samples=within_cluster_min_samples, 
                                   cluster_method=within_cluster_cluster_method, 
                                   xi=within_cluster_xi
                                   ).fit(within_cluster_dtw_distance_matrix)

    # get labels for within cluster model
    labels_within_cluster = get_clusters_from_optics_labels(optics_within_cluster.labels_)


    # plot within cluster clusters
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    title_str = f'OPTICS within Cluster #{cluster_label}' + '\n'\
        + f'Optimization init ID: {params_uid if params_uid is not None else "n/a"}' + '\n' + \
        f'Optimization within cluster ID: {within_cluster_params_uid if within_cluster_params_uid is not None else "n/a"}'
    render_vehicle_track_cluster_in_notebook(axs, df_cuid, df_grouped_within_cluster_filtered, labels_within_cluster)
    plot_vehicle_tracks_in_notebook(axs, df_cuid, df_cuid_grouped, '', color='gray', alpha=0.5)