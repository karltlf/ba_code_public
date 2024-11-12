
import os
import sys

import pandas as pd

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.realpath(__file__))
ba_code_project_dir = os.path.abspath(os.path.join(script_dir, '../../../../..'))
sys.path.append(ba_code_project_dir)

from src.models.optics.get_dtw_optics_params import get_dtw_optics_params
import json
import numpy as np
from sklearn.cluster import OPTICS

def get_subcluster_indices_for_cluster(intersection_name, 
                                            id_init_params_intersection_optics_model, 
                                            id_init_params_in_cluster_optics, 
                                            intersection_eval_metric,
                                            in_cluster_eval_metric,
                                            dtw_distance_matrix_dict,
                                            ):
    
    # load dataframe
    df_cuid_grouped = pd.read_csv(f'{ba_code_project_dir}/data/processed/{intersection_name}_cuid_grouped.csv', index_col=0)

    # Load intersection optics parameters and distance matrix
    if id_init_params_intersection_optics_model is None:
        with open(f'{ba_code_project_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params.json') as f:
            intersection_optics_optimized_params = json.load(f)
    else:
        with open(f'{ba_code_project_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params_{id_init_params_intersection_optics_model}.json') as f:
            intersection_optics_optimized_params = json.load(f)

    # extract optimized dtw distance matrix and parameters
    intersection_dtw_matrix_key, intersection_optics_optimized_params = get_dtw_optics_params(intersection_optics_optimized_params, intersection_eval_metric)
    intersection_dtw_distance_matrix = dtw_distance_matrix_dict[intersection_dtw_matrix_key]

    # Fit optics model for the full dataset
    optics_intersection = OPTICS(**intersection_optics_optimized_params, metric='precomputed').fit(intersection_dtw_distance_matrix)

    # get all cluster IDs without outliers (-1)
    labels = optics_intersection.labels_
    cluster_ids = np.unique(labels[labels != -1])

    subcluster_indices = {}

    # Iterate over all clusters
    for cluster_id in cluster_ids:

        # Load in-cluster optics parameters and distance matrix for the specific cluster
        try:
            with open(f'{ba_code_project_dir}/data/processed/within_cluster_clustering_optimization/'
                    f'{intersection_name}_optics_optimized_vehicle_paths_{id_init_params_intersection_optics_model}_'
                    f'{intersection_eval_metric}_within_cluster_{cluster_id}_optimized_params_{id_init_params_in_cluster_optics}.json') as f:
                in_cluster_optics_optimized_params = json.load(f)

        # if no in-cluster clustering optimization results are found, return all indices of the cluster
        except FileNotFoundError:
            # print(f'No in-cluster clustering optimization results found for cluster {cluster_id}.')
            # print(f'Returning all indices of cluster {cluster_id} in cluster 0.')
            subcluster_indices[cluster_id] = {}
            subcluster_indices[cluster_id][0] = df_cuid_grouped.index[optics_intersection.labels_ == cluster_id].tolist()
            continue

        # extract optimized dtw distance matrix and parameters
        in_cluster_dtw_matrix_key, in_cluster_optics_optimized_params = get_dtw_optics_params(in_cluster_optics_optimized_params, in_cluster_eval_metric)
        in_cluster_dtw_distance_matrix = dtw_distance_matrix_dict[in_cluster_dtw_matrix_key]

        # Filter data for the specific cluster
        filtered_rows = np.array(in_cluster_dtw_distance_matrix)[optics_intersection.labels_ == cluster_id]
        in_cluster_dtw_distance_matrix = filtered_rows[:, optics_intersection.labels_ == cluster_id]

        # Fit optics model for the in-cluster data
        optics_in_cluster = OPTICS(**in_cluster_optics_optimized_params, metric='precomputed').fit(in_cluster_dtw_distance_matrix)

        # iterate over subclusters
        subcluster_indices[cluster_id] = {}
        for subcluster_id in np.unique(optics_in_cluster.labels_):
            subcluster_indices[cluster_id][subcluster_id] = df_cuid_grouped.index[optics_intersection.labels_ == cluster_id][optics_in_cluster.labels_ == subcluster_id].tolist()


    return subcluster_indices

