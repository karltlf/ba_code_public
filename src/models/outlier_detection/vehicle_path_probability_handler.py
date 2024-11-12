import json
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
import os
import sys
import numpy as np
import pandas as pd
import ast
from tslearn.metrics import dtw, dtw_path_from_metric
from scipy.stats import percentileofscore
from concurrent.futures import ThreadPoolExecutor

# Import necessary modules from your project (adjust the paths as needed)
# Assuming code_root_dir is correctly set and accessible
from src.models.OUTLIER_DETECTION.OPTICS_MODELS.within_cluster_clustering.get_subcluster_indices_for_cluster import get_subcluster_indices_for_cluster
from src.data.load_dtw_matrices_from_json import load_dtw_matrices_from_json
from src.models.OUTLIER_DETECTION.OPTICS_MODELS.within_cluster_clustering.get_subcluster_paths_from_subcluster_indices import get_subcluster_paths_from_subcluster_indices

def calculate_dtw_distance(path1, path2, metric='euclidean', itakura_slope=1.0):
    """
    Calculate the normalized DTW distance between two paths.
    """
    warping_path, dtw_score = dtw_path_from_metric(
        path1, path2, metric=metric
    )
    # Normalize by the length of the warping path
    warping_path_length = len(warping_path)
    normalized_dtw_score = dtw_score / warping_path_length
    return normalized_dtw_score

def extract_itakura_and_metric(key):
    """
    Extract the itakura slope and metric from a key string.
    """
    # Split the key on underscores
    parts = key.split('_')
    # Extract the number after "itakura" and the metric after "dist"
    itakura_value = parts[1]  # The second part after "itakura"
    distance_metric = parts[3]  # The fourth part after "dist"
    return float(itakura_value), distance_metric

def load_subcluster_optimized_params(intersection_name, intersection_cluster_init_id, intersection_eval_metric, in_cluster_init_id, in_cluster_eval_metric, cluster_key):
    try:
        with open(f'{code_root_dir}/data/processed/within_cluster_clustering_optimization/'
                  f'{intersection_name}_optics_optimized_vehicle_paths_{intersection_cluster_init_id}_'
                  f'{intersection_eval_metric}_within_cluster_{cluster_key}_optimized_params_{in_cluster_init_id}.json') as f:
            in_cluster_optics_optimized_params = json.load(f)
    except FileNotFoundError:
        # If no in-cluster clustering optimization results are found, return the intersection-level optimized params
        if intersection_cluster_init_id is None or intersection_cluster_init_id == '':
            with open(f'{code_root_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params.json') as f:
                in_cluster_optics_optimized_params = json.load(f)
        else:
            with open(f'{code_root_dir}/data/processed/{intersection_name}_optics_vehicle_paths_optimized_params_{intersection_cluster_init_id}.json') as f:
                in_cluster_optics_optimized_params = json.load(f)
    in_cluster_optics_optimized_params = in_cluster_optics_optimized_params[in_cluster_eval_metric]
    return in_cluster_optics_optimized_params

def evaluate_path_segment(input_segment, subcluster_paths, subcluster_indices, intersection_name, intersection_cluster_init_id, intersection_eval_metric, in_cluster_init_id, in_cluster_eval_metric):
    """
    Evaluate the input path segment against subcluster paths and compute DTW scores.
    """
    scores = []

    # Iterate over clusters
    for cluster_key, subclusters in subcluster_paths.items():
        # Iterate over subclusters
        for subcluster_key, paths in subclusters.items():
            # Exclude subclusters with label -1
            if int(subcluster_key) == -1:
                continue  # Skip subclusters with label -1

            # Get the DTW distance metric and itakura slope for the subcluster
            in_cluster_optics_optimized_params = load_subcluster_optimized_params(
                intersection_name, intersection_cluster_init_id, intersection_eval_metric,
                in_cluster_init_id, in_cluster_eval_metric, cluster_key
            )
            # Extract itakura slope and DTW distance metric from the DTW dist matrix key
            itakura_slope, dtw_dist_metric = extract_itakura_and_metric(in_cluster_optics_optimized_params['dtw_key'])

            # Iterate over the paths in the subcluster
            for path_idx, x_y_path_list in paths.items():
                if path_idx == 102: # temporary fix to evaluate path from dataset
                    continue
                x_path, y_path = x_y_path_list[0], x_y_path_list[1]
                sub_path_combined = np.array(list(zip(x_path, y_path)))

                score = calculate_dtw_distance(input_segment, sub_path_combined, metric=dtw_dist_metric, itakura_slope=itakura_slope)

                scores.append((cluster_key, subcluster_key, path_idx, score))

    scores.sort(key=lambda x: x[3])  # Sort by lowest score
    return scores

def vehicle_path_probability_handler(intersection_name: str, 
                                     id_intersection_optics_optimization_init_params: str, 
                                     id_in_cluster_optics_optimization_init_params: str,
                                     intersection_eval_metric: str,
                                     in_cluster_eval_metric: str,
                                     x_input: list[float],
                                     y_input: list[float]):
    """
    Compute an outlier probability score for a given path based on the average in-subcluster DTW score.
    If the best matching subcluster contains only one path, find the next best match with a subcluster that has more than one path.
    """
    # Load precomputed DTW matrices
    intersection_dtw_distance_matrix_dict = load_dtw_matrices_from_json(
        f'{code_root_dir}/data/processed/{intersection_name}_diff_itakura_slope_dtw_matrices.json')

    # Get subcluster indices
    subcluster_indices = get_subcluster_indices_for_cluster(
        intersection_name, 
        id_intersection_optics_optimization_init_params, 
        id_in_cluster_optics_optimization_init_params, 
        intersection_eval_metric, 
        in_cluster_eval_metric, 
        intersection_dtw_distance_matrix_dict)

    # Load the grouped DataFrame
    df_cuid_grouped = pd.read_csv(f'{code_root_dir}/data/processed/{intersection_name}_cuid_grouped.csv', index_col=0)

    # Convert 'x' and 'y' columns to lists of coordinates
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(ast.literal_eval)
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(ast.literal_eval)

    # Extract subcluster paths from the DataFrame
    subcluster_paths = get_subcluster_paths_from_subcluster_indices(subcluster_indices, df_cuid_grouped)

    # Create input path as an array of (x, y) coordinates
    input_path = np.array(list(zip(x_input, y_input)))

    # Evaluate the input path against the subcluster paths
    scores = evaluate_path_segment(
        input_path, 
        subcluster_paths, 
        subcluster_indices, 
        intersection_name, 
        id_intersection_optics_optimization_init_params, 
        intersection_eval_metric, 
        id_in_cluster_optics_optimization_init_params, 
        in_cluster_eval_metric)

    # Check if scores is empty (all subclusters might be -1)
    if not scores:
        print("No valid subclusters to evaluate. All subclusters are labeled as -1 (outliers).")
        outlier_probability = 1.0  # Consider it as an outlier
        result = {
            'outlier_probability_score': outlier_probability,
            'best_match': None,
            'input_path_best_score': None
        }
        return result

    # Initialize variables
    best_match = None
    dtw_input_to_best_match = None
    best_subcluster_paths = None

    # Iterate over sorted scores to find the best subcluster with more than one path
    for match in scores:
        cluster_id, subcluster_id, path_idx, dtw_score = match
        if int(subcluster_id) == -1:
            continue  # Skip outlier subclusters

        # Get all paths in the same subcluster as the current match
        current_subcluster_paths = subcluster_paths[cluster_id][subcluster_id]

        # Check if the subcluster has more than one path
        if len(current_subcluster_paths) > 1:
            best_match = match
            dtw_input_to_best_match = dtw_score
            best_subcluster_paths = current_subcluster_paths
            best_match_cluster_id = cluster_id
            best_match_subcluster_id = subcluster_id
            break  # Found a suitable subcluster
        else:
            continue  # Continue to the next best match

    if best_match is None:
        print("No suitable subcluster with more than one path found. Input path is considered an outlier.")
        outlier_probability = 1.0
        result = {
            'outlier_probability_score': outlier_probability,
            'best_match': scores[0],  # Return the best match even if subcluster has only one path
            'input_path_best_score': scores[0][3]
        }
        return result

    # Compute DTW distances between all pairs of paths within the best matching subcluster
    subcluster_path_indices = list(best_subcluster_paths.keys())
    in_subcluster_dtw_distances = []

    for i in range(len(subcluster_path_indices)):
        for j in range(i+1, len(subcluster_path_indices)):
            idx_i = subcluster_path_indices[i]
            idx_j = subcluster_path_indices[j]
            x_y_path_i = best_subcluster_paths[idx_i]
            x_y_path_j = best_subcluster_paths[idx_j]

            x_i, y_i = x_y_path_i[0], x_y_path_i[1]
            x_j, y_j = x_y_path_j[0], x_y_path_j[1]

            path_i = np.array(list(zip(x_i, y_i)))
            path_j = np.array(list(zip(x_j, y_j)))

            dtw_score = calculate_dtw_distance(path_i, path_j)
            in_subcluster_dtw_distances.append(dtw_score)

    # Continue with the rest of the code
    if not in_subcluster_dtw_distances:
        # This should not happen since we have ensured the subcluster has more than one path
        outlier_probability = 1.0
    else:
        # Apply any intersection-specific filtering if needed
        if intersection_name == 'k729_2022':
            percentile_50 = np.percentile(in_subcluster_dtw_distances, 50)
            filtered_dtw_distances = [d for d in in_subcluster_dtw_distances if d <= percentile_50]
        elif intersection_name == 'k733_2020':
            filtered_dtw_distances = in_subcluster_dtw_distances
        elif intersection_name == 'k733_2018':
            percentile_50 = np.percentile(in_subcluster_dtw_distances, 50)
            filtered_dtw_distances = [d for d in in_subcluster_dtw_distances if d <= percentile_50]
        else:
            filtered_dtw_distances = in_subcluster_dtw_distances  # Default case

        if not filtered_dtw_distances:
            filtered_dtw_distances = in_subcluster_dtw_distances

        # Calculate the average DTW distance within the subcluster using the filtered distances
        average_in_subcluster_dtw = np.mean(filtered_dtw_distances)
        std_in_subcluster_dtw = np.std(filtered_dtw_distances)

        # Compute the deviation
        deviation = dtw_input_to_best_match - average_in_subcluster_dtw

        print('***********************')
        print(f"Best Match: {best_match}")
        print(f'Current Subcluster: {best_match_cluster_id}_{best_match_subcluster_id}')
        print(f"Average DTW in subcluster: {average_in_subcluster_dtw}")
        print(f"Standard Deviation in subcluster: {std_in_subcluster_dtw}")
        print(f"Deviation: {deviation}")
        print('***********************')

        # Handle zero standard deviation
        if std_in_subcluster_dtw == 0:
            if deviation > 0:
                outlier_probability = 1.0
            else:
                outlier_probability = 0.0
        else:
            # Compute a z-score
            z_score = deviation / std_in_subcluster_dtw
            print('***********************')
            print(f"Z-Score: {z_score}")
            print('***********************')
            # Use a sigmoid function to map z-score to [0,1]
            outlier_probability = 1 / (1 + np.exp(-z_score))

    # Ensure the score is between 0 and 1
    outlier_probability = min(max(outlier_probability, 0), 1)

    print(f"Outlier Probability Score: {outlier_probability}")

    result = {
        'outlier_probability_score': outlier_probability,
        'best_match': best_match,
        'input_path_best_score': dtw_input_to_best_match
    }

    return result
