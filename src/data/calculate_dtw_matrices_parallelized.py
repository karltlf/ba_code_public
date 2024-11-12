from joblib import Parallel, delayed
import numpy as np
from tslearn.metrics import dtw_path_from_metric
import warnings
import os
import json

iterations = 0


def calculate_dtw_matrices_parallelized(data,
                                 dtw_metrics=None,
                                 itakura_max_slope_range=None,
                                 n_jobs=-1,
                                 results_file_path=None,
                                 **kwargs):
    
    # check if all necessary parameters are provided
    if results_file_path is None:
        raise ValueError("No results_file_path provided.")
    if dtw_metrics is None:
        raise ValueError("No dtw_metrics provided.")
    if itakura_max_slope_range is None:
        raise ValueError("No itakura_max_slope_range provided.")


    # Initialize dictionary to store all DTW matrices
    dtw_matrices_dict = {}


    # # ensure distance metric params !NOT IMPLEMENTED AS OF NOW!
    # metric_params = ensure_distance_metric_params(data, dtw_metrics)



    # Parallel execution for different combinations of epsilon, min_samples, and metrics
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_dtw_distance_matrix)(data, 
                                 dtw_metric, 
                                 itakura_max_slope, 
                                #  metric_params = metric_params[dtw_metric]
        )
        for dtw_metric in dtw_metrics
        for itakura_max_slope in itakura_max_slope_range
    )

    for key, matrix in results:
        dtw_matrices_dict[key] = matrix

    # Save all DTW matrices to a single JSON file
    save_all_dtw_matrices_to_json(dtw_matrices_dict, results_file_path)

    return results, dtw_matrices_dict



# def evaluate_dtw_distance_matrix(data, dtw_metric, itakura_max_slope, metric_params):
def evaluate_dtw_distance_matrix(data, dtw_metric, itakura_max_slope):

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Access the global iterations variable
    global iterations  # Access the global iterations variable
    if iterations % 10 == 0:
        print(f'Iterations: {iterations}')  # Print current iteration count
    iterations += 1  # Increment iteration counter


    # Calculate DTW distance matrix if not precomputed
    n = len(data)
    X = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1,n):
            # if metric_params is not None:
            #     _, distance  = dtw_path_from_metric(data[i], data[j], metric=dtw_metric, itakura_max_slope=itakura_max_slope, **metric_params)
            # else:
            _, distance  = dtw_path_from_metric(data[i], data[j], metric=dtw_metric, itakura_max_slope=itakura_max_slope)
            X[i][j] = distance
            X[j][i] = distance


    # handle nan and inf values
    X = np.nan_to_num(X, nan=1e10, posinf=1e10)


    # save the matrix as a JSON file
    key = f"itakura_{itakura_max_slope:.2f}_dist_{dtw_metric}"

    return key, X.tolist()



def save_all_dtw_matrices_to_json(dtw_matrices_dict, file_name):
    """
    Save all DTW matrices to a single JSON file.
    
    :param dtw_matrices_dict: Dictionary containing DTW matrices with keys based on Itakura slope and distance metric.
    :param file_name: The output file where matrices will be saved.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    # Save the entire dictionary as JSON
    with open(file_name, 'w') as json_file:
        json.dump(dtw_matrices_dict, json_file)
    
    print(f"All DTW matrices saved to {file_name}")