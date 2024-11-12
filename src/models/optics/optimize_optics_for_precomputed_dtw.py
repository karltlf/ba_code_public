from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import warnings

iterations = 0

def evaluate_optics(dtw_key, dtw_matrix, epsilon, min_samples, cluster_method, xi):
    '''
    Evaluate the OPTICS clustering algorithm with the given parameters using the precomputed DTW distance matrix.

    :param dtw_matrix: Precomputed DTW distance matrix (2D numpy array)
    :param epsilon: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :param cluster_method: The clustering method to be used ('dbscan' or 'xi')
    :param xi: The minimum steepness on the reachability plot for a cluster boundary (used only with 'xi' method)

    :return: A dictionary containing evaluation results (silhouette, calinski_harabasz, davies_bouldin)
    '''

    warnings.filterwarnings('ignore')  
    
    # Access the global iterations variable
    global iterations
    if iterations % 1 == 0:
        print(f'Iterations: {iterations}')  # Print current iteration count
    iterations += 1  # Increment iteration counter

    # Handle nan and inf values in the precomputed DTW matrix
    dtw_matrix = np.nan_to_num(dtw_matrix, nan=1e10, posinf=1e10)

    # Fit OPTICS model
    optics = OPTICS(
        max_eps=epsilon,
        min_samples=min_samples,
        metric='precomputed',
        cluster_method=cluster_method,
        xi=xi
    ).fit(dtw_matrix)

    # Get clustering labels
    labels = optics.labels_

    if len(set(labels)) == 1 or len(set(labels)) == 2:
        # Only one cluster + (outliers) found, skip this iteration
        return None
    
    # --------- UNCOMMENT THIS SECTION TO FILTER OUT OUTLIERS ---------
    # --------- DID NOT WORK WELL, SEE RESULTS IN nb/expl/plot_optimized_no_outliers_dtw_optics.ipynb ---------
    # # Check if there are outliers in the labels
    # if -1 in labels:
    #     # Get a mask for non-outliers (i.e., labels that are not -1)
    #     mask = labels != -1
        
    #     # Filter the labels and the corresponding rows and columns of the distance matrix
    #     labels = labels[mask]
    #     dtw_matrix = dtw_matrix[mask][:, mask]  # Filter both rows and columns

    # Calculate clustering evaluation scores
    current_silhouette_score = silhouette_score(dtw_matrix, labels, metric='precomputed')
    current_calinski_harabasz_score = calinski_harabasz_score(dtw_matrix, labels)
    current_davies_bouldin_score = davies_bouldin_score(dtw_matrix, labels)

    return {
        'dtw_key': dtw_key,
        'epsilon': epsilon,
        'min_samples': min_samples,
        'cluster_method': cluster_method,
        'xi': xi if cluster_method == 'xi' else None,
        'silhouette_score': current_silhouette_score,
        'calinski_harabasz_score': current_calinski_harabasz_score,
        'davies_bouldin_score': current_davies_bouldin_score
    }


def optimize_optics_for_precomputed_dtw(dtw_matrices_dict, 
                                 max_eps_range, 
                                 min_samples_range, 
                                 cluster_methods=['dbscan'], 
                                 xis=np.array([0.05]),
                                 n_jobs=-1):
    '''
    Optimize the OPTICS clustering algorithm using precomputed DTW distance matrices.

    :param dtw_matrices_dict: A dictionary of precomputed DTW distance matrices (keyed by parameters)
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param cluster_methods: The cluster methods to be tested ('dbscan' or 'xi')
    :param xis: The xi values to be tested (used only for the 'xi' method)
    :param n_jobs: The number of parallel jobs to run

    :return: A dictionary containing the best results for silhouette, calinski_harabasz, and davies_bouldin scores
    '''

    # Convert to np.array if necessary
    max_eps_range = np.array(max_eps_range) if not isinstance(max_eps_range, np.ndarray) else max_eps_range
    min_samples_range = np.array(min_samples_range) if not isinstance(min_samples_range, np.ndarray) else min_samples_range
    cluster_methods = list(cluster_methods) if not isinstance(cluster_methods, list) else cluster_methods
    xis = np.array(xis) if not isinstance(xis, np.ndarray) else xis

    # Parallel execution for different combinations of epsilon, min_samples, and cluster methods
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_optics)(dtw_key, dtw_matrix, epsilon, min_samples, cluster_method, xi)
        for dtw_key, dtw_matrix in dtw_matrices_dict.items()
        for epsilon in max_eps_range
        for min_samples in min_samples_range
        for cluster_method in cluster_methods
        for xi in xis
    )

    # Filter out None results (where only one cluster was found)
    results = [r for r in results if r is not None]

    if not results:
        raise ValueError("No valid clusters were found.")

    # Find the best scores using max/min of the evaluation metrics
    best_silhouette_result = max(results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz_result = max(results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin_result = min(results, key=lambda x: x['davies_bouldin_score'])

    # Return the best results with additional metadata (return xi only if cluster_method is 'xi')
    return {
        'silhouette': {
            'score': best_silhouette_result['silhouette_score'],
            'dtw_key': best_silhouette_result['dtw_key'],
            'epsilon': best_silhouette_result['epsilon'],
            'min_samples': best_silhouette_result['min_samples'],
            'cluster_method': best_silhouette_result['cluster_method'],
            'xi': best_silhouette_result['xi'] if best_silhouette_result['cluster_method'] == 'xi' else None
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_result['calinski_harabasz_score'],
            'dtw_key': best_calinski_harabasz_result['dtw_key'],
            'epsilon': best_calinski_harabasz_result['epsilon'],
            'min_samples': best_calinski_harabasz_result['min_samples'],
            'cluster_method': best_calinski_harabasz_result['cluster_method'],
            'xi': best_calinski_harabasz_result['xi'] if best_calinski_harabasz_result['cluster_method'] == 'xi' else None
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_result['davies_bouldin_score'],
            'dtw_key': best_davies_bouldin_result['dtw_key'],
            'epsilon': best_davies_bouldin_result['epsilon'],
            'min_samples': best_davies_bouldin_result['min_samples'],
            'cluster_method': best_davies_bouldin_result['cluster_method'],
            'xi': best_davies_bouldin_result['xi'] if best_davies_bouldin_result['cluster_method'] == 'xi' else None
        }
    }

