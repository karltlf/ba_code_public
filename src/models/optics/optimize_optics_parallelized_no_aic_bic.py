from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from src.models.optics.calculate_aic_bic_for_optics import calculate_aic_bic_for_optics
from src.models.ensure_distance_metric_params import ensure_distance_metric_params

iterations = 0

def evaluate_optics(X, epsilon, min_samples, metric, cluster_method, xi, metric_params):
    '''
    Evaluate the OPTICS clustering algorithm with the given parameters.

    :param X: The dataset to cluster
    :param epsilon: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :param metric: The distance metric to be used
    :param cluster_method: The clustering method to be used ('dbscan' or 'xi')
    :param xi: The minimum steepness on the reachability plot for a cluster boundary (used only with 'xi' method)
    :param metric_params: Additional parameters for the distance metric

    :return: A dictionary containing evaluation results (silhouette, calinski_harabasz, davies_bouldin, aic, bic, and other parameters)
    '''
    
    global iterations  # Access the global iterations variable
    if iterations % 100 == 0:
        print(f'Iterations: {iterations}')  # Print current iteration count
    iterations += 1  # Increment iteration counter

    # Choose between xi and dbscan cluster methods
    optics = OPTICS(
        max_eps=epsilon,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,  # Pass metric_params if needed
        cluster_method=cluster_method,
        xi=xi
    ).fit(X)

    labels = optics.labels_
    if len(set(labels)) == 1:
        # Only one cluster found, skip this iteration
        return None

    # Calculate scores
    if metric_params is None:
        current_silhouette_score = silhouette_score(X, labels, metric=metric)
    else:
        current_silhouette_score = silhouette_score(X, labels, metric=metric, **metric_params)
    current_calinski_harabasz_score = calinski_harabasz_score(X, labels)
    current_davies_bouldin_score = davies_bouldin_score(X, labels)

    return {
        'epsilon': epsilon,
        'min_samples': min_samples,
        'metric': metric,
        'cluster_method': cluster_method,
        'xi': xi if cluster_method == 'xi' else None,  # Only return xi if the cluster_method is 'xi'
        'silhouette_score': current_silhouette_score,
        'calinski_harabasz_score': current_calinski_harabasz_score,
        'davies_bouldin_score': current_davies_bouldin_score
    }


def optimize_optics_parallelized_no_aic_bic(X,
                                 max_eps_range, 
                                 min_samples_range, 
                                 metrics=['euclidean'], 
                                 cluster_methods=['dbscan'], 
                                 xis=np.array([0.05]),
                                 n_jobs=-1,
                                 **kwargs):
    '''
    Optimize the OPTICS clustering algorithm by finding the best `epsilon`, `min_samples`, `cluster_method`, `xi`, and `metrics` parameters.

    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param metrics: The distance metrics to be tested
    :param cluster_methods: The cluster methods to be tested ('dbscan' or 'xi')
    :param xis: The xi values to be tested (used only for the 'xi' method)
    :param n_jobs: The number of parallel jobs to run
    :param kwargs: Additional keyword arguments for the distance metrics

    :return: A dictionary containing the best results for silhouette, calinski_harabasz, davies_bouldin, aic, and bic scores
    '''

    # Generate metric_params for each metric
    metric_params_dict = ensure_distance_metric_params(X, metrics, **kwargs)

    # Convert to np.array if necessary
    max_eps_range = np.array(max_eps_range) if not isinstance(max_eps_range, np.ndarray) else max_eps_range
    min_samples_range = np.array(min_samples_range) if not isinstance(min_samples_range, np.ndarray) else min_samples_range
    metrics = list(metrics) if not isinstance(metrics, list) else metrics
    cluster_methods = list(cluster_methods) if not isinstance(cluster_methods, list) else cluster_methods
    xis = np.array(xis) if not isinstance(xis, np.ndarray) else xis

    # Parallel execution for different combinations of epsilon, min_samples, and metrics
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_optics)(
            X, epsilon, min_samples, metric, cluster_method, xi, metric_params_dict.get(metric)
        )
        for epsilon in max_eps_range
        for min_samples in min_samples_range
        for metric in metrics
        for cluster_method in cluster_methods
        for xi in xis
    )

    # Filter out None results (where only one cluster was found)
    results = [r for r in results if r is not None]

    if not results:
        raise ValueError("No valid clusters were found.")

    # Find the best scores using min and max
    best_silhouette_result = max(results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz_result = max(results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin_result = min(results, key=lambda x: x['davies_bouldin_score'])


    # Return the best results with additional metadata (return xi only if cluster_method is 'xi')
    return {
        'silhouette': {
            'score': best_silhouette_result['silhouette_score'],
            'epsilon': best_silhouette_result['epsilon'],
            'min_samples': best_silhouette_result['min_samples'],
            'metric': best_silhouette_result['metric'],
            'cluster_method': best_silhouette_result['cluster_method'],
            'xi': best_silhouette_result['xi'] if best_silhouette_result['cluster_method'] == 'xi' else None
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_result['calinski_harabasz_score'],
            'epsilon': best_calinski_harabasz_result['epsilon'],
            'min_samples': best_calinski_harabasz_result['min_samples'],
            'metric': best_calinski_harabasz_result['metric'],
            'cluster_method': best_calinski_harabasz_result['cluster_method'],
            'xi': best_calinski_harabasz_result['xi'] if best_calinski_harabasz_result['cluster_method'] == 'xi' else None
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_result['davies_bouldin_score'],
            'epsilon': best_davies_bouldin_result['epsilon'],
            'min_samples': best_davies_bouldin_result['min_samples'],
            'metric': best_davies_bouldin_result['metric'],
            'cluster_method': best_davies_bouldin_result['cluster_method'],
            'xi': best_davies_bouldin_result['xi'] if best_davies_bouldin_result['cluster_method'] == 'xi' else None
        }
    }

