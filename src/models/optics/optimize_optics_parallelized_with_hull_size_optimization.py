from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from scipy.spatial import ConvexHull
from src.models.ensure_distance_metric_params import ensure_distance_metric_params

iterations = 0

def calculate_total_hull_size(X, labels):
    """
    Calculate the total size (area) of all convex hulls from the OPTICS labels.

    :param X: np.ndarray, the dataset
    :param labels: np.ndarray, the labels from OPTICS

    :return: float, the total area of all convex hulls
    """
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    total_hull_area = 0

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) >= 3:  # Convex hull needs at least 3 points
            hull = ConvexHull(cluster_points)
            total_hull_area += hull.area

    return total_hull_area

def evaluate_optics(X, epsilon, min_samples, metric, cluster_method, xi, metric_params, alpha=0.5, beta=0.5):
    '''
    Evaluate the OPTICS clustering algorithm with the given parameters.

    :param X: The dataset to cluster
    :param epsilon: The maximum distance between two samples for one to be considered as in the neighborhood of the other
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point
    :param metric: The distance metric to be used
    :param cluster_method: The clustering method to be used ('dbscan' or 'xi')
    :param xi: The minimum steepness on the reachability plot for a cluster boundary (used only with 'xi' method)
    :param metric_params: Additional parameters for the distance metric
    :param alpha: Weight for the silhouette, calinski-harabasz scores
    :param beta: Weight for minimizing the convex hull size

    :return: A dictionary containing evaluation results (silhouette, calinski_harabasz, davies_bouldin, and hull size)
    '''
    
    global iterations  # Access the global iterations variable
    if iterations % 1000 == 0:
        print(f'Iterations: {iterations}')  # Print current iteration count
    iterations += 1  # Increment iteration counter

    # Choose between xi and dbscan cluster methods
    optics = OPTICS(
        eps=epsilon,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,  # Pass metric_params if needed
        cluster_method=cluster_method,
        xi=xi
    ).fit(X)

    labels = optics.labels_
    if len(set(labels)) == 1 or len(set(labels)) == 2:
        print(f"None or only one cluster found, skipping this iteration. Number of clusters: {len(set(labels))}")
        return None

    # Calculate convex hull size
    hull_size = calculate_total_hull_size(X, labels)

    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)

    # Calculate combined scores
    combined_silhouette = alpha * silhouette - beta * hull_size
    combined_calinski_harabasz = alpha * calinski_harabasz - beta * hull_size
    combined_davies_bouldin = alpha * (1 / davies_bouldin) - beta * hull_size

    return {
        'epsilon': epsilon,
        'min_samples': min_samples,
        'metric': metric,
        'cluster_method': cluster_method,
        'xi': xi if cluster_method == 'xi' else None,  # Only return xi if the cluster_method is 'xi',
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski_harabasz,
        'davies_bouldin_score': davies_bouldin,
        'hull_size': hull_size,
        'combined_silhouette': combined_silhouette,
        'combined_calinski_harabasz': combined_calinski_harabasz,
        'combined_davies_bouldin': combined_davies_bouldin
    }

def optimize_optics_parallelized_with_hull_size_optimization(X,
                                 max_eps_range, 
                                 min_samples_range, 
                                 metrics=['euclidean'], 
                                 cluster_methods=['dbscan'], 
                                 xis=np.array([0.05]),
                                 alpha=0.5,
                                 beta=0.5,
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
    :param alpha: Weight for the silhouette and calinski-harabasz scores
    :param beta: Weight for minimizing the convex hull size
    :param n_jobs: The number of parallel jobs to run
    :param kwargs: Additional keyword arguments for the distance metrics

    :return: A dictionary containing the best results for silhouette, calinski_harabasz, davies_bouldin, and hull size
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
            X, epsilon, min_samples, metric, cluster_method, xi, metric_params_dict.get(metric), alpha, beta
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

    # Find the best scores using the combined score, silhouette, calinski_harabasz, and davies_bouldin
    best_combined_silhouette = max(results, key=lambda x: x['combined_silhouette'])
    best_combined_calinski_harabasz = max(results, key=lambda x: x['combined_calinski_harabasz'])
    best_combined_davies_bouldin = max(results, key=lambda x: x['combined_davies_bouldin'])

    # Return the best results
    return {
        'combined_silhouette': {
            'score': best_combined_silhouette['combined_silhouette'],
            'epsilon': best_combined_silhouette['epsilon'],
            'min_samples': best_combined_silhouette['min_samples'],
            'metric': best_combined_silhouette['metric'],
            'cluster_method': best_combined_silhouette['cluster_method'],
            'xi': best_combined_silhouette['xi']
        },
        'combined_calinski_harabasz': {
            'score': best_combined_calinski_harabasz['combined_calinski_harabasz'],
            'epsilon': best_combined_calinski_harabasz['epsilon'],
            'min_samples': best_combined_calinski_harabasz['min_samples'],
            'metric': best_combined_calinski_harabasz['metric'],
            'cluster_method': best_combined_calinski_harabasz['cluster_method'],
            'xi': best_combined_calinski_harabasz['xi']
        },
        'combined_davies_bouldin': {
            'score': best_combined_davies_bouldin['combined_davies_bouldin'],
            'epsilon': best_combined_davies_bouldin['epsilon'],
            'min_samples': best_combined_davies_bouldin['min_samples'],
            'metric': best_combined_davies_bouldin['metric'],
            'cluster_method': best_combined_davies_bouldin['cluster_method'],
            'xi': best_combined_davies_bouldin['xi']
        }
    }
