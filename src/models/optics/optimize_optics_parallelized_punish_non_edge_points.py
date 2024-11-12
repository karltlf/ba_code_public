from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from src.models.optics.calculate_aic_bic_for_optics import calculate_aic_bic_for_optics

# Helper function to evaluate OPTICS for each combination of epsilon and min_samples
def evaluate_optics(X, epsilon, min_samples, x_range, y_range, punishment_factor_input, punishment_type, metric, cluster_method, xi):
    if cluster_method == 'xi':
        optics = OPTICS(eps=epsilon,
                        min_samples=min_samples,
                        metric=metric,
                        cluster_method=cluster_method,
                        xi=xi).fit(X)
    else:
        optics = OPTICS(eps=epsilon,
                        min_samples=min_samples,
                        metric=metric,
                        cluster_method=cluster_method).fit(X)

    labels = optics.labels_
    if len(set(labels)) == 1:
        # Only one cluster found, skip this iteration
        return None

    # Calculate cluster centers
    unique_labels = np.unique(labels[labels != -1])
    cluster_centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

    # Punish if non-edge points are in one cluster
    punishment_factor = 0
    for i in range(len(labels)):
        if labels[i] != -1:
            if x_range[0] < X[i][0] < x_range[1] and y_range[0] < X[i][1] < y_range[1]:
                punishment_factor = punishment_factor_input
                break

    # Calculate scores and apply the punishment
    current_silhouette_score = silhouette_score(X, labels)
    current_calinski_harabasz_score = calinski_harabasz_score(X, labels)
    current_davies_bouldin_score = davies_bouldin_score(X, labels)
    current_aic_score, current_bic_score = calculate_aic_bic_for_optics(X, labels, cluster_centers)

    if punishment_type == 'add':
        current_silhouette_score -= punishment_factor
        current_calinski_harabasz_score -= punishment_factor
        current_davies_bouldin_score += punishment_factor
        current_aic_score += punishment_factor
        current_bic_score += punishment_factor
    elif punishment_type == 'multiply':
        current_silhouette_score /= punishment_factor
        current_calinski_harabasz_score /= punishment_factor
        current_davies_bouldin_score *= punishment_factor
        current_aic_score *= punishment_factor
        current_bic_score *= punishment_factor
    else:
        raise ValueError('Invalid punishment type. Choose between "add" or "multiply".')

    return {
        'epsilon': epsilon,
        'min_samples': min_samples,
        'silhouette_score': current_silhouette_score,
        'calinski_harabasz_score': current_calinski_harabasz_score,
        'davies_bouldin_score': current_davies_bouldin_score,
        'aic_score': current_aic_score,
        'bic_score': current_bic_score
    }

# Main function with parallelization
def optimize_optics_parallelized_punish_non_edge_points(X, max_eps_range: np.array, 
                                                        min_samples_range: np.array, 
                                                        x_range: list[float], 
                                                        y_range: list[float], 
                                                        punishment_factor_input: float, 
                                                        punishment_type: str, 
                                                        metric='euclidean', 
                                                        cluster_method='dbscan', 
                                                        xi=0.05, n_jobs=-1):

    # Parallel execution for different combinations of epsilon and min_samples
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_optics)(
            X, epsilon, min_samples, x_range, y_range, punishment_factor_input, punishment_type, metric, cluster_method, xi
        )
        for epsilon in max_eps_range
        for min_samples in min_samples_range
    )

    # Filter out None results (where only one cluster was found)
    results = [r for r in results if r is not None]

    if not results:
        raise ValueError("No valid clusters were found.")

    # Find the best scores using min and max
    best_silhouette_result = max(results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz_result = max(results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin_result = min(results, key=lambda x: x['davies_bouldin_score'])
    best_aic_result = min(results, key=lambda x: x['aic_score'])
    best_bic_result = min(results, key=lambda x: x['bic_score'])

    # Return the best results
    return {
        'silhouette': {
            'score': best_silhouette_result['silhouette_score'],
            'epsilon': best_silhouette_result['epsilon'],
            'min_samples': best_silhouette_result['min_samples']
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_result['calinski_harabasz_score'],
            'epsilon': best_calinski_harabasz_result['epsilon'],
            'min_samples': best_calinski_harabasz_result['min_samples']
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_result['davies_bouldin_score'],
            'epsilon': best_davies_bouldin_result['epsilon'],
            'min_samples': best_davies_bouldin_result['min_samples']
        },
        'aic': {
            'score': best_aic_result['aic_score'],
            'epsilon': best_aic_result['epsilon'],
            'min_samples': best_aic_result['min_samples']
        },
        'bic': {
            'score': best_bic_result['bic_score'],
            'epsilon': best_bic_result['epsilon'],
            'min_samples': best_bic_result['min_samples']
        }
    }
