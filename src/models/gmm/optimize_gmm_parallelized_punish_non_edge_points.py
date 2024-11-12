from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

# Global iteration counter
iterations = 0

# Function to evaluate GMM for each combination of parameters
def evaluate_gmm_punished(X, n_clusters, covariance_type, n_init, init_param, max_iter, random_state, x_non_edge_range, y_non_edge_range, punishment_factor_input, punishment_type):
    global iterations  # Access the global iterations variable
    iterations += 1  # Increment iteration counter
    

    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, 
                          n_init=n_init, init_params=init_param, max_iter=max_iter, 
                          random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)

    if len(np.unique(labels)) == 1:
        print('Only one cluster, skipping...')
        return None  # Skip this iteration if only one cluster is found
    
    punishment_factor = 0
    # Punish non-edge points
    for i in range(len(labels)):
        if labels[i] != -1:
            # check if point is between x_non_edge_range and y_non_edge_range
            if x_non_edge_range[0] < X[i][0] < x_non_edge_range[1] and y_non_edge_range[0] < X[i][1] < y_non_edge_range[1]:
                punishment_factor = punishment_factor_input
                break

    if punishment_type == 'add':
        current_silhouette_score = silhouette_score(X, labels) - punishment_factor
        current_calinski_harabasz_score = calinski_harabasz_score(X, labels) - punishment_factor
        current_davies_bouldin_score = davies_bouldin_score(X, labels) + punishment_factor
        current_bic_score = gmm.bic(X) + punishment_factor
        current_aic_score = gmm.aic(X) + punishment_factor
    elif punishment_type == 'multiply':
        current_silhouette_score = silhouette_score(X, labels) / punishment_factor
        current_calinski_harabasz_score = calinski_harabasz_score(X, labels) / punishment_factor
        current_davies_bouldin_score = davies_bouldin_score(X, labels) * punishment_factor
        current_bic_score = gmm.bic(X) * punishment_factor
        current_aic_score = gmm.aic(X) * punishment_factor
    else:
        raise ValueError('Invalid punishment type. Choose between "add" or "multiply".')
    
    if iterations % 500 == 0:
        print(f'Iterations: {iterations}')  # Print current iteration count
        print(f'Current silhouette score: {current_silhouette_score}')
        print(f'Current calinski_harabasz_score: {current_calinski_harabasz_score}')
        print(f'Current davies_bouldin_score: {current_davies_bouldin_score}')
        print(f'Current bic_score: {current_bic_score}')
        print(f'Current aic_score: {current_aic_score}')

    return {
        'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 
        'init_param': init_param, 'max_iter': max_iter, 'random_state': random_state, 
        'silhouette_score': current_silhouette_score, 'calinski_harabasz_score': current_calinski_harabasz_score,
        'davies_bouldin_score': current_davies_bouldin_score, 'bic_score': current_bic_score, 'aic_score': current_aic_score
    }

# Main optimization function
def optimize_gmm_parallelized_punish_non_edge_points(X: np.array, n_clusters_range: np.array, covariance_type_range: np.array, 
                 n_init_range: np.array, init_params: np.array, max_iter_range: np.array, 
                 random_state_range: np.array, 
                 x_non_edge_range:range, y_non_edge_range:range, punishment_factor:float, punishment_type:str,
                 n_jobs=-1):
    '''
    Punishes non-edge points by increasing all scores.
    Punishment can be done by adding(subtracting) a factor or multiplying(dividing) by a factor.

    :param punisment_type: 'add' or 'multiply'
    '''
    print(f'Punishing non-edge points with {punishment_type} factor {punishment_factor}')
    print(f'Non-edge points range: x: {x_non_edge_range}, y: {y_non_edge_range}')

    global iterations  # Ensure global access to the iterations counter

    # Parallelize GMM evaluation over all parameter combinations
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_gmm_punished)(
        X, n_clusters, covariance_type, n_init, init_param, max_iter, random_state, x_non_edge_range, y_non_edge_range, punishment_factor, punishment_type
    ) 
    for n_clusters in n_clusters_range
    for covariance_type in covariance_type_range
    for n_init in n_init_range
    for max_iter in max_iter_range
    for random_state in random_state_range
    for init_param in init_params
    )
    
    # Filter out None results (in case some configurations were skipped)
    results = [r for r in results if r is not None]
    
    # Finding the best results based on various criteria
    best_silhouette = max(results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz = max(results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin = min(results, key=lambda x: x['davies_bouldin_score'])
    best_bic = min(results, key=lambda x: x['bic_score'])
    best_aic = min(results, key=lambda x: x['aic_score'])

    
    optimization_results = {
        'best_silhouette': best_silhouette,
        'best_calinski_harabasz': best_calinski_harabasz,
        'best_davies_bouldin': best_davies_bouldin,
        'best_bic': best_bic,
        'best_aic': best_aic
    }
    
    return optimization_results



