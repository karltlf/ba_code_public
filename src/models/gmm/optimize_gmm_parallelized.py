from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

# Global iteration counter
iterations = 0

# Function to evaluate GMM for each combination of parameters
def evaluate_gmm(X, n_clusters, covariance_type, n_init, init_param, max_iter, random_state):
    global iterations  # Access the global iterations variable
    iterations += 1  # Increment iteration counter
    print(f'Iterations: {iterations}')  # Print current iteration count

    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, 
                          n_init=n_init, init_params=init_param, max_iter=max_iter, 
                          random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)

    if len(np.unique(labels)) == 1:
        print('Only one cluster, skipping...')
        return None  # Skip this iteration if only one cluster is found
    
    current_silhouette_score = silhouette_score(X, labels)
    current_calinski_harabasz_score = calinski_harabasz_score(X, labels)
    current_davies_bouldin_score = davies_bouldin_score(X, labels)
    current_bic_score = gmm.bic(X)
    current_aic_score = gmm.aic(X)

    return {
        'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 
        'init_param': init_param, 'max_iter': max_iter, 'random_state': random_state, 
        'silhouette_score': current_silhouette_score, 'calinski_harabasz_score': current_calinski_harabasz_score,
        'davies_bouldin_score': current_davies_bouldin_score, 'bic_score': current_bic_score, 'aic_score': current_aic_score
    }

# Main optimization function
def optimize_gmm_parallelized(X: np.array, n_clusters_range: np.array, covariance_type_range: np.array, 
                 n_init_range: np.array, init_params: np.array, max_iter_range: np.array, 
                 random_state_range: np.array, n_jobs=-1):
    
    global iterations  # Ensure global access to the iterations counter

    # Parallelize GMM evaluation over all parameter combinations
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_gmm)(
        X, n_clusters, covariance_type, n_init, init_param, max_iter, random_state
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



