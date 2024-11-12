import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def optimize_gmm(X:np.array, n_clusters_range:np.array, covariance_type_range:np.array, n_init_range:np.array, init_params:np.array, max_iter_range:np.array, random_state_range:np.array):
    
    best_silhouette_score = -np.inf
    best_calinski_harabasz_score = -np.inf
    best_davies_bouldin_score = np.inf
    best_bic_score = np.inf
    best_aic_score = np.inf

    iterations = 0

    for n_clusters in n_clusters_range:
        for covariance_type in covariance_type_range:
            for n_init in n_init_range:
                for max_iter in max_iter_range:
                    for random_state in random_state_range:
                        for init_param in init_params:
                            iterations += 1
                            print(f'Iterations: {iterations}')

                            gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, init_params=init_param, max_iter=max_iter, random_state=random_state)
                            gmm.fit(X)
                            labels = gmm.predict(X)

                            if len(np.unique(labels)) == 1:
                                print('Only one cluster, skipping...')
                                continue
                                
                            current_silhouette_score = silhouette_score(X, labels)
                            current_calinski_harabasz_score = calinski_harabasz_score(X, labels)
                            current_davies_bouldin_score = davies_bouldin_score(X, labels)
                            current_bic_score = gmm.bic(X)
                            current_aic_score = gmm.aic(X)

                            if current_silhouette_score > best_silhouette_score:
                                best_silhouette_score = current_silhouette_score
                                best_silhouette_results = {'score': best_silhouette_score, 'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if current_calinski_harabasz_score > best_calinski_harabasz_score:
                                best_calinski_harabasz_score = current_calinski_harabasz_score
                                best_calinski_harabasz_results = {'score': best_calinski_harabasz_score, 'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if current_davies_bouldin_score < best_davies_bouldin_score:
                                best_davies_bouldin_score = current_davies_bouldin_score
                                best_davies_bouldin_results = {'score': best_davies_bouldin_score, 'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if current_bic_score < best_bic_score:
                                best_bic_score = current_bic_score
                                best_bic_results = {'score': best_bic_score, 'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if current_aic_score < best_aic_score:
                                best_aic_score = current_aic_score
                                best_aic_results = {'score': best_aic_score, 'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
        
    optimization_results = {'silhouette': best_silhouette_results, 'calinski_harabasz': best_calinski_harabasz_results, 'davies_bouldin': best_davies_bouldin_results, 'bic': best_bic_results, 'aic': best_aic_results}
    return optimization_results

def optimize_gmm_no_noise(X:np.array, n_clusters_range:np.array, covariance_type_range:np.array, n_init_range:np.array, init_params:np.array, max_iter_range:np.array, random_state_range:np.array):
    
    best_silhouette_score = -np.inf
    best_calinski_harabasz_score = -np.inf
    best_davies_bouldin_score = np.inf
    best_bic_score = np.inf
    best_aic_score = np.inf
    best_params = None

    iterations = 0
    
    for n_clusters in n_clusters_range:
        for covariance_type in covariance_type_range:
            for n_init in n_init_range:
                for max_iter in max_iter_range:
                    for random_state in random_state_range:
                        for init_param in init_params:


                            iterations += 1
                            print(f'Iterations: {iterations}')

                            gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, n_init=n_init, init_params=init_param, max_iter=max_iter, random_state=random_state)
                            gmm.fit(X)
                            labels = gmm.predict(X)

                            labels_no_noise = labels[labels != -1]
                            X_no_noise = X[labels != -1]

                            if len(np.unique(labels_no_noise)) == 1:
                                print('Only one cluster, skipping...')
                            if len(np.unique(labels_no_noise)) == 0:
                                print('Only noise, skipping...')
                                continue

                            silhouette_score = silhouette_score(X_no_noise, labels_no_noise)
                            calinski_harabasz_score = calinski_harabasz_score(X_no_noise, labels_no_noise)
                            davies_bouldin_score = davies_bouldin_score(X_no_noise, labels_no_noise)
                            bic_score = gmm.bic(X_no_noise)
                            aic_score = gmm.aic(X_no_noise)

                            if silhouette_score > best_silhouette_score:
                                best_silhouette_score = silhouette_score
                                best_silhouette_params = {'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if calinski_harabasz_score > best_calinski_harabasz_score:
                                best_calinski_harabasz_score = calinski_harabasz_score
                                best_calinski_harabasz_params = {'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if davies_bouldin_score < best_davies_bouldin_score:
                                best_davies_bouldin_score = davies_bouldin_score
                                best_davies_bouldin_params = {'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if bic_score < best_bic_score:
                                best_bic_score = bic_score
                                best_bic_params = {'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
                            if aic_score < best_aic_score:
                                best_aic_score = aic_score
                                best_aic_params = {'n_clusters': n_clusters, 'covariance_type': covariance_type, 'n_init': n_init, 'max_iter': max_iter}
    
    best_params = {'silhouette': best_silhouette_params, 'calinski_harabasz': best_calinski_harabasz_params, 'davies_bouldin': best_davies_bouldin_params, 'bic': best_bic_params, 'aic': best_aic_params}
    return best_params
