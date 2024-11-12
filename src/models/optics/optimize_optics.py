import numpy as np
from sklearn.cluster import OPTICS
from sklearn import metrics
from scipy.spatial.distance import cdist

# ********************************************************************
# ********************************************************************
# **THIS FILE SHOULD NOT BE USED BECAUSE OF WRONG SCORE CALCULATIONS**
# ********************************************************************
# ********************************************************************

def optimize_optics_1(X, max_eps_range:np.array, min_samples_range:np.array, metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    This function optimizes the OPTICS clustering algorithm by finding the best epsilon and min_samples values
    for the given dataset X. The optimization is done by maximizing the Silhouette score, Calinski_Harabasz Score
    and Davies-Boulin Score.

    First loop: epsilon
    Second loop: min_samples
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0
    for epsilon in max_eps_range:
        for min_samples in min_samples_range:
            print(f'Iteration {iterations}')
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            if(len(set(labels)) == 1):
                print('Only one cluster found, skipping...')
                continue
            silhouette_score = metrics.silhouette_score(X, labels)
            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples
        }
    }


def optimize_optics_2(X, max_eps_range:np.array, min_samples_range:np.array, metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    This function optimizes the OPTICS clustering algorithm by finding the best epsilon and min_samples values
    for the given dataset X. The optimization is done by maximizing the Silhouette score, Calinski_Harabasz Score
    and Davies-Boulin Score.

    First loop: min_samples
    Second loop: epsilon
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0
    for min_samples in min_samples_range:
        for epsilon in max_eps_range:
            print(f'Iteration {iterations}')
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            if(len(set(labels)) == 1):
                print('Only one cluster found, skipping...')
                continue
            silhouette_score = metrics.silhouette_score(X, labels)
            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels)
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels)
            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples
        }
    }


def optimize_optics_mediod_edge_adjusted_medioids(X, max_eps_range:np.array, min_samples_range:np.array, x_range:list[float], y_range:list[float], metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    Optimizes OPTICS clustering w/ Silhouette, Calinski-Harabasz and Davies-Bouldin scores, with a penalty factor for mediods inside the x and y ranges.
    Penalty factor is calculated using $score*penalty$ with the penalty factor being $penalty=1.5$ if the mediod is inside the x and y ranges, and $1$ (no penalty) otherwise.

    First loop: min_samples
    Second loop: epsilon
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param x_range: The range of x values for the mediods
    :param y_range: The range of y values for the mediods
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0
    for min_samples in min_samples_range:
        for epsilon in max_eps_range:
            print(f'Iteration {iterations}')
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            if(len(set(labels)) == 1):
                print('Only one cluster found, skipping...')
                continue

            medoids = calculate_medoids(X, labels)
            penalty_factor = 1
            for medoid in medoids:
                if (x_range[0] < medoid[0] < x_range[1] and y_range[0] < medoid[1] < y_range[1]):
                    penalty_factor = 1.5

            silhouette_score = metrics.silhouette_score(X, labels) / penalty_factor # higher is better
            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels) / penalty_factor # higher is better
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels) * penalty_factor # lower is better

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples
        }
    }


def optimize_optics_mediod_edge_adjusted_cluster_members(X, max_eps_range:np.array, min_samples_range:np.array, x_range:list[float], y_range:list[float], penalty=1.5, metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    Optimizes OPTICS clustering w/ Silhouette, Calinski-Harabasz and Davies-Bouldin scores, with a penalty factor for mediods inside the x and y ranges.
    Penalty factor is calculated using $score*penalty$ with the penalty factor being $penalty=1.5$ if the mediod is inside the x and y ranges, and $1$ (no penalty) otherwise.

    First loop: min_samples
    Second loop: epsilon
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param x_range: The range of x values for the mediods
    :param y_range: The range of y values for the mediods
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0
    for min_samples in min_samples_range:
        for epsilon in max_eps_range:
            print(f'Iteration {iterations}')
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            if(len(set(labels)) == 1):
                print('Only one cluster found, skipping...')
                continue

            # get a list of the cluster members
            cluster_members = []
            for label in np.unique(labels):
                cluster_members.append(X[labels == label])

            penalty_factor = 1
            for cluster in cluster_members:
                for point in cluster:
                    if (x_range[0] < point[0] < x_range[1] and y_range[0] < point[1] < y_range[1]):
                        penalty_factor = penalty

            silhouette_score = metrics.silhouette_score(X, labels) / penalty_factor # higher is better
            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels) / penalty_factor # higher is better
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels) * penalty_factor # lower is better

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples
        }
    }


def optimize_optics_mediod_edge_adjusted_cluster_members_accumulating(X, max_eps_range:np.array, min_samples_range:np.array, x_range:list[float], y_range:list[float], penalty=1.5, metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    Optimizes OPTICS clustering w/ Silhouette, Calinski-Harabasz and Davies-Bouldin scores, with a penalty factor for mediods inside the x and y ranges.
    Penalty factor is calculated using $score*penalty$ with the penalty factor being $penalty=1.5$ if the mediod is inside the x and y ranges, and $1$ (no penalty) otherwise.

    First loop: min_samples
    Second loop: epsilon
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param x_range: The range of x values for the mediods
    :param y_range: The range of y values for the mediods
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0

    for min_samples in min_samples_range:
        for epsilon in max_eps_range:
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            if(len(set(labels)) == 1):
                print('Only one cluster found, skipping...')
                continue

            # get a list of the cluster members
            cluster_members = []
            for label in np.unique(labels):
                cluster_members.append(X[labels == label])


            penalty_factor = 1

            for i, cluster in enumerate(cluster_members):
                # if iterations % 100 == 0:
                #         print(f'Iteration {iterations}')
                #         print(f'Cluster {np.unique(labels)[i]}')
                for point in cluster:
                    if (x_range[0] < point[0] < x_range[1] and y_range[0] < point[1] < y_range[1]):
                        penalty_factor += penalty

            silhouette_score = metrics.silhouette_score(X, labels) / penalty_factor
            calinski_harabasz_score = metrics.calinski_harabasz_score(X, labels) / penalty_factor
            davies_bouldin_score = metrics.davies_bouldin_score(X, labels) * penalty_factor # lower is better

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
                penalty_factor_silhouette = penalty_factor
                print(f'new best silhouette score: {best_silhouette_score} w penalty factor {penalty_factor_silhouette}')
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
                penalty_factor_calinski_harabasz = penalty_factor
                print(f'new best calinski harabasz score: {best_calinski_harabasz_score} w penalty factor {penalty_factor_calinski_harabasz}')
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples
                penalty_factor_davies_bouldin = penalty_factor
                print(f'new best davies bouldin score: {best_davies_bouldin_score} w penalty factor {penalty_factor_davies_bouldin}')



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples,
            'penalty_factor': penalty_factor_silhouette
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples,
            'penalty_factor': penalty_factor_calinski_harabasz
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples,
            'penalty_factor': penalty_factor_davies_bouldin
        }
    }


def optimize_optics_no_noise_regarded(X, max_eps_range:np.array, min_samples_range:np.array, metric='euclidean', cluster_method='dbscan', xi=0.05):
    '''
    This function optimizes the OPTICS clustering algorithm by finding the best epsilon and min_samples values
    for the given dataset X. The optimization is done by maximizing the Silhouette score, Calinski_Harabasz Score
    and Davies-Boulin Score.

    This does not count the noise points in the calculation of the scores.

    First loop: min_samples
    Second loop: epsilon
    
    :param X: The dataset to be clustered
    :param max_eps_range: The range of epsilon values to be tested
    :param min_samples_range: The range of min_samples values to be tested
    :param metric: The distance metric to be used, can be `euclidean`, `manhattan`, `cosine`, etc.
    :param cluster_method: The clustering method to be used, can be `dbscan` or `xi`
    :param xi: The xi value to be used if cluster_method is `xi`

    :return: A dictionary containing the best epsilon and min_samples values for silhouette, calinski_harabasz and davies_bouldin scores
    '''


    best_silhouette_score = 0
    best_calinski_harabasz_score = 0
    best_davies_bouldin_score = 0

    iterations = 0
    for min_samples in min_samples_range:
        for epsilon in max_eps_range:
            print(f'Iteration {iterations}')
            iterations += 1
            if(cluster_method == 'xi'):
                optics = OPTICS(eps=epsilon,
                                    min_samples=min_samples,
                                    metric=metric,
                                    cluster_method=cluster_method,
                                    xi=xi
                                    ).fit(X)
            else:
                optics = OPTICS(eps=epsilon,
                                min_samples=min_samples,
                                metric=metric,
                                cluster_method=cluster_method,
                                ).fit(X)
                
            labels = optics.labels_
            labels_no_noise = labels[labels != -1]
            X_no_noise = X[labels != -1]
            if(len(set(labels_no_noise)) == 0):
                print('Only noise found, skipping...')
                continue
            if (len(set(labels_no_noise)) == 1):
                print('Only one cluster found, skipping...')
                continue
            silhouette_score = metrics.silhouette_score(X_no_noise, labels_no_noise)
            calinski_harabasz_score = metrics.calinski_harabasz_score(X_no_noise, labels_no_noise)
            davies_bouldin_score = metrics.davies_bouldin_score(X_no_noise, labels_no_noise)
            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_silhouette_epsilon = epsilon
                best_silhouette_min_samples = min_samples
            if calinski_harabasz_score > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz_score
                best_calinski_harabasz_epsilon = epsilon
                best_calinski_harabasz_min_samples = min_samples
            if davies_bouldin_score > best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin_score
                best_davies_bouldin_epsilon = epsilon
                best_davies_bouldin_min_samples = min_samples



    return {
        'silhouette': {
            'score': best_silhouette_score,
            'epsilon': best_silhouette_epsilon,
            'min_samples': best_silhouette_min_samples
        },
        'calinski_harabasz': {
            'score': best_calinski_harabasz_score,
            'epsilon': best_calinski_harabasz_epsilon,
            'min_samples': best_calinski_harabasz_min_samples
        },
        'davies_bouldin': {
            'score': best_davies_bouldin_score,
            'epsilon': best_davies_bouldin_epsilon,
            'min_samples': best_davies_bouldin_min_samples
        }
    }


def calculate_medoids(X, labels, metric='euclidean'):
    """Calculate the medoid for each cluster given data points and cluster labels.
    
    :param X: 2D array of data points
    :param labels: List or array of cluster labels (length: n_samples)
    :return: List of medoids, one for each cluster, sorted by the clusters
    """
    medoids = []
    unique_labels = np.unique(labels)  # Find the unique cluster labels
    
    for label in unique_labels:
        # Get the data points for the current cluster
        cluster_points = X[labels == label]
        
        # Compute pairwise distances within the cluster
        distances = cdist(cluster_points, cluster_points, metric=metric)
        
        # Find the index of the medoid (point with minimum sum of distances to other points)
        medoid_index = np.argmin(np.sum(distances, axis=1))
        
        # Add the medoid to the list
        medoids.append(cluster_points[medoid_index])
    
    return np.array(medoids)