import numpy as np
from scipy.spatial.distance import cdist

def calculate_cluster_medoids(X, optics_model, return_type="ndarray"):
    """
    Calculates the medoids of clusters from an OPTICS model.

    :param X: np.ndarray, the dataset
    :param optics_model: OPTICS, the fitted OPTICS model
    :param return_type: str, either "dict" to return a dictionary or "ndarray" to return an ndarray of medoids
    
    :return: dict or np.ndarray, medoids of the clusters
    """
    
    labels = optics_model.labels_
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    medoids = {}

    for label in unique_labels:
        cluster_points = X[labels == label]
        distances = cdist(cluster_points, cluster_points, 'euclidean')
        medoid_index = np.argmin(np.sum(distances, axis=1))  # Find the point with the smallest sum of distances
        medoids[label] = cluster_points[medoid_index]

    if return_type == "ndarray":
        # Convert the dict to ndarray
        return np.array(list(medoids.values()))
    else:
        return medoids