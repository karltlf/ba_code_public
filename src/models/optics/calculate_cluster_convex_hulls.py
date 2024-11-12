import numpy as np
from scipy.spatial import ConvexHull

def calculate_cluster_convex_hulls(X, optics_model, return_type="ndarray"):
    """
    Calculates the convex hulls of clusters from an OPTICS model.

    :param X: np.ndarray, the dataset
    :param optics_model: OPTICS, the fitted OPTICS model
    :param return_type: str, either "ndarray" to return a list of convex hulls as arrays 
                        or "dict" to return as a dictionary
    
    :return: list of np.ndarray or dict, convex hulls of the clusters
    """
    
    labels = optics_model.labels_
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    hulls = {}

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) >= 3:  # Convex hull needs at least 3 points
            hull = ConvexHull(cluster_points)
            hulls[label] = cluster_points[hull.vertices]
    
    if return_type == "ndarray":
        # Return a list of arrays (one array per cluster)
        return [hull for hull in hulls.values()]
    else:
        return hulls
