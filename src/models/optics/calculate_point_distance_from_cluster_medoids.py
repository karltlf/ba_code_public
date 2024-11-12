import numpy as np

def calculate_point_distance_from_cluster_medoids(point: list, cluster_medoids: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance from a given point to all cluster medoids.

    Args:
    point (list): A list representing the coordinates of the point (e.g., [x, y]).
    cluster_medoids (np.ndarray): A 2D numpy array where each row is the coordinates of a medoid (e.g., [[x1, y1], [x2, y2], ...]).

    Returns:
    np.ndarray: An array of distances from the point to each cluster medoid.
    """
    point_array = np.array(point)
    distances = np.linalg.norm(cluster_medoids - point_array, axis=1)
    return distances
