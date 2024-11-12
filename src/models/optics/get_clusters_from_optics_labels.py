from sklearn.cluster import OPTICS



def get_clusters_from_optics_labels(optics_clustering_labels)->dict:
    '''
    Returns a dictionary of format cluster_label:indices_in_cluster

    :param optics_clustering: OPTICS object which was trained
    :return cluster: A dictionary with the cluster label as key and the indices as `int` of the data points in the cluster as value.
    '''
    clusters = {}
    for unique_label in list(set(optics_clustering_labels)):
        clusters[unique_label] = []

    for (key, value) in clusters.items():
        clusters[key] = [i for i, label in enumerate(optics_clustering_labels) if label==key]

    return clusters