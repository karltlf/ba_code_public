from tslearn.metrics import dtw_path_from_metric
import numpy as np
def calculate_dtw_distance_matrix(data:list[list], **dtw_metrics):
    '''

    :param data: List of lists, where each sublist contains list-entries of x-y tuples, e.g. [sublist_1,...,sublist_n] = [[[x_11,y_11],...,[x_1m,y_1m]],...,[[x_n1,y_n1],...,[x_nk,y_nk]]]
    :param **dtw_metrics: One or multiple of the following: `sakoe_chiba_radius`, `metric`, `itakura_max_slope`, `global_constraint`
    '''
    n = len(data)
    dtw_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1,n):
            path, distance  = dtw_path_from_metric(data[i], data[j], **dtw_metrics)
            dtw_matrix[i][j] = distance
            dtw_matrix[j][i] = distance

    return dtw_matrix