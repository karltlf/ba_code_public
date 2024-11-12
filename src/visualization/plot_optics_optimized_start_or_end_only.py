from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

from src.features.get_x_y_tuples import * 
from sklearn.cluster import OPTICS
from matplotlib.gridspec import GridSpec
from src.visualization.get_cluster_colors import get_cluster_colors
from src.visualization.get_cluster_legend_handles import get_cluster_legend_handles
from src.visualization.format_optimization_input_params_to_text import *
from src.visualization.format_clustering_parameter_dict_to_text import *
from src.models.clustering_optimization.optimize_optics_parallelized import *
from src.models.clustering_optimization.optimize_optics_parallelized_punish_non_edge_points import *
from src.features.get_first_and_last_x_y_coordinates import *


def plot_optics_optimized_start_or_end_only(df_path:str, max_eps_range:np.array, min_samples_range:np.array, metrics:list[str], cluster_methods:list[str], xis:np.array, get_first_or_last_coords_function, **kwargs):
    ''' 
    Plots optimized OPTICS clustering results for a given dataframe using only the start or end points of the trajectories.
    Evaluates clustering based on silhouette, calinski_harabasz, davies_bouldin scores.

    :param df_path: Path to DataFrame
    :param max_eps_range: Maximum epsilon (distance allowed between two points) range, typically 0-15 for this dataset
    :param min_samples_range: Minimum number of points per cluster, typically 2-15 for this dataset
    :param metrics: Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    :param cluster_methods: `xi` or `dbscan`. The OPTICS algorithm can be used with the `xi` cluster method, which is a variant of DBSCAN that uses the `xi` metric to determine the minimum number of points in a cluster. The `dbscan` cluster method uses the DBSCAN algorithm to determine the minimum number of points in a cluster.
    :param get_first_or_last_coords_function: Function to get either the first or last coordinates of the trajectories
    :param xis: Range of xi values to use for the `xi` cluster method. This parameter is only used if `cluster_method` is set to `xi`.
    :param **kwargs: Additional keyword arguments for the distance metrics
    '''

    # 1 LOAD DATA
    df_cuid = pd.read_csv(df_path)
    df_cuid_grouped_path = df_path.replace('.csv', '_grouped.csv')
    df_cuid_grouped = pd.read_csv(df_cuid_grouped_path)
    # 1.1 CONVERT FEATURES TO NUMBERS
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(lambda x: ast.literal_eval(x))
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(lambda y: ast.literal_eval(y))
    # 1.2 CREATE FEATURE MATRIX
    list_x_y_tuples = get_x_y_tuple_list(df_cuid_grouped, ['x','y'])
    first_or_last_x_coords, first_or_last_y_coords = get_first_or_last_coords_function(list_x_y_tuples)
    X = np.array([first_or_last_x_coords, first_or_last_y_coords]).T

    
    
    # 2 OPTIMIZE OPTICS
    # 2.1 OPTIMIZE OPTICS
    metric_params = ensure_distance_metric_params(X, metrics, **kwargs)
    optics_optimization_results = optimize_optics_parallelized(X, max_eps_range, min_samples_range, metrics, cluster_methods, xis, **metric_params)
    # 2.2 GET OPTIMIZED PARAMETERS
    _, epsilon_optimized_silhouette, min_samples_optimized_silhouette, metric_optimized_silhouette, cluster_method_optimized_silhouette, xi_optimized_silhouette = optics_optimization_results['silhouette'].values()
    _, epsilon_optimized_calinski_harabasz, min_samples_optimized_calinski_harabasz, metric_optimized_calinski_harabasz, cluster_method_optimized_calinski_harabasz, xi_optimized_calinski_harabasz = optics_optimization_results['calinski_harabasz'].values()
    _, epsilon_optimized_davies_bouldin, min_samples_optimized_davies_bouldin, metric_optimized_davies_bouldin, cluster_method_optimized_davies_bouldin, xi_optimized_davies_bouldin = optics_optimization_results['davies_bouldin'].values()
    

    
    
    # 3 FIT OPTIMIZED OPTICS MODELS   
    # 3.1 SILHOUETTE OPTIMIZED OPTICS MODEL 
    labels_silhouette, centers_silhouette = fit_optics(epsilon_optimized_silhouette, min_samples_optimized_silhouette, X, metric_optimized_silhouette, cluster_method_optimized_silhouette, xi_optimized_silhouette, **kwargs)

    # 3.2 CALINSKI-HARABASZ OPTIMIZED OPTICS MODEL
    labels_calinski_harabasz, centers_calinski_harabasz = fit_optics(epsilon_optimized_calinski_harabasz, min_samples_optimized_calinski_harabasz, X, metric_optimized_calinski_harabasz, cluster_method_optimized_calinski_harabasz, xi_optimized_calinski_harabasz, **kwargs)

    # 3.3 DAVIES-BOULDIN OPTIMIZED OPTICS MODEL
    labels_davies_bouldin, centers_davies_bouldin = fit_optics(epsilon_optimized_davies_bouldin, min_samples_optimized_davies_bouldin, X, metric_optimized_davies_bouldin, cluster_method_optimized_davies_bouldin, xi_optimized_davies_bouldin, **kwargs)


    
    
    # 4 PLOT OPTICS OPTIMIZED RESULTS
    point_colors_start_end_silhouette_optimized, colors_start_end_silhouette_optimized = get_cluster_colors(labels_silhouette)
    legend_handles_start_end_silhouette_optimized = get_cluster_legend_handles(colors_start_end_silhouette_optimized, labels_silhouette)

    point_colors_start_end_calinski_harabasz_optimized, colors_start_end_calinski_harabasz_optimized = get_cluster_colors(labels_calinski_harabasz)
    legend_handles_start_end_calinski_harabasz_optimized = get_cluster_legend_handles(colors_start_end_calinski_harabasz_optimized, labels_calinski_harabasz)

    point_colors_start_end_davies_bouldin_optimized, colors_start_end_davies_bouldin_optimized = get_cluster_colors(labels_davies_bouldin)
    legend_handles_start_end_davies_bouldin_optimized = get_cluster_legend_handles(colors_start_end_davies_bouldin_optimized, labels_davies_bouldin)

    optimization_parameters_description = format_optimization_input_params_to_text(
        max_eps_range=max_eps_range, 
        min_samples_range=min_samples_range, 
        metric=metrics, 
        cluster_method=cluster_methods, 
        xi=xis if 'xi' in cluster_methods else 'not applicable',
        **kwargs
    )

    clustering_description = format_clustering_parameter_dict_to_text(optics_optimization_results)

    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0]) 
    ax1.scatter(first_or_last_x_coords, first_or_last_y_coords, c=point_colors_start_end_silhouette_optimized)
    ax1.scatter(centers_silhouette[:, 0], centers_silhouette[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
    ax1.set_title(f'Silhouette optimized clustering', fontsize=15)
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.legend(handles=legend_handles_start_end_silhouette_optimized)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(first_or_last_x_coords, first_or_last_y_coords, c=point_colors_start_end_calinski_harabasz_optimized)
    ax2.scatter(centers_calinski_harabasz[:, 0], centers_calinski_harabasz[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
    ax2.set_title(f'Calinski-Harabasz optimized clustering', fontsize=15)
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Y-axis")
    ax2.legend(handles=legend_handles_start_end_calinski_harabasz_optimized)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(first_or_last_x_coords, first_or_last_y_coords, c=point_colors_start_end_davies_bouldin_optimized)
    ax3.scatter(centers_davies_bouldin[:, 0], centers_davies_bouldin[:, 1], c='red', s=200, alpha=0.75, label='Cluster Centers')
    ax3.set_title(f'Davies-Bouldin optimized clustering', fontsize=15)
    ax3.set_xlabel("X-axis")
    ax3.set_ylabel("Y-axis")
    ax3.legend(handles=legend_handles_start_end_davies_bouldin_optimized)


    props_results = dict(boxstyle='square', facecolor='green', alpha=0.15)
    fig.text(0.012,-0.3,f'{clustering_description}', fontsize=12, bbox=props_results)
    props_optimization_parameters = dict(boxstyle='square', facecolor='blue', alpha=0.15)
    fig.text(0.012,-0.55,f'Optimization Parameters\n{optimization_parameters_description}', fontsize=12, bbox=props_optimization_parameters)
    plt.suptitle(f'Optimized OPTICS', fontsize=20)


    # Display the plots
    plt.tight_layout()
    plt.show()

    return optics_optimization_results

def get_metric_params(metric, X, **kwargs):
    if metric == 'seuclidean':
        metric_params = {'V': np.var(X, axis=0)}
    elif metric == 'mahalanobis':
        VI = np.linalg.inv(np.cov(X.T))
        metric_params = {'VI': VI}
    elif metric == 'minkowski':
        metric_params = {'p': kwargs.get('p', 2)}
    else:
        metric_params = None
    return metric_params

def fit_optics(epsilon, min_samples, X, metric, cluster_method, xi, **kwargs):
        metric_params = get_metric_params(metric, X, **kwargs)
        if metric_params is None:
            if cluster_method == 'xi':
                optics = OPTICS(eps=epsilon, min_samples=min_samples, metric=metric, cluster_method=cluster_method, xi=xi)
            else:
                optics = OPTICS(eps=epsilon, min_samples=min_samples, metric=metric, cluster_method=cluster_method)
        else:
             optics = OPTICS(eps=epsilon, min_samples=min_samples, metric=metric, cluster_method=cluster_method, metric_params=metric_params)
        optics.fit(X)
        labels = optics.labels_
        unique_labels = np.unique(labels[labels != -1])
        centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])
        return labels, centers

