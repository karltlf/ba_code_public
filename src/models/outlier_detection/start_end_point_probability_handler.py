import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
target_dir_name = 'ba_code_project'
while True:
    # Check if the target directory exists in the current directory
    potential_target = os.path.join(current_dir, target_dir_name)
    if os.path.isdir(potential_target):
        code_root_dir = potential_target
        break
    # Move one level up
    parent_dir = os.path.dirname(current_dir)
    # If we're at the root of the file system and still haven't found it, stop
    if parent_dir == current_dir:
        code_root_dir = None
        break
    current_dir = parent_dir
if code_root_dir:
    # Add the found target directory to sys.path
    sys.path.append(code_root_dir)
    print(f'Found and appended to sys.path: {code_root_dir}')
else:
    print(f'Target directory not found.')

import json
import numpy as np
import pandas as pd
import ast
from sklearn.cluster import OPTICS
from src.features.get_first_and_last_x_y_coordinates import get_first_and_last_x_y_coordinates
from src.features.get_x_y_tuple_list import get_x_y_tuple_list
from src.models.DISTANCE_METRICS_WITH_ADDITIONAL_ARGS import DISTANCE_METRICS_WITH_ADDITIONAL_ARGS
from scipy.spatial.distance import cdist
from src.models.ensure_distance_metric_params import ensure_distance_metric_params

# Import additional libraries for KDE and percentile calculation
from sklearn.neighbors import KernelDensity
from scipy.stats import percentileofscore

def start_end_point_probability_handler(intersection_name:str, eval_metric:str, point_list:list[list[float]]) -> dict:
    
    data_path = f'{code_root_dir}/data/processed/{intersection_name}_cuid.csv'
    df_cuid = pd.read_csv(data_path)
    df_cuid_grouped_path = data_path.replace('.csv', '_grouped.csv')
    df_cuid_grouped = pd.read_csv(df_cuid_grouped_path)
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(lambda x: ast.literal_eval(x))
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(lambda y: ast.literal_eval(y))
    list_x_y_tuples = get_x_y_tuple_list(df_cuid_grouped, ['x','y'])
    first_last_x_coords, first_last_y_coords = get_first_and_last_x_y_coordinates(list_x_y_tuples)
    X = np.array([first_last_x_coords, first_last_y_coords]).T

    start_point = [point_list[0][0], point_list[0][1]]
    end_point = [point_list[-1][0], point_list[-1][1]]
    # **OPTICS Implementation Starts Here**

    # Define OPTICS params
    metric = eval_metric

    # Get optimized optics params
    optics_params_path = f'{code_root_dir}/src/models/OUTLIER_DETECTION/OPTICS_MODELS/{intersection_name}_optics_optimized_params.json'

    with open(f'{optics_params_path}') as f:
        optimization_parameters = json.load(f)

    optics_params = optimization_parameters[metric]

    min_samples = optics_params['min_samples']
    max_eps = optics_params['epsilon']
    distance_metric = optics_params['metric']
    cluster_method = optics_params['cluster_method']
    xi = optics_params['xi']

    # Define additional parameters for distance metric if needed
    if distance_metric in DISTANCE_METRICS_WITH_ADDITIONAL_ARGS:
        metric_params = ensure_distance_metric_params(X, [distance_metric])[distance_metric]
    else:
        metric_params = {}

    # Define and fit OPTICS model
    if cluster_method == 'dbscan':
        optics_model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=distance_metric,
            cluster_method=cluster_method,
            metric_params=metric_params
        )
    else:
        optics_model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=distance_metric,
            xi=xi,
            metric_params=metric_params
        )

    # Fit model
    optics_model.fit(X)

    # Calculate medoids
    labels = optics_model.labels_
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    medoids = {}

    for label in unique_labels:
        cluster_points = X[labels == label]
        distances = cdist(cluster_points, cluster_points, 'euclidean')
        medoid_index = np.argmin(np.sum(distances, axis=1))  # Find the point with the smallest sum of distances
        medoids[label] = cluster_points[medoid_index]

    medoids = np.array(list(medoids.values()))


    # Calculate cluster with lowest distance to medoid for start and end point
    distances_to_medoids_start = cdist([start_point], medoids, 'euclidean')
    closest_medoid_cluster_index_start = np.argmin(distances_to_medoids_start)

    distances_to_medoids_end = cdist([end_point], medoids, 'euclidean')
    closest_medoid_cluster_index_end = np.argmin(distances_to_medoids_end)

    if closest_medoid_cluster_index_start == closest_medoid_cluster_index_end:
        same_cluster_penalty = True
    else:
        same_cluster_penalty = False

    # # Calculate convex hulls
    # def calculate_cluster_convex_hulls(X, optics_model, return_type="ndarray"):

    #     pattern = r"(conv|convex)[\s_-]*(hull)?|obj(?:ect)?"

    #     hulls = {}
    #     hulls_ConvHullObj = []
    #     labels = optics_model.labels_
    #     unique_labels = np.unique(labels[labels != -1])
    #     for label in unique_labels:
    #         cluster_points = X[labels == label]
    #         if len(cluster_points) >= 3:  # Convex hull needs at least 3 points
    #             hull = ConvexHull(cluster_points)
    #             hulls_ConvHullObj.append(hull)
    #             hulls[label] = cluster_points[hull.vertices]

    #     if return_type == "ndarray":
    #         # Return a list of arrays (one array per cluster)
    #         return [hull for hull in hulls.values()]
    #     # Perform a case-insensitive search for the pattern in the return_type string
    #     if bool(re.search(pattern, return_type, re.IGNORECASE)):
    #         # Return a list of ConvexHull objects (one object per cluster)
    #         return [hull for hull in hulls.values()], hulls_ConvHullObj
    #     else:
    #         return hulls

    # hulls_as_points, hulls_as_obj = calculate_cluster_convex_hulls(X, optics_model, return_type="obj")

    # # Check if point inside hull
    # def is_point_in_any_hull(point, hulls):
    #     """
    #     Use halfspace equations and see if point satisfies them to check if a point is inside any hull.
    #     """
    #     point = np.array(point)

    #     # Check if the point is inside any hull by using the equations of each hull
    #     for hull in hulls:
    #         equations = hull.equations
    #         # Debug: Print equations
    #         # print(f'{[eq for eq in equations]}')
    #         if all((np.dot(eq[:-1], point) + eq[-1]) <= 0 for eq in equations):
    #             return True

    #     return False

    # is_in_hull = is_point_in_any_hull(input_point, hulls_as_obj)

    # **OPTICS Implementation Ends Here**

    # **KDE Implementation Starts Here**

    # load the json file
    with open(f'{code_root_dir}/src/models/OUTLIER_DETECTION/KDE_MODELS/kde_params.json') as f:
        kde_params = json.load(f)
    
    kde_params = kde_params[intersection_name]

    # Initialize and fit the KDE model
    # You can adjust the bandwidth parameter as needed
    kde_bandwidth = kde_params['bandwidth']  # Example value; consider tuning this parameter
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel=kde_params['kernel'])
    kde.fit(X)

    # Compute log densities for all training points
    log_density_train = kde.score_samples(X)
    density_train = np.exp(log_density_train)

    # Prepare the input point for KDE (needs to be 2D array)
    input_point_array_start = np.array([[start_point[0], start_point[1]]])
    input_point_array_end = np.array([[end_point[0], end_point[1]]])

    # Compute the log density of the input point
    log_density_start = kde.score_samples(input_point_array_start)
    log_density_end = kde.score_samples(input_point_array_end)
    density_start = np.exp(log_density_start)[0]  # Convert log density to actual density
    density_end = np.exp(log_density_end)[0]  # Convert log density to actual density

    # **Probabilistic Outlier Scoring Starts Here**

    # Compute the percentile of the input point's density within the training densities
    # Lower density corresponds to higher outlierness
    percentile_start = percentileofscore(density_train, density_start, kind='rank')  # 'rank' counts <=
    outlier_score_start = 1 - (percentile_start / 100) # Normalize to [0,1], where 1 is most likely an outlier

    percentile_end = percentileofscore(density_train, density_end, kind='rank')  # 'rank' counts <=
    outlier_score_end = 1 - (percentile_end / 100) # Normalize to [0,1], where 1 is most likely an outlier

    print(f'Percentile of start point density: {percentile_start}%')
    print(f'Percentile of end point density: {percentile_end}%')
    if same_cluster_penalty:
        outlier_score_start = 1
        outlier_score_end = 1

    # **Probabilistic Outlier Scoring Ends Here**

    # **KDE Implementation Ends Here**

    # Print KDE results
    print(f'Density of start point: {density_start}')
    print(f'Density percentile: {percentile_start}%')
    print(f'Outlier score based on KDE: {outlier_score_start}')

    print(f'Density of end point: {density_end}')
    print(f'Density percentile: {percentile_end}%')
    print(f'Outlier score based on KDE: {outlier_score_end}')



    # Optionally, you can return these values or integrate them further into your logic
    return {
        'outlier_score_start': outlier_score_start,
        'outlier_score_end': outlier_score_end,
        'density_start': density_start,
        'density_end': density_end,
        'density_start_percentile': percentile_start,
        'density_end_percentile': percentile_end,
        'same_cluster_penalty': same_cluster_penalty
    }