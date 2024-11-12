import os
import sys
current_dir = os.getcwd()
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../..')) # tweak so that you get dir of code project
sys.path.append(parent_parent_dir)

import json
import numpy as np
import pandas as pd
import ast
from sklearn.cluster import OPTICS
from src.features.get_first_and_last_x_y_coordinates import get_first_and_last_x_y_coordinates
from src.features.get_x_y_tuple_list import get_x_y_tuple_list
from src.models.DISTANCE_METRICS_WITH_ADDITIONAL_ARGS import DISTANCE_METRICS_WITH_ADDITIONAL_ARGS
from src.models.ensure_distance_metric_params import ensure_distance_metric_params
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import re



def calculate_outlier_probability_for_start_end_points(intersection_name:str, x_input:float, y_input:float):

    class OutlierProbability:
        def __init__(self, in_hull:bool, medoid_distance_similarity:float, hulls:list[ConvexHull], hulls_as_points:list, medoids):
            self.in_hull = in_hull
            self.medoid_distance_similarity = medoid_distance_similarity
            self.hulls = hulls
            self.hulls_as_points = hulls_as_points
            self.medoids = medoids
    

    # -----------------------------------------
    # ----------------- IMPORTS ---------------
    # -----------------------------------------
    # define training data
    data_path = f'{parent_parent_dir}/data/processed/{intersection_name}_cuid.csv'

    df_cuid = pd.read_csv(data_path)
    df_cuid_grouped_path = data_path.replace('.csv', '_grouped.csv')
    df_cuid_grouped = pd.read_csv(df_cuid_grouped_path)
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(lambda x: ast.literal_eval(x))
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(lambda y: ast.literal_eval(y))
    list_x_y_tuples = get_x_y_tuple_list(df_cuid_grouped, ['x','y'])
    first_last_x_coords, first_last_y_coords = get_first_and_last_x_y_coordinates(list_x_y_tuples)
    X = np.array([first_last_x_coords, first_last_y_coords]).T

    # -----------------------------------------
    # ------- START END POINT OPTICS ----------
    # -----------------------------------------

    # define optics params
    metric = 'silhouette' # eval metric for which params are optimized


    # get optimized optics params
    optics_params_path = f'{parent_parent_dir}/src/models/OUTLIER_DETECTION/OPTICS_MODELS/{intersection_name}_optics_optimized_params.json'

    with open(f'{optics_params_path}') as f:
        optimization_parameters = json.load(f)

    optics_params = optimization_parameters[metric]

    min_samples = optics_params['min_samples']
    max_eps = optics_params['epsilon']
    distance_metric = optics_params['metric']
    cluster_method = optics_params['cluster_method']
    xi = optics_params['xi']


    # define additional parameters for distance metric if needed
    if distance_metric in DISTANCE_METRICS_WITH_ADDITIONAL_ARGS:
        metric_params = ensure_distance_metric_params(X, [distance_metric])[distance_metric]
    else:
        metric_params = {}


    # define model
    if cluster_method=='dbscan':
        optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=distance_metric, cluster_method=cluster_method, metric_params=metric_params)
    else:
        optics_model = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=distance_metric, xi=xi, metric_params=metric_params)


    # fit model
    optics_model.fit(X)


    # calculate medoids
    labels = optics_model.labels_
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise points (-1)
    medoids = {}

    for label in unique_labels:
        cluster_points = X[labels == label]
        distances = cdist(cluster_points, cluster_points, 'euclidean')
        medoid_index = np.argmin(np.sum(distances, axis=1))  # Find the point with the smallest sum of distances
        medoids[label] = cluster_points[medoid_index]

    medoids = np.array(list(medoids.values()))


    # calculate average max of in cluster distance
    max_in_cluster_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        max_distance = np.max(cdist(cluster_points, cluster_points, 'euclidean'))
        max_in_cluster_distances.append(max_distance)
    

    # calculate cluster with lowest distance to medoid for x_input and y_input
    input_point = [x_input, y_input]
    distances_to_medoids = cdist([input_point], medoids, 'euclidean')
    closest_medoid_cluster_index = np.argmin(distances_to_medoids)
    closest_medoid_distance = distances_to_medoids[0][closest_medoid_cluster_index]
    

    # get cluster index of second to lowest distance to medoid
    distances_to_medoids[0][closest_medoid_cluster_index] = np.inf
    second_closest_medoid_medoid_index = np.argmin(distances_to_medoids)
    second_clostes_medoid_distance = distances_to_medoids[0][second_closest_medoid_medoid_index]


    # calculate lowest and second lowest distance similarity
    print(f'Lowest distance to medoid: {closest_medoid_distance}')
    print(f'Second lowest distance to medoid: {second_clostes_medoid_distance}')


    # calculate similarity of lowest and second lowest distance
    distance_similarity = second_clostes_medoid_distance / closest_medoid_distance # ≥ 1
    

    # calcuate hulls
    def calculate_cluster_convex_hulls(X, optics_model, return_type="ndarray"):

        pattern = r"(conv|convex)[\s_-]*(hull)?|obj(?:ect)?"
    
        hulls = {}
        hulls_ConvHullObj = []
        labels = optics_model.labels_
        unique_labels = np.unique(labels[labels != -1])
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) >= 3:  # Convex hull needs at least 3 points
                hull = ConvexHull(cluster_points)
                hulls_ConvHullObj.append(hull)
                hulls[label] = cluster_points[hull.vertices]
        
        if return_type == "ndarray":
            # Return a list of arrays (one array per cluster)
            return [hull for hull in hulls.values()]
        # Perform a case-insensitive search for the pattern in the return_type string
        if bool(re.search(pattern, return_type, re.IGNORECASE)):
            # Return a list of ConvexHull objects (one object per cluster)
            return [hull for hull in hulls.values()], hulls_ConvHullObj
        else:
            return hulls

    hulls_as_points, hulls_as_obj = calculate_cluster_convex_hulls(X, optics_model, return_type="obj")


    # check if point inside hull
    def is_point_in_any_hull(point, hulls):
        """
        Use halfspace equations and see if point satisfies them to check if a point is inside any hull, see also 
        https://stackoverflow.com/questions/26809630/how-to-convert-the-half-spaces-that-constitute-a-convex-hull-to-a-set-of-extreme

        """
        point = np.array(point)

        # Check if the point is inside any hull by using the equations of each hull
        for hull in hulls:
            equations = hull.equations
            print(f'{[eq for eq in equations]}')
            if all((np.dot(eq[:-1], point) + eq[-1]) <= 0 for eq in equations):
                return True
        
        return False
    
    is_in_hull = is_point_in_any_hull(input_point, hulls_as_obj)


    # do some calculation with this information for the outlier probability
    # ------------------ TBD ------------------

    # ----------------------------------
    # -------- DTW lane fitting --------
    # ----------------------------------

    

    

    return OutlierProbability(is_in_hull, distance_similarity, hulls_as_obj, hulls_as_points, medoids)