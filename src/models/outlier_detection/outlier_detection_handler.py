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

import numpy as np
from src.models.OUTLIER_DETECTION.start_end_point_probability_handler import start_end_point_probability_handler
from src.models.OUTLIER_DETECTION.vehicle_path_probability_handler import vehicle_path_probability_handler


def outlier_detection_handler(intersection_name: str, 
                                     id_intersection_optics_optimization_init_params: str, 
                                     id_in_cluster_optics_optimization_init_params: str,
                                     intersection_eval_metric: str,
                                     in_cluster_eval_metric: str,
                                     tuple_list: list[list[float]]
                                     ) -> tuple[dict, dict]:
    
    x_input, y_input = zip(*tuple_list)
    start_end_results = start_end_point_probability_handler(intersection_name, intersection_eval_metric, tuple_list)
    path_results = vehicle_path_probability_handler(intersection_name,
                                                    id_intersection_optics_optimization_init_params,
                                                    id_in_cluster_optics_optimization_init_params,
                                                    intersection_eval_metric,
                                                    in_cluster_eval_metric,
                                                    x_input,
                                                    y_input)
    
    outlier_score_start = start_end_results['outlier_score_start']
    outlier_score_end = start_end_results['outlier_score_end']
    outlier_score_path = path_results['outlier_probability_score']
    path_best_match = path_results['best_match']
    best_match_cluster_id = path_best_match[0]
    best_match_subcluster_id = path_best_match[1]
    best_match_dtw_distance = path_best_match[3]


    weight_start_end = 0.15
    weight_path = 0.7
    # calculate deviation of start and end point
    if np.abs(outlier_score_start - outlier_score_end) > 0.4:
        weight_start_end = 0.3
        weight_path = 0.4
    elif outlier_score_end == 1 and outlier_score_start == 1: # if start and end point are the same, we say it is for sure an outlier
        outlier_score_path = 1

    # use weighted average
    outlier_score = weight_path*outlier_score_path + weight_start_end*(outlier_score_start + outlier_score_end)

    results_dict = {'outlier_score': outlier_score, 
                    'outlier_score_start': outlier_score_start, 
                    'outlier_score_end': outlier_score_end, 
                    'outlier_score_path': outlier_score_path,
                    'best_match_cluster_id': best_match_cluster_id,
                    'best_match_subcluster_id': best_match_subcluster_id,
                    'best_match_dtw_distance': best_match_dtw_distance
                    }
    return results_dict

