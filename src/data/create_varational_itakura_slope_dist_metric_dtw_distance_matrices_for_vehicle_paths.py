import os
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
code_root_dir = os.path.abspath(os.path.join(current_file_dir, '../..'))
sys.path.append(code_root_dir)
print(f'Appending to sys.path: {code_root_dir}')


import numpy as np
import pandas as pd
import ast
from sklearn.cluster import OPTICS
from src.features.get_first_and_last_x_y_coordinates import get_first_and_last_x_y_coordinates
from src.features.get_x_y_tuple_list import get_x_y_tuple_list

# define intersections
intersection_names = ['k729_2022', 'k733_2020', 'k733_2018']

for intersection_name in intersection_names:
    
    # Load the data
    data_path = f'{code_root_dir}/data/processed/{intersection_name}_cuid.csv'
    df_cuid = pd.read_csv(data_path)
    df_cuid_grouped_path = data_path.replace('.csv', '_grouped.csv')
    df_cuid_grouped = pd.read_csv(df_cuid_grouped_path)
    df_cuid_grouped['x'] = df_cuid_grouped['x'].apply(lambda x: ast.literal_eval(x))
    df_cuid_grouped['y'] = df_cuid_grouped['y'].apply(lambda y: ast.literal_eval(y))
    list_x_y_tuples = get_x_y_tuple_list(df_cuid_grouped, ['x','y'])
    first_last_x_coords, first_last_y_coords = get_first_and_last_x_y_coordinates(list_x_y_tuples)
    X = np.array([first_last_x_coords, first_last_y_coords]).T

    # create variation parameters
    dtw_metrics_no_add_params = ['euclidean', 'sqeuclidean', 'chebyshev', 'manhattan']
    itakura_max_slope_range = np.arange(1, 5.1, 0.1)

    # create & save dtw matrices
    from src.models.dtw.calculate_dtw_matrices_parallelized import calculate_dtw_matrices_parallelized
    results, dtw_matrix_dict = calculate_dtw_matrices_parallelized(list_x_y_tuples, dtw_metrics_no_add_params, itakura_max_slope_range, results_file_path=f'{code_root_dir}/data/processed/{intersection_name}_diff_itakura_slope_dtw_matrices.json')
