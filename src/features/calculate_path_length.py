# PASTE THIS TO THE FIRST CELL OF THE NOTEBOOK IN ORDER TO HAVE WORKING IMPORTS
import sys
import os
current_dir = os.getcwd()
parent_parent_dir = os.path.abspath(os.path.join(current_dir, '../..')) # tweak so that you get the root project folder

sys.path.append(parent_parent_dir)
import math
import pandas as pd
from src.features.get_x_y_tuple_list import get_x_y_tuple_list


def calculate_path_length(x_y_tuple_lists:list[list]) -> list:
    """
    Calculate the total length of a path or paths.
    
    :param x_y_tuple_lists: A list of lists of tuples. Each list of tuples represents a path.
    :return path_lengths: The total length of the path or paths.
    """

    path_lengths = []
    for path in x_y_tuple_lists:
        total_length = 0
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += distance
        path_lengths.append(total_length)
    return path_lengths


def calculate_path_length_from_df(df_grouped:pd.DataFrame) -> list:
    """
    Calculate the total length of a path or paths.
    
    :param df_grouped: Grouped dataframe with list of x and y coordinates for each row in the dataframe.
    :return path_lengths: The total length of the path or paths.
    """

    list_x_y_tuples = get_x_y_tuple_list(df_grouped, ['x', 'y'])

    return calculate_path_length(list_x_y_tuples)