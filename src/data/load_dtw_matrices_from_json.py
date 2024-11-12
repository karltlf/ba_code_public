import json
import numpy as np


def load_dtw_matrices_from_json(file_path):
    """
    Load the DTW distance matrices from a JSON file.

    :param file_path: The path to the JSON file containing DTW matrices.
    :return: A dictionary where the keys are parameter names and the values are the DTW matrices (as NumPy arrays).
    """
    with open(file_path, 'r') as json_file:
        dtw_matrices_dict = json.load(json_file)

    # Convert the matrices from lists back to NumPy arrays
    dtw_matrices_dict = {key: np.array(matrix) for key, matrix in dtw_matrices_dict.items()}

    return dtw_matrices_dict