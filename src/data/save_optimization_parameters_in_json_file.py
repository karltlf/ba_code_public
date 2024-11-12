import json
import os
import numpy as np

def save_optimization_parameters_in_json_file(file_path: str, **kwargs):
    """
    Converts the provided keyword arguments (parameters) into a JSON format and saves them to the specified file path.
    
    :param file_path: The file path to save the parameters to.
    """

    # Convert any np.ndarray, np.int64, np.float64 values in kwargs to make them JSON serializable
    def convert_numpy_types(value):
        if isinstance(value, np.ndarray):
            return value.tolist()  # Convert ndarray to list
        elif isinstance(value, (np.int64, np.int32)):
            return int(value)  # Convert numpy int to Python int
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)  # Convert numpy float to Python float
        elif isinstance(value, dict):
            return {k: convert_numpy_types(v) for k, v in value.items()}  # Recursively convert for dicts
        elif isinstance(value, list):
            return [convert_numpy_types(v) for v in value]  # Recursively convert for lists
        else:
            return value

    # Convert the kwargs before saving
    kwargs_converted = {k: convert_numpy_types(v) for k, v in kwargs.items()}

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the dictionary (kwargs_converted) to the file as JSON
    with open(file_path, 'w') as json_file:
        json.dump(kwargs_converted, json_file, indent=4)

    print(f"Parameters saved to {file_path}")